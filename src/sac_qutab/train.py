from __future__ import annotations

import contextlib
import io
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import median
import time
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler, Subset

from .config import ResolvedRun, config_digest
from .data import ManifestDataset, validate_split_contract
from .models import ModelOutput, build_model, model_metadata
from .runtime import JsonlLogger, atomic_write_bytes, atomic_write_json, capture_rng_state, environment_metadata, restore_rng_state, seed_everything, sha256_file
from .tracking import checkpoint_wandb_run_id, init_wandb_tracker


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    initialized_here: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(backend: str, timeout_seconds: int) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():  # type: ignore[possibly-missing-attribute]
        selected = backend
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA; use gloo only for bounded CPU wiring")
        dist.init_process_group(selected, timeout=timedelta(seconds=timeout_seconds))  # type: ignore[possibly-missing-attribute]
        initialized_here = True
    return DistributedContext(rank, world_size, local_rank, initialized_here)


def close_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():  # type: ignore[possibly-missing-attribute]
        dist.destroy_process_group()  # type: ignore[possibly-missing-attribute]


class GlobalBatchShardSampler(Sampler[int]):
    """No-duplicate sampler that drops one incomplete global batch consistently."""

    def __init__(self, size: int, global_batch: int, rank: int, world_size: int, seed: int) -> None:
        if global_batch % world_size:
            raise ValueError("global batch must divide by world size")
        self.size = size
        self.global_batch = global_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        usable = len(order) // self.global_batch * self.global_batch
        shard = self.global_batch // self.world_size
        selected = []
        for start in range(0, usable, self.global_batch):
            selected.extend(order[start + self.rank * shard : start + (self.rank + 1) * shard])
        return iter(selected)

    def __len__(self) -> int:
        return (self.size // self.global_batch * self.global_batch) // self.world_size


@dataclass
class Progress:
    epoch: int = 0
    next_batch: int = 0
    global_step: int = 0
    best_score: float = -math.inf
    best_epoch: int = -1
    bad_epochs: int = 0


def _device(run: ResolvedRun, context: DistributedContext) -> torch.device:
    requested = run.config.train.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(context.local_rank)
        return torch.device("cuda", context.local_rank)
    if requested != "cpu":
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device("cpu")


def _resolved_amp_dtype(config_dtype: str, device: torch.device) -> str:
    if device.type == "cpu" or config_dtype == "none":
        return "none"
    if config_dtype == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if config_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 configured but unsupported by this CUDA device; use amp_dtype=auto or fp16")
    return config_dtype


def _autocast(resolved_dtype: str, device: torch.device):
    if resolved_dtype == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if resolved_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_ratio: float) -> LambdaLR:
    if total_steps <= warmup_steps:
        raise ValueError("total optimizer steps must exceed warmup steps")
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(progress, 1.0)))
    return LambdaLR(optimizer, multiplier)


def _checkpoint_bytes(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def save_checkpoint(path: Path, run: ResolvedRun, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR, scaler: torch.amp.GradScaler | None, progress: Progress, context: DistributedContext, dataset_hashes: dict[str, str], pretrained_checkpoint_sha256: str | None = None, wandb_run_id: str | None = None) -> None:
    local_rank_state = {"rank": context.rank, "rng": capture_rng_state(), "epoch": progress.epoch, "next_batch": progress.next_batch, "global_step": progress.global_step}
    rank_states: list[Any]
    if context.world_size > 1:
        rank_states = [None] * context.world_size
        dist.all_gather_object(rank_states, local_rank_state)  # type: ignore[possibly-missing-attribute]
    else:
        rank_states = [local_rank_state]
    if context.is_main:
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        payload = {
            "schema_version": "sac-checkpoint-v2",
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "progress": progress.__dict__.copy(),
            "rank_states": rank_states,
            "world_size": context.world_size,
            "global_batch": run.effective_global_batch,
            "resolved_run": run.canonical(),
            "resolved_run_sha256": config_digest(run),
            "dataset_hashes": dataset_hashes,
            "pretrained_checkpoint_sha256": pretrained_checkpoint_sha256,
            "tracking": {"wandb_run_id": wandb_run_id, "optimizer_step": progress.global_step},
        }
        atomic_write_bytes(path, _checkpoint_bytes(payload))
    if context.world_size > 1:
        dist.barrier()  # type: ignore[possibly-missing-attribute]


def load_checkpoint(path: Path, run: ResolvedRun, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR, scaler: torch.amp.GradScaler | None, context: DistributedContext, dataset_hashes: dict[str, str], pretrained_checkpoint_sha256: str | None = None) -> Progress:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "sac-checkpoint-v2":
        raise ValueError("unsupported checkpoint schema")
    if payload["world_size"] != context.world_size or payload["global_batch"] != run.effective_global_batch:
        raise ValueError("exact resume requires the same world size and global-batch contract")
    if (payload["resolved_run_sha256"] != config_digest(run) or payload["dataset_hashes"] != dataset_hashes
            or payload.get("pretrained_checkpoint_sha256") != pretrained_checkpoint_sha256):
        raise ValueError("checkpoint config/data/pretrained-weight provenance mismatch")
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint is missing the configured scaler")
        scaler.load_state_dict(payload["scaler"])
    rank_state = payload["rank_states"][context.rank]
    if rank_state["rank"] != context.rank:
        raise ValueError("rank-keyed RNG state mismatch")
    restore_rng_state(rank_state["rng"])
    return Progress(**payload["progress"])


def _loader_kwargs(num_workers: int, pin_memory: bool, *, persistent_workers: bool) -> dict[str, Any]:
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }


def _move_batch_to_device(
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    *,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda" and pin_memory
    return images.to(device, non_blocking=non_blocking), targets.to(device, non_blocking=non_blocking)


def _visible_device_selectors() -> list[str] | None:
    visible_text = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_text is None:
        return None
    return [item.strip() for item in visible_text.split(",") if item.strip()]


def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    context: DistributedContext,
    num_classes: int,
    *,
    pin_memory: bool,
) -> dict[str, float]:
    eval_model = model.module if isinstance(model, DistributedDataParallel) else model
    eval_model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = _move_batch_to_device(images, targets, device, pin_memory=pin_memory)
            output: ModelOutput = eval_model(images)
            loss_sum += nn.functional.cross_entropy(output.logits, targets, reduction="sum")
            count += len(targets)
            pred = output.logits.argmax(1)
            flat = targets * num_classes + pred
            confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    if context.world_size > 1:
        dist.all_reduce(loss_sum)  # type: ignore[possibly-missing-attribute]
        dist.all_reduce(count)  # type: ignore[possibly-missing-attribute]
        dist.all_reduce(confusion)  # type: ignore[possibly-missing-attribute]
    if count.item() == 0:
        raise ValueError("validation dataset is empty")
    tp = confusion.diag()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(torch.finfo(torch.float64).eps)
    return {"loss": float((loss_sum / count).item()), "accuracy": float((tp.sum() / count).item()), "macro_f1": float(f1.mean().item())}


def _projection_summary(
    run: ResolvedRun,
    median_step_seconds: float | None,
    measured_sample_count: int,
    measured_optimizer_steps: int,
    full_train_sample_count: int,
    *,
    measured_validation_sample_count: int = 0,
    full_validation_sample_count: int = 0,
    measured_validation_seconds: float | None = None,
) -> dict[str, Any]:
    projected_steps_per_epoch = full_train_sample_count // run.effective_global_batch
    if projected_steps_per_epoch < 1:
        raise ValueError("full training split is too small for one global batch")
    training_hours = None if median_step_seconds is None else median_step_seconds * projected_steps_per_epoch * run.config.train.epochs / 3600
    projected_validation_seconds = None
    if (measured_validation_seconds is not None and measured_validation_seconds > 0
            and measured_validation_sample_count > 0 and full_validation_sample_count > 0):
        projected_validation_seconds = measured_validation_seconds * full_validation_sample_count / measured_validation_sample_count
    combined_hours = None
    if training_hours is not None and projected_validation_seconds is not None:
        combined_hours = training_hours + projected_validation_seconds * run.config.train.epochs / 3600
    return {
        "measured_sample_count": measured_sample_count,
        "measured_optimizer_steps_rank0": measured_optimizer_steps,
        "projected_full_train_sample_count": full_train_sample_count,
        "projected_optimizer_steps_per_epoch": projected_steps_per_epoch,
        "projected_epoch_cap_hours_from_median_rank0": training_hours,
        "measured_validation_sample_count": measured_validation_sample_count,
        "projected_full_validation_sample_count": full_validation_sample_count,
        "measured_validation_seconds_rank0": measured_validation_seconds,
        "projected_validation_seconds_per_epoch_rank0": projected_validation_seconds,
        "projected_training_validation_hours": combined_hours,
    }


def train_model(
    run: ResolvedRun,
    model: nn.Module,
    train_dataset: ManifestDataset,
    validation_dataset: ManifestDataset,
    run_dir: Path,
    *,
    resume_path: Path | None = None,
    dry_run: bool = False,
    full_train_sample_count: int | None = None,
    full_validation_sample_count: int | None = None,
    pretrained_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    context = init_distributed(run.config.distributed.backend, run.config.distributed.timeout_seconds)
    tracker = None
    succeeded = False
    try:
        if context.world_size != run.world_size:
            raise ValueError("resolved world size differs from torchrun WORLD_SIZE")
        device = _device(run, context)
        resolved_amp_dtype = _resolved_amp_dtype(run.config.train.amp_dtype, device)
        seed_everything(run.seed + context.rank, run.config.train.deterministic)
        model.to(device)
        if context.world_size > 1:
            model = DistributedDataParallel(model, device_ids=[context.local_rank] if device.type == "cuda" else None, broadcast_buffers=False)
        sampler = GlobalBatchShardSampler(len(train_dataset), run.effective_global_batch, context.rank, context.world_size, run.seed)
        pin_memory = run.config.data.pin_memory and device.type == "cuda"
        loader = DataLoader(
            train_dataset,
            batch_size=run.batch_size_per_device,
            sampler=sampler,
            drop_last=False,
            **_loader_kwargs(run.config.data.num_workers, pin_memory, persistent_workers=False),
        )
        validation_indices = list(range(context.rank, len(validation_dataset), context.world_size))
        validation_loader = DataLoader(
            Subset(validation_dataset, validation_indices),
            batch_size=run.batch_size_per_device,
            shuffle=False,
            **_loader_kwargs(run.config.data.num_workers, pin_memory, persistent_workers=True),
        )
        optimizer = AdamW(model.parameters(), lr=run.model.learning_rate, weight_decay=run.config.train.weight_decay)
        optimizer_steps_per_epoch = len(loader) // run.grad_accum_steps
        if optimizer_steps_per_epoch < 1:
            raise ValueError("training split is too small for one global batch")
        epochs = 1 if dry_run else run.config.train.epochs
        max_steps = run.config.train.dry_run_max_steps if dry_run else math.inf
        scheduler = _scheduler(optimizer, min(run.config.train.warmup_epochs * optimizer_steps_per_epoch, max(optimizer_steps_per_epoch * epochs - 1, 0)), optimizer_steps_per_epoch * epochs, run.config.train.min_lr_ratio)
        scaler = torch.amp.GradScaler("cuda", enabled=resolved_amp_dtype == "fp16")
        scaler_or_none = scaler if scaler.is_enabled() else None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        run_started = time.perf_counter()
        optimizer_step_started = run_started
        optimizer_step_times: list[float] = []
        validation_times: list[float] = []
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = JsonlLogger(run_dir / "events.jsonl") if context.is_main else None
        dataset_hashes = {"manifest": sha256_file(run.config.output_path(run.config.data.manifest_relpath)), "splits": sha256_file(run.config.output_path(run.config.data.split_relpath))}
        progress = Progress()
        resume_wandb_run_id = None
        if resume_path:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            resume_wandb_run_id = checkpoint_wandb_run_id(resume_payload)
            progress = load_checkpoint(resume_path, run, model, optimizer, scheduler, scaler_or_none, context, dataset_hashes, pretrained_checkpoint_sha256)
        if context.is_main:
            atomic_write_json(run_dir / "resolved_run.json", run.canonical())
            atomic_write_json(run_dir / "environment.json", environment_metadata())
            atomic_write_json(run_dir / "model.json", model_metadata(model.module if isinstance(model, DistributedDataParallel) else model, run.model, len(run.config.data.class_names), pretrained_checkpoint_sha256))
            tracker = init_wandb_tracker(run, run_dir, resume_run_id=resume_wandb_run_id)
        criterion = nn.CrossEntropyLoss(label_smoothing=run.config.train.label_smoothing)
        stop = False
        for epoch in range(progress.epoch, epochs):
            sampler.set_epoch(epoch)
            train_dataset.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_index, (images, targets, _) in enumerate(loader):
                if epoch == progress.epoch and batch_index < progress.next_batch:
                    continue
                images, targets = _move_batch_to_device(images, targets, device, pin_memory=pin_memory)
                with _autocast(resolved_amp_dtype, device):
                    output: ModelOutput = model(images)
                    loss = criterion(output.logits, targets) / run.grad_accum_steps
                if scaler_or_none:
                    scaler_or_none.scale(loss).backward()
                else:
                    loss.backward()
                if (batch_index + 1) % run.grad_accum_steps != 0:
                    continue
                if scaler_or_none:
                    scaler_or_none.step(optimizer)
                    scaler_or_none.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                progress.global_step += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                optimizer_step_times.append(now - optimizer_step_started)
                optimizer_step_started = now
                progress.epoch = epoch
                progress.next_batch = batch_index + 1
                if logger:
                    logger.log("optimizer_step", epoch=epoch, batch=batch_index, global_step=progress.global_step, loss=float(loss.detach().cpu()) * run.grad_accum_steps, learning_rate=optimizer.param_groups[0]["lr"])
                if tracker and progress.global_step % run.config.wandb.log_interval_steps == 0:
                    tracker.log({
                        "optimizer_step": progress.global_step,
                        "epoch": epoch,
                        "train/loss": float(loss.detach().cpu()) * run.grad_accum_steps,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    })
                if progress.global_step % run.config.train.checkpoint_every_steps == 0 or progress.global_step >= max_steps:
                    save_checkpoint(run_dir / "checkpoints" / "latest.pt", run, model, optimizer, scheduler, scaler_or_none, progress, context, dataset_hashes, pretrained_checkpoint_sha256, tracker.run_id if tracker else None)
                if progress.global_step >= max_steps:
                    stop = True
                    break
            if stop:
                if dry_run:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    validation_started = time.perf_counter()
                    metrics = _validate_epoch(
                        model, validation_loader, device, context, len(run.config.data.class_names),
                        pin_memory=pin_memory,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    validation_elapsed = torch.tensor(time.perf_counter() - validation_started, dtype=torch.float64, device=device)
                    if context.world_size > 1:
                        dist.all_reduce(validation_elapsed, op=dist.ReduceOp.MAX)  # type: ignore[possibly-missing-attribute]
                    validation_times.append(float(validation_elapsed.item()))
                    if logger:
                        logger.log("validation", epoch=epoch, global_step=progress.global_step, **metrics, best=False, bounded=True)
                    if tracker:
                        tracker.log({
                            "optimizer_step": progress.global_step,
                            "epoch": epoch,
                            "validation/loss": metrics["loss"],
                            "validation/accuracy": metrics["accuracy"],
                            "validation/macro_f1": metrics["macro_f1"],
                            "validation/bounded_smoke": 1,
                        })
                break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validation_started = time.perf_counter()
            metrics = _validate_epoch(
                model, validation_loader, device, context, len(run.config.data.class_names),
                pin_memory=pin_memory,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validation_elapsed = torch.tensor(time.perf_counter() - validation_started, dtype=torch.float64, device=device)
            if context.world_size > 1:
                dist.all_reduce(validation_elapsed, op=dist.ReduceOp.MAX)  # type: ignore[possibly-missing-attribute]
            validation_times.append(float(validation_elapsed.item()))
            optimizer_step_started = time.perf_counter()
            improved = metrics["macro_f1"] > progress.best_score + run.config.train.min_delta
            if improved:
                progress.best_score, progress.best_epoch, progress.bad_epochs = metrics["macro_f1"], epoch, 0
            else:
                progress.bad_epochs += 1
            progress.epoch, progress.next_batch = epoch + 1, 0
            save_checkpoint(run_dir / "checkpoints" / "latest.pt", run, model, optimizer, scheduler, scaler_or_none, progress, context, dataset_hashes, pretrained_checkpoint_sha256, tracker.run_id if tracker else None)
            if improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", run, model, optimizer, scheduler, scaler_or_none, progress, context, dataset_hashes, pretrained_checkpoint_sha256, tracker.run_id if tracker else None)
            if logger:
                logger.log("validation", epoch=epoch, **metrics, best=improved)
            if tracker:
                tracker.log({
                    "optimizer_step": progress.global_step,
                    "epoch": epoch,
                    "validation/loss": metrics["loss"],
                    "validation/accuracy": metrics["accuracy"],
                    "validation/macro_f1": metrics["macro_f1"],
                    "validation/best": int(improved),
                })
            should_stop = progress.bad_epochs >= run.config.train.early_stopping_patience
            if context.world_size > 1:
                tensor = torch.tensor(int(should_stop), device=device)
                dist.broadcast(tensor, src=0)  # type: ignore[possibly-missing-attribute]
                should_stop = bool(tensor.item())
            if should_stop:
                break
        wall_seconds = time.perf_counter() - run_started
        local_peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        local_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        rank_memory: list[Any]
        local_memory = {"rank": context.rank, "allocated_bytes": int(local_peak_allocated), "reserved_bytes": int(local_peak_reserved)}
        if context.world_size > 1:
            rank_memory = [None] * context.world_size
            dist.all_gather_object(rank_memory, local_memory)  # type: ignore[possibly-missing-attribute]
        else:
            rank_memory = [local_memory]
        telemetry = torch.tensor([float(local_peak_allocated), float(local_peak_reserved), wall_seconds], dtype=torch.float64, device=device)
        if context.world_size > 1:
            dist.all_reduce(telemetry, op=dist.ReduceOp.MAX)  # type: ignore[possibly-missing-attribute]
        timing_warmup_steps = min(5, len(optimizer_step_times))
        timed_steps = optimizer_step_times[timing_warmup_steps:]
        median_step = median(timed_steps) if timed_steps else None
        median_validation_seconds = median(validation_times) if validation_times else None
        properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
        device_record = {
            "type": device.type,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "local_rank": context.local_rank,
            "cuda_ordinal": context.local_rank if device.type == "cuda" else None,
            "visible_device_selectors": _visible_device_selectors(),
            "total_memory_bytes": int(properties.total_memory) if properties is not None else None,
            "multi_processor_count": int(properties.multi_processor_count) if properties is not None else None,
            "cuda_runtime": torch.version.cuda,
        }
        summary = {
            "status": "dry_run_complete" if dry_run else "training_complete",
            "model_name": run.model.name,
            "training_seed": run.seed,
            "resolved_run_sha256": config_digest(run),
            "manifest_sha256": dataset_hashes["manifest"],
            "split_sha256": dataset_hashes["splits"],
            "pretrained_checkpoint_sha256": pretrained_checkpoint_sha256,
            "device": device_record,
            "global_steps": progress.global_step,
            "best_validation_macro_f1": None if progress.best_score == -math.inf else progress.best_score,
            "best_epoch": progress.best_epoch,
            "batch_size_per_device": run.batch_size_per_device,
            "grad_accum_steps": run.grad_accum_steps,
            "world_size": context.world_size,
            "effective_global_batch": run.effective_global_batch,
            "precision": resolved_amp_dtype,
            "wandb": {"mode": tracker.mode if tracker else "disabled", "run_id": tracker.run_id if tracker else None, "error": tracker.error if tracker else None},
            "peak_gpu_allocated_bytes": int(telemetry[0].item()),
            "peak_gpu_reserved_bytes": int(telemetry[1].item()),
            "peak_gpu_memory_by_rank": rank_memory,
            "wall_seconds": float(telemetry[2].item()),
            "timing_warmup_steps_rank0": timing_warmup_steps,
            "timed_optimizer_steps_rank0": len(timed_steps),
            "median_optimizer_step_seconds_rank0": median_step,
            **_projection_summary(
                run,
                median_step,
                len(train_dataset),
                len(optimizer_step_times),
                len(train_dataset) if full_train_sample_count is None else full_train_sample_count,
                measured_validation_sample_count=len(validation_dataset),
                full_validation_sample_count=(
                    len(validation_dataset) if full_validation_sample_count is None else full_validation_sample_count
                ),
                measured_validation_seconds=median_validation_seconds,
            ),
        }
        if context.is_main:
            atomic_write_json(run_dir / "summary.json", summary)
            if tracker:
                best_path = run_dir / "checkpoints" / "best.pt"
                latest_path = run_dir / "checkpoints" / "latest.pt"
                final_telemetry = {
                    "hardware/peak_gpu_allocated_bytes": summary["peak_gpu_allocated_bytes"],
                    "hardware/peak_gpu_reserved_bytes": summary["peak_gpu_reserved_bytes"],
                    "runtime/wall_seconds": summary["wall_seconds"],
                    "runtime/median_optimizer_step_seconds": summary["median_optimizer_step_seconds_rank0"],
                    "runtime/projected_training_validation_hours": summary["projected_training_validation_hours"],
                    "batch/physical_per_device": run.batch_size_per_device,
                    "batch/gradient_accumulation": run.grad_accum_steps,
                    "batch/effective_global": run.effective_global_batch,
                }
                tracker.log(final_telemetry)
                tracker.log_table("hardware/peak_memory_by_rank", rank_memory)
                tracker.summarize({
                    "final/global_steps": progress.global_step,
                    "final/best_validation_macro_f1": summary["best_validation_macro_f1"],
                    "final/best_epoch": progress.best_epoch,
                    "final/precision": resolved_amp_dtype,
                    "checkpoint/best_path": str(best_path) if best_path.exists() else None,
                    "checkpoint/best_sha256": sha256_file(best_path) if best_path.exists() else None,
                    "checkpoint/latest_path": str(latest_path) if latest_path.exists() else None,
                    **final_telemetry,
                })
                contract_files = [
                    run_dir / "resolved_run.json", run_dir / "environment.json",
                    run_dir / "model.json", run_dir / "summary.json",
                ]
                tracker.log_artifact_files(
                    f"sac-qutab-{run.model.name}-seed{run.seed}-training-contract",
                    "training-contract",
                    contract_files,
                    metadata={
                        "resolved_run_sha256": config_digest(run),
                        "best_checkpoint_sha256": sha256_file(best_path) if best_path.exists() else None,
                        "latest_checkpoint_sha256": sha256_file(latest_path) if latest_path.exists() else None,
                        "checkpoint_binaries_uploaded": False,
                    },
                    root=run_dir,
                )
        succeeded = True
        return summary
    finally:
        if tracker is not None and context.is_main:
            tracker.finish(exit_code=0 if succeeded else 1)
        close_distributed(context)


def run_training(run: ResolvedRun, run_dir: Path, *, resume_path: Path | None = None, dry_run: bool = False, pretrained_checkpoint_path: str | Path | None = None, pretrained_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    manifest_path = run.config.output_path(run.config.data.manifest_relpath)
    split_path = run.config.output_path(run.config.data.split_relpath)
    samples, split = validate_split_contract(run.config, manifest_path, split_path)
    roles = split["sample_roles"]
    train_samples = [sample for sample in samples if roles[sample.sample_id] == "train"]
    full_train_sample_count = len(train_samples)
    validation_samples = [sample for sample in samples if roles[sample.sample_id] == "validation"]
    full_validation_sample_count = len(validation_samples)
    if dry_run:
        train_samples = train_samples[: run.config.train.dry_run_max_samples]
        validation_samples = validation_samples[: run.config.train.dry_run_max_samples]
    train_dataset = ManifestDataset(run.config.data.root, train_samples, run.config.data.input_size, train=True, seed=run.seed, horizontal_p=run.config.data.horizontal_flip_probability, vertical_p=run.config.data.vertical_flip_probability)
    validation_dataset = ManifestDataset(run.config.data.root, validation_samples, run.config.data.input_size, train=False, seed=run.seed, horizontal_p=0, vertical_p=0)
    if run.model.name != "synthetic_tiny_cnn" and (pretrained_checkpoint_path is None or pretrained_checkpoint_sha256 is None):
        raise ValueError("core training requires the exact local pretrained checkpoint frozen by weight preflight")
    if run.model.name != "synthetic_tiny_cnn":
        assert pretrained_checkpoint_path is not None and pretrained_checkpoint_sha256 is not None
        pretrained_path = Path(pretrained_checkpoint_path)
        if (not pretrained_path.is_file()
                or sha256_file(pretrained_path) != pretrained_checkpoint_sha256
                or pretrained_checkpoint_sha256 != run.model.checkpoint_sha256):
            raise ValueError("local pretrained checkpoint no longer matches the frozen full SHA-256")
    seed_everything(run.seed, run.config.train.deterministic)
    model = build_model(
        run.model,
        len(run.config.data.class_names),
        pretrained_checkpoint=None if resume_path is not None else pretrained_checkpoint_path,
        expected_checkpoint_sha256=None if resume_path is not None else pretrained_checkpoint_sha256,
    )
    return train_model(
        run,
        model,
        train_dataset,
        validation_dataset,
        run_dir,
        resume_path=resume_path,
        dry_run=dry_run,
        full_train_sample_count=full_train_sample_count,
        full_validation_sample_count=full_validation_sample_count,
        pretrained_checkpoint_sha256=pretrained_checkpoint_sha256,
    )

from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import uuid
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from sac_qutab.config import ModelPreset, ResolvedRun, load_config
from sac_qutab.models import SyntheticTinyCNN
from sac_qutab.runtime import seed_everything, sha256_file
from sac_qutab.train import DistributedContext, Progress, load_checkpoint, save_checkpoint


def worker(rank: int, root_text: str, init_file: str) -> None:
    root = Path(root_text)
    dist.init_process_group("gloo", init_method=f"file:///{Path(init_file).resolve().as_posix()}", rank=rank, world_size=2)
    try:
        output = root / "artifacts"
        cfg = load_config("configs/core.yaml", [f"project.output_dir={json.dumps(str(output))}"])
        preset = ModelPreset("synthetic_tiny_cnn", "synthetic", "none", "00000000", "synthetic-v1", .001, "pooled8")
        run = ResolvedRun(cfg, preset, 17, 2, 4, 1, 8)
        model = SyntheticTinyCNN(10)
        optimizer = AdamW(model.parameters(), lr=.001)
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
        hashes = {"manifest": sha256_file(output / "data/manifest.jsonl"), "splits": sha256_file(output / "data/splits.json")}
        context = DistributedContext(rank, 2, rank, False)
        seed_everything(17 + rank)
        checkpoint = root / "ddp.pt"
        progress = Progress(epoch=1, next_batch=2, global_step=3)
        save_checkpoint(checkpoint, run, model, optimizer, scheduler, None, progress, context, hashes)
        expected = torch.rand(5)
        torch.rand(9)
        restored = load_checkpoint(checkpoint, run, model, optimizer, scheduler, None, context, hashes)
        actual = torch.rand(5)
        if restored != progress or not torch.equal(expected, actual):
            raise RuntimeError(f"rank {rank} checkpoint/RNG mismatch")
        dist.barrier()
        if rank == 0:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if [state["rank"] for state in payload["rank_states"]] != [0, 1]:
                raise RuntimeError("rank-keyed states are incomplete")
    finally:
        dist.destroy_process_group()


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    output = root / "artifacts/data"
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (output / "splits.json").write_text("{}\n", encoding="utf-8")
    init_file = Path(tempfile.gettempdir()) / f"sac-gloo-{uuid.uuid4().hex}"
    if init_file.exists():
        init_file.unlink()
    mp.spawn(worker, args=(str(root), str(init_file)), nprocs=2, join=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from sac_qutab.config import ModelPreset, ResolvedRun, load_config
from sac_qutab.models import SyntheticTinyCNN
from sac_qutab.runtime import seed_everything, sha256_file
from sac_qutab.train import DistributedContext, Progress, close_distributed, init_distributed, load_checkpoint, save_checkpoint


root = Path(sys.argv[1])
output = root / "artifacts"
output.joinpath("data").mkdir(parents=True, exist_ok=True)
for relative in ("data/manifest.jsonl", "data/splits.json"):
    path = output / relative
    if not path.exists():
        path.write_text("{}\n", encoding="utf-8")
cfg = load_config("configs/core.yaml", [f"project.output_dir={json.dumps(str(output))}", "distributed.backend=gloo", "train.checkpoint_every_steps=1"])
preset = ModelPreset("synthetic_tiny_cnn", "synthetic", "none", "00000000", "synthetic-v1", .001, "pooled8")
run = ResolvedRun(cfg, preset, 17, 2, 4, 1, 8)
context = init_distributed("gloo", 60)
try:
    model = SyntheticTinyCNN(10)
    optimizer = AdamW(model.parameters(), lr=.001)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    hashes = {"manifest": sha256_file(output / "data/manifest.jsonl"), "splits": sha256_file(output / "data/splits.json")}
    seed_everything(17 + context.rank)
    checkpoint = root / "ddp.pt"
    progress = Progress(epoch=1, next_batch=2, global_step=3)
    save_checkpoint(checkpoint, run, model, optimizer, scheduler, None, progress, context, hashes)
    expected = torch.rand(5)
    torch.rand(9)
    restored = load_checkpoint(checkpoint, run, model, optimizer, scheduler, None, context, hashes)
    actual = torch.rand(5)
    if restored != progress or not torch.equal(expected, actual):
        raise RuntimeError(f"rank {context.rank} checkpoint/RNG mismatch")
    if context.rank == 0:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if [state["rank"] for state in payload["rank_states"]] != [0, 1]:
            raise RuntimeError("rank-keyed states are incomplete")
finally:
    close_distributed(context)

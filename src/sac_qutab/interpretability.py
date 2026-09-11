from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def attention_rollout(model: nn.Module, images: torch.Tensor) -> np.ndarray:
    """Return CLS-to-patch attention rollout for timm ViT-style blocks."""
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if backbone is None or blocks is None:
        raise ValueError("attention rollout requires a FeatureTimmModel with ViT-style blocks")
    captured: list[torch.Tensor] = []
    handles: list[Any] = []

    def hook_for(attention: nn.Module):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise RuntimeError("unexpected timm qkv output for attention rollout")
            batch, tokens, triple_width = output.shape
            heads = int(getattr(attention, "num_heads"))
            if triple_width % (3 * heads):
                raise RuntimeError("attention qkv width is incompatible with num_heads")
            head_dim = triple_width // (3 * heads)
            qkv = output.reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            query, key = qkv[0], qkv[1]
            q_norm = getattr(attention, "q_norm", None)
            k_norm = getattr(attention, "k_norm", None)
            if q_norm is not None:
                query = q_norm(query)
            if k_norm is not None:
                key = k_norm(key)
            scale = float(getattr(attention, "scale", head_dim**-0.5))
            weights = (query * scale) @ key.transpose(-2, -1)
            captured.append(weights.softmax(dim=-1).mean(dim=1).detach())

        return capture

    try:
        for block in blocks:
            attention = getattr(block, "attn", None)
            qkv = getattr(attention, "qkv", None)
            if attention is None or qkv is None:
                raise ValueError("attention rollout supports timm ViT attention blocks with qkv projections")
            handles.append(qkv.register_forward_hook(hook_for(attention)))
        with torch.no_grad():
            model(images)
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise RuntimeError("no attention tensors were captured")
    tokens = captured[0].shape[-1]
    if any(tuple(item.shape) != (len(images), tokens, tokens) for item in captured):
        raise RuntimeError("attention block token geometry changed during rollout")
    identity = torch.eye(tokens, dtype=captured[0].dtype, device=captured[0].device)[None]
    joint = identity.expand(len(images), -1, -1)
    for weights in captured:
        augmented = weights + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        joint = augmented @ joint
    prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
    patch_scores = joint[:, 0, prefix_tokens:]
    grid = math.isqrt(patch_scores.shape[1])
    if grid * grid != patch_scores.shape[1]:
        raise RuntimeError("attention rollout requires a square patch grid")
    heat = patch_scores.reshape(len(images), 1, grid, grid)
    heat = F.interpolate(heat, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    minimum = heat.amin(dim=(1, 2), keepdim=True)
    maximum = heat.amax(dim=(1, 2), keepdim=True)
    heat = (heat - minimum) / (maximum - minimum).clamp_min(torch.finfo(heat.dtype).eps)
    return heat.float().cpu().numpy()

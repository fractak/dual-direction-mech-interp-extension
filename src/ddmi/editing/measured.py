"""Measured single-direction interventions for prospective experiments.

Relative mode is a new intervention, not a calibrated projection ablation.
It matches requested per-token relative L2 displacement along a direction.
Statistics measure the actual cast-back tensor, including rounding error.
"""

from __future__ import annotations

import math

import torch


class MeasuredIntervention:
    def __init__(self, module_name, direction, mode, amount):
        if mode not in ("observe", "projection", "relative_l2"):
            raise ValueError(f"Unknown intervention mode: {mode}")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Intervention amount must be finite and nonnegative")
        d = torch.as_tensor(direction, dtype=torch.float32)
        if d.ndim != 1 or not torch.isfinite(d).all() or d.norm() <= 0:
            raise ValueError("Direction must be a finite nonzero vector")
        self.module_name, self.direction = module_name, d / d.norm()
        self.mode, self.amount = mode, amount
        self._handle = None
        self.reset()

    def reset(self):
        self.calls = []

    def attach(self, model):
        if self._handle is not None:
            raise ValueError("Intervention already attached")
        module = dict(model.named_modules()).get(self.module_name)
        if module is None:
            raise ValueError(f"Missing intervention module: {self.module_name}")
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError("Measured experiments require one unpadded sequence per call")
        if tensor.shape[-1] != self.direction.numel():
            raise ValueError("Direction dimension does not match hidden state")
        h = tensor.detach().float()
        d = self.direction.to(h.device)
        projection = h @ d
        norm = h.norm(dim=-1)
        if self.mode == "observe" or self.amount == 0:
            edited = tensor
        elif self.mode == "projection":
            edited = (h - self.amount * projection.unsqueeze(-1) * d).to(tensor.dtype)
        else:
            # A zero projection uses the positive direction; record these cases.
            sign = torch.where(projection < 0, -torch.ones_like(projection), 1.0)
            step = self.amount * norm * sign
            edited = (h - step.unsqueeze(-1) * d).to(tensor.dtype)
        displacement = (edited.float() - h).norm(dim=-1)
        relative = displacement / norm.clamp_min(1e-12)
        # First call is prefill even if the input has only one token. Generation
        # must use a KV cache; later calls must contain exactly one new token.
        if self.calls and tensor.shape[1] != 1:
            raise ValueError("Decode must use cache and contain exactly one token")

        def summary(values):
            return {
                "mean": float(values.mean()),
                "rms": float(values.square().mean().sqrt()),
                "min": float(values.min()),
                "max": float(values.max()),
                "last": float(values[0, -1]),
            }

        self.calls.append(
            {
                "phase": "prefill" if not self.calls else "decode",
                "tokens": tensor.shape[1],
                "activation_norm": summary(norm),
                "absolute_projection": summary(projection.abs()),
                "displacement_l2": summary(displacement),
                "relative_displacement_l2": summary(relative),
                "zero_projection_tokens": int((projection == 0).sum()),
                "zero_norm_tokens": int((norm == 0).sum()),
            }
        )
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

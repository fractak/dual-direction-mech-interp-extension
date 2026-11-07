"""Measured single-direction interventions for prospective experiments.

Relative mode is a new intervention, not a calibrated projection ablation.
Relative and absolute modes match per-token L2 displacement along a direction.
Statistics measure the actual cast-back tensor, including rounding error.
"""

from __future__ import annotations

import math

import torch


TOKEN_LOCALIZATION_MODES = {
    "position_final_fixed": ("selected", "fixed", None),
    "position_previous_fixed": ("selected", "fixed", None),
    "position_two_back_fixed": ("selected", "fixed", None),
    "position_user_last_fixed": ("selected", "fixed", None),
    "last_prefill_fixed": ("last", "fixed", None),
    "early_prefill_adaptive": ("early", "adaptive", None),
    "early_prefill_fixed": ("early", "fixed", None),
    "decode_only_adaptive": (None, None, "adaptive"),
    "decode_only_fixed": (None, None, "fixed"),
    "early_fixed_else_adaptive": ("early_fixed_else_adaptive", None, "adaptive"),
}


class MeasuredIntervention:
    def __init__(self, module_name, direction, mode, amount):
        if mode not in (
            "observe", "projection", "relative_l2", "absolute_l2",
            "absolute_l2_add", "absolute_l2_subtract",
            "adaptive_prefill_fixed_decode", "fixed_prefill_adaptive_decode",
            *TOKEN_LOCALIZATION_MODES,
        ):
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
        self.prefill_position = None

    def set_prefill_position(self, index):
        if self.calls or type(index) is not int or index < 0:
            raise ValueError("Position must be a nonnegative index set before prefill")
        self.prefill_position = index

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
        localization = TOKEN_LOCALIZATION_MODES.get(self.mode)
        if localization:
            mask, prefill_rule, decode_rule = localization
            adaptive = torch.where(projection < 0, 1.0, -1.0)
            signs = torch.zeros_like(projection)
            if self.calls:
                if decode_rule == "adaptive":
                    signs = adaptive
                elif decode_rule == "fixed":
                    signs.fill_(-1)
            elif mask == "early_fixed_else_adaptive":
                signs.fill_(-1)
                signs[:, -1] = adaptive[:, -1]
            elif mask == "selected":
                if self.prefill_position is None or self.prefill_position >= h.shape[1]:
                    raise ValueError("Selected prefill position is missing or out of range")
                signs[:, self.prefill_position] = -1
            elif mask == "last":
                signs[:, -1] = -1
            elif mask == "early":
                signs[:, :-1] = adaptive[:, :-1] if prefill_rule == "adaptive" else -1
            if self.amount == 0:
                signs.zero_()
            edited = (h + self.amount * signs.unsqueeze(-1) * d).to(tensor.dtype)
        crossover = self.mode in ("adaptive_prefill_fixed_decode", "fixed_prefill_adaptive_decode")
        if self.mode == "observe" or self.amount == 0:
            edited = tensor
        elif localization:
            pass
        elif self.mode == "projection":
            edited = (h - self.amount * projection.unsqueeze(-1) * d).to(tensor.dtype)
        elif self.mode == "absolute_l2_add":
            edited = (h + self.amount * d).to(tensor.dtype)
        elif self.mode == "absolute_l2_subtract":
            edited = (h - self.amount * d).to(tensor.dtype)
        else:
            # A zero projection uses the positive direction; record these cases.
            sign = torch.where(projection < 0, -torch.ones_like(projection), 1.0)
            magnitude = self.amount * norm if self.mode == "relative_l2" else self.amount
            if crossover:
                adaptive = (self.mode == "adaptive_prefill_fixed_decode") == (not self.calls)
                if not adaptive:
                    sign = torch.ones_like(projection)
            step = magnitude * sign
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
        if self.mode in ("absolute_l2_add", "absolute_l2_subtract"):
            self.calls[-1]["signed_projection"] = summary(projection)
            self.calls[-1]["signed_displacement"] = summary((edited.float() - h) @ d)
        if crossover and self.amount > 0:
            adaptive_sign = torch.where(projection < 0, -torch.ones_like(projection), 1.0)
            expected_sign = adaptive_sign if adaptive else torch.ones_like(projection)
            signed_displacement = (edited.float() - h) @ d
            error = (signed_displacement + self.amount * expected_sign).abs()
            self.calls[-1].update(
                sign_rule="adaptive" if adaptive else "fixed_subtract",
                signed_error_max=float(error.max()),
                positive_step_tokens=int((signed_displacement > 0).sum()),
                negative_projection_tokens=int((projection < 0).sum()),
            )
        if localization:
            delta = (edited.float() - h) @ d
            self.calls[-1]["token_localization"] = {
                "projection": projection[0].tolist(),
                "sign": signs[0].to(torch.int8).tolist(),
                "signed_displacement": delta[0].tolist(),
                "displacement_l2": displacement[0].tolist(),
            }
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

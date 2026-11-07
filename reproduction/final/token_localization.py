"""Recompute the reported token-position localization contrasts from saved labels."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

RECEIVERS = ("original_instruct", "safety42")
MODES = ("last_prefill_fixed", "early_prefill_adaptive", "early_prefill_fixed",
         "decode_only_adaptive", "decode_only_fixed", "early_fixed_else_adaptive")


def analyze(root, reviewed):
    grid = {}
    prompts = None
    plan_sha256 = None
    sources = (root / "artifacts/review/fresh_absolute_resolved/reviewed_source.json",
               root / "artifacts/review/fixed_sign_resolved/reviewed_source.json", reviewed)
    for index, source in enumerate(sources):
        data = json.loads(source.read_text())
        if data["complete"] is not True:
            raise ValueError(f"Incomplete reviewed source: {source}")
        for c in data["conditions"]:
            receiver = c["receiver"].replace("safety_lora", "safety42")
            if receiver not in RECEIVERS:
                continue
            if index == 0:
                mode = {"both": "baseline", "evidence": "adaptive"}.get(c["source"])
            elif index == 1:
                mode = "fixed" if c["mode"] == "absolute_l2_subtract" else None
            else:
                mode = c["mode"]
                if mode not in MODES:
                    raise ValueError(f"Unexpected localization mode: {mode}")
            if mode is None:
                continue
            rows = {r["prompt_id"]: r for r in c["rows"]}
            roster = {pid: (r["domain"], r["prompt"]) for pid, r in rows.items()}
            if prompts is None:
                prompts = roster
            if (len(rows) != 120 or len(c["rows"]) != 120 or roster != prompts
                    or (receiver, mode) in grid or c["layer"] != 20
                    or c["amount"] != (0 if mode == "baseline" else 4)):
                raise ValueError(f"Unexpected localization cell: {receiver}/{mode}")
            grid[receiver, mode] = rows
        if index == 2:
            plan_sha256 = data["plan_sha256"]
    modes = (*MODES, "adaptive", "fixed")
    if grid.keys() != {(r, m) for r in RECEIVERS for m in (*modes, "baseline")}:
        raise ValueError("Incomplete localization grid")
    if Counter(Counter(family for family, _ in prompts.values()).values()) != {4: 30}:
        raise ValueError("Expected 30 families with four phrasings each")

    def estimates(ids, outcome, stream):
        families = sorted({prompts[pid][0] for pid in ids})
        mask = np.array([[prompts[pid][0] == family for pid in ids] for family in families], dtype=int)
        draws = np.random.default_rng(42).integers(0, len(families), (20_000, len(families)))
        denominators = mask.sum(axis=1)[draws].sum(axis=1)
        rates = {key: np.array([r[pid]["independent_review"][stream] == outcome for pid in ids], dtype=int)
                 for key, r in grid.items()}
        o, a = RECEIVERS
        baseline = rates[a, "baseline"] - rates[o, "baseline"]
        values = {m: rates[a, m] - rates[o, m] - baseline for m in modes}
        values.update(earlier_adaptive_minus_fixed=values["early_prefill_adaptive"]-values["early_prefill_fixed"],
                      decode_adaptive_minus_fixed=values["decode_only_adaptive"]-values["decode_only_fixed"],
                      adaptive_minus_earlier_fixed=values["adaptive"]-values["early_fixed_else_adaptive"],
                      adaptive_minus_fixed=values["adaptive"]-values["fixed"])
        return {name: [float(100 * v.mean()), *np.quantile(
            100 * (mask @ v)[draws].sum(axis=1)/denominators, (.025,.975)).tolist()]
                for name, v in values.items()}

    ids = list(prompts)
    uncapped = [pid for pid in ids if all(not r[pid]["hit_token_limit"] for r in grid.values())]
    results = {subset: {outcome: {stream: estimates(group, outcome, stream)
                for stream in ("outcome", "a", "b")} for outcome in ("acknowledgment", "unsupported")}
               for subset, group in (("all", ids), ("uncapped", uncapped)) if group}
    ack = results["all"]["acknowledgment"]
    published = {
        ("outcome", "last_prefill_fixed"): -35.8,
        ("outcome", "early_prefill_adaptive"): -2.5,
        ("outcome", "decode_only_adaptive"): 1.7,
        ("outcome", "adaptive_minus_earlier_fixed"): 11.7,
        ("a", "adaptive_minus_earlier_fixed"): -4.2,
        ("b", "adaptive_minus_earlier_fixed"): 12.5,
    }
    for (stream, contrast), estimate in published.items():
        if round(ack[stream][contrast][0], 1) != estimate:
            raise ValueError(f"Localization {stream}/{contrast} differs from paper")
    reported_intervals = {
        "last_prefill_fixed": (-50.8, -20.8),
        "early_prefill_adaptive": (-9.2, 4.2),
        "decode_only_adaptive": (-5.8, 8.3),
    }
    for contrast, bounds in reported_intervals.items():
        if tuple(round(v, 1) for v in ack["outcome"][contrast][1:]) != bounds:
            raise ValueError(f"Localization interval differs from paper: {contrast}")
    return {"plan_sha256": plan_sha256, "uncapped_prompts": len(uncapped), "estimates": results}

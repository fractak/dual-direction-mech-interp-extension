"""Analyze independently reviewed prospective runs without editing frozen evidence."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ddmi.paper.evidence import OUTCOMES
from ddmi.paper.stats import bootstrap_indices, delta_summary
from ddmi.publication import validate_plan


def cluster_summary(delta, domains, samples=20000, seed=42):
    names = sorted(set(domains))
    if len(names) < 2:
        return None
    totals = np.array([sum(x for x, d in zip(delta, domains) if d == name) for name in names])
    sizes = np.array([domains.count(name) for name in names])
    draws = bootstrap_indices(len(names), samples, seed)
    estimates = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {
        "domain_clusters": len(names),
        "change": float(np.mean(delta)),
        "bootstrap_95_ci": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "assumption": (
            "Domains are exchangeable clusters; all prompts within a drawn domain stay together"
        ),
    }


def analyze_reviewed(plan, run):
    validate_plan(plan)
    if run.get("complete") is not True or run["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Require a complete run paired to this exact plan")
    payload = plan["payload"]
    prompts = payload["prompts"]
    ids = [p["id"] for p in prompts]
    n = len(ids)
    domains = [p["domain"] for p in prompts]
    conditions = {c["id"]: c for c in run["conditions"]}
    expected = {c["id"]: c for c in payload["conditions"]}
    if len(conditions) != len(run["conditions"]) or conditions.keys() != expected.keys():
        raise ValueError("Missing, duplicate or unexpected conditions")
    summaries, indicators, baseline = {}, {}, {}
    for key, condition in conditions.items():
        spec = expected[key]
        if any(condition[k] != v for k, v in spec.items()):
            raise ValueError("Condition settings differ from frozen plan")
        rows = condition["rows"]
        if [r["prompt_id"] for r in rows] != ids or [r["prompt"] for r in rows] != [
            p["prompt"] for p in prompts
        ]:
            raise ValueError("Prompt pairing differs from plan")
        if [r["domain"] for r in rows] != domains:
            raise ValueError("Prompt domain assignments differ from plan")
        labels = [r.get("independent_review", {}).get("outcome") for r in rows]
        if any(label not in OUTCOMES for label in labels):
            raise ValueError("Every response needs a resolved independent review")
        counts = Counter(labels)
        indicators[key] = {
            outcome: np.array([label == outcome for label in labels], dtype=float)
            for outcome in OUTCOMES
        }
        if spec["mode"] == "observe":
            baseline[(spec["receiver"], spec["layer"])] = key
        # Last prefill token is a common input position across receivers. Decode
        # statistics are conditional on different generated trajectories.
        telemetry = {}
        expected_sources = (
            {"original_instruct", "safety_lora"} if spec["source"] == "both" else {spec["source"]}
        )
        if any(set(r["telemetry"]) != expected_sources for r in rows):
            raise ValueError("Missing or unexpected telemetry sources")
        sources = expected_sources
        for source in sorted(sources):
            calls = [r["telemetry"][source] for r in rows]
            if any(not seq or seq[0]["phase"] != "prefill" for seq in calls):
                raise ValueError("Missing prefill telemetry")
            telemetry[source] = {
                "mean_last_prefill_absolute_projection": float(
                    np.mean([seq[0]["absolute_projection"]["last"] for seq in calls])
                ),
                "mean_last_prefill_displacement_l2": float(
                    np.mean([seq[0]["displacement_l2"]["last"] for seq in calls])
                ),
                "mean_last_prefill_relative_displacement_l2": float(
                    np.mean([seq[0]["relative_displacement_l2"]["last"] for seq in calls])
                ),
                "max_token_relative_displacement_l2": max(
                    call["relative_displacement_l2"]["max"] for seq in calls for call in seq
                ),
                "zero_projection_tokens": sum(
                    call["zero_projection_tokens"] for seq in calls for call in seq
                ),
            }
        summaries[key] = {
            **spec,
            "n": n,
            "counts": {o: counts[o] for o in OUTCOMES},
            "hit_token_limit": sum(r["hit_token_limit"] for r in rows),
            "input_truncated": sum(r["input_truncated"] for r in rows),
            "telemetry": telemetry,
        }
    indices = bootstrap_indices(n)
    comparisons = []
    for key, spec in expected.items():
        if spec["receiver"] != "safety_lora" or spec["mode"] == "observe":
            continue
        match = next(
            (
                other
                for other, ospec in expected.items()
                if ospec["receiver"] == "original_instruct"
                and all(
                    ospec[k] == spec[k]
                    for k in ("layer", "mode", "amount", "source", "random_seed")
                )
            ),
            None,
        )
        if match is None:
            raise ValueError("Missing paired receiver condition")
        base_lora = baseline[("safety_lora", spec["layer"])]
        base_original = baseline[("original_instruct", spec["layer"])]
        effects = {}
        for outcome in ("acknowledgment", "unsupported"):
            delta = (indicators[key][outcome] - indicators[base_lora][outcome]) - (
                indicators[match][outcome] - indicators[base_original][outcome]
            )
            effects[outcome] = {
                "prompt_paired": delta_summary(delta, indices),
                "domain_clustered": cluster_summary(delta, domains)
                if all(isinstance(d, str) and d for d in domains)
                else None,
            }
        comparisons.append({"lora_condition": key, "original_condition": match, "effects": effects})
    return {
        "plan_sha256": plan["plan_sha256"],
        "n_prompts": n,
        "conditions": summaries,
        "receiver_differences_in_changes": comparisons,
        "inference": (
            "Exploratory paired intervals conditional on this adapter and "
            "prompt/domain sampling assumptions. No multiplicity correction or inference "
            "across training seeds; do not treat grid selection as confirmation."
        ),
    }

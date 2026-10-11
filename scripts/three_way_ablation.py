#!/usr/bin/env python3
"""Experiment 2: Cross-ablation — ablate one direction, measure both behaviors.

The 2×2 experiment:
  - Baseline (no ablation) → measure safety refusal + epistemic abstention rates
  - Ablate safety direction → measure both rates
  - Ablate epistemic direction → measure both rates

This is the core experiment: if ablating the safety direction also kills epistemic
abstention, they share a mechanism. If independent, they're mechanistically distinct.

Usage:
    python scripts/cross_ablation.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --safety-direction artifacts/directions/llama31_8b_safety.json \
        --epistemic-direction artifacts/directions/llama31_8b_epistemic.json \
        --output artifacts/cross_ablation/llama31_8b_results.json \
        --load-in-4bit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import (
    load_capability_prompts,
    load_epistemic_prompts,
    load_safety_prompts,
    prompts_to_text_list,
)
from ddmi.editing.apply_edit import (
    ActivationDirectionAblator,
    EditSpec,
    LayerwiseActivationDirectionAblator,
)
from ddmi.evaluation.detectors import (
    abstention_rate,
    capability_refusal_rate,
    classify_responses,
    is_capability_refusal,
    refusal_rate,
)
from ddmi.evaluation.metrics import bootstrap_rate_ci
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, read_json, set_seed, write_json

logger = configure_logging(name="cross_ablation")


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    safety_prompts: List[str],
    epistemic_prompts: List[str],
    capability_prompts: List[str],
    gen_config: TextGenerationConfig,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Evaluate safety, epistemic, and capability behavior."""
    # Generate responses
    logger.info("  Generating on %d safety prompts...", len(safety_prompts))
    safety_responses = generate_batch(model, tokenizer, safety_prompts, config=gen_config)

    logger.info("  Generating on %d epistemic prompts...", len(epistemic_prompts))
    epistemic_responses = generate_batch(model, tokenizer, epistemic_prompts, config=gen_config)

    logger.info("  Generating on %d capability prompts...", len(capability_prompts))
    capability_responses = generate_batch(model, tokenizer, capability_prompts, config=gen_config)

    # Classify
    safety_classes = classify_responses(safety_responses)
    epistemic_classes = classify_responses(epistemic_responses)

    safety_refusal_labels = [label == "refusal" for label in safety_classes]
    safety_abstention_labels = [label == "abstention" for label in safety_classes]
    epistemic_refusal_labels = [label == "refusal" for label in epistemic_classes]
    epistemic_abstention_labels = [label == "abstention" for label in epistemic_classes]
    capability_refusal_labels = [
        is_capability_refusal(response) for response in capability_responses
    ]

    return {
        "safety_refusal_rate": refusal_rate(safety_responses),
        "safety_abstention_rate": abstention_rate(safety_responses),
        "epistemic_refusal_rate": refusal_rate(epistemic_responses),
        "epistemic_abstention_rate": abstention_rate(epistemic_responses),
        "capability_refusal_rate": capability_refusal_rate(capability_responses),
        "safety_refusal_rate_ci_95": bootstrap_rate_ci(
            safety_refusal_labels,
            num_bootstrap=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "safety_abstention_rate_ci_95": bootstrap_rate_ci(
            safety_abstention_labels,
            num_bootstrap=bootstrap_samples,
            seed=bootstrap_seed + 1,
        ),
        "epistemic_refusal_rate_ci_95": bootstrap_rate_ci(
            epistemic_refusal_labels,
            num_bootstrap=bootstrap_samples,
            seed=bootstrap_seed + 2,
        ),
        "epistemic_abstention_rate_ci_95": bootstrap_rate_ci(
            epistemic_abstention_labels,
            num_bootstrap=bootstrap_samples,
            seed=bootstrap_seed + 3,
        ),
        "capability_refusal_rate_ci_95": bootstrap_rate_ci(
            capability_refusal_labels,
            num_bootstrap=bootstrap_samples,
            seed=bootstrap_seed + 4,
        ),
        "safety_response_classes": {
            "refusal": safety_classes.count("refusal"),
            "abstention": safety_classes.count("abstention"),
            "answer": safety_classes.count("answer"),
        },
        "epistemic_response_classes": {
            "refusal": epistemic_classes.count("refusal"),
            "abstention": epistemic_classes.count("abstention"),
            "answer": epistemic_classes.count("answer"),
        },
        # Store a few example responses for qualitative inspection
        "safety_examples": safety_responses[:5],
        "epistemic_examples": epistemic_responses[:5],
        "capability_examples": capability_responses[:5],
    }


def build_direction_map(
    artifact: Dict[str, Any],
    top_k_layers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    ranked = artifact["ranked_layers"][:top_k_layers]
    directions = {
        row["name"]: artifact["directions"][row["name"]]["direction"]
        for row in ranked
    }
    return ranked, directions


def build_ablator(
    directions_by_layer: Dict[str, List[float]],
    spec: EditSpec,
) -> Any:
    if len(directions_by_layer) == 1:
        module_name, direction = next(iter(directions_by_layer.items()))
        return ActivationDirectionAblator(
            module_names=[module_name],
            direction=direction,
            spec=spec,
        )
    return LayerwiseActivationDirectionAblator(
        directions_by_module=directions_by_layer,
        spec=spec,
    )


def evaluate_ablation_run(
    model: Any,
    tokenizer: Any,
    safety_prompts: List[str],
    epistemic_prompts: List[str],
    capability_prompts: List[str],
    gen_config: TextGenerationConfig,
    baseline: Dict[str, Any],
    safety_ranked: List[Dict[str, Any]],
    safety_directions: Dict[str, List[float]],
    epistemic_ranked: List[Dict[str, Any]],
    epistemic_directions: Dict[str, List[float]],
    capability_ranked: List[Dict[str, Any]],
    capability_directions: Dict[str, List[float]],
    strength: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    spec = EditSpec(strength=strength)
    results: List[Dict[str, Any]] = [dict(baseline)]

    safety_ablator = build_ablator(safety_directions, spec).attach(model)
    try:
        t0 = time.time()
        ablate_safety = evaluate_condition(
            model,
            tokenizer,
            safety_prompts,
            epistemic_prompts,
            capability_prompts,
            gen_config,
            bootstrap_samples,
            bootstrap_seed + 10,
        )
    finally:
        safety_ablator.close()

    ablate_safety["condition"] = "ablate_safety"
    ablate_safety["ablated_direction"] = "safety"
    ablate_safety["ablated_layers"] = [r["name"] for r in safety_ranked]
    ablate_safety["strength"] = strength
    ablate_safety["elapsed_seconds"] = time.time() - t0
    results.append(ablate_safety)

    epistemic_ablator = build_ablator(epistemic_directions, spec).attach(model)
    try:
        t0 = time.time()
        ablate_epistemic = evaluate_condition(
            model,
            tokenizer,
            safety_prompts,
            epistemic_prompts,
            capability_prompts,
            gen_config,
            bootstrap_samples,
            bootstrap_seed + 20,
        )
    finally:
        epistemic_ablator.close()

    ablate_epistemic["condition"] = "ablate_epistemic"
    ablate_epistemic["ablated_direction"] = "epistemic"
    ablate_epistemic["ablated_layers"] = [r["name"] for r in epistemic_ranked]
    ablate_epistemic["strength"] = strength
    ablate_epistemic["elapsed_seconds"] = time.time() - t0
    results.append(ablate_epistemic)

    capability_ablator = build_ablator(capability_directions, spec).attach(model)
    try:
        t0 = time.time()
        ablate_capability = evaluate_condition(
            model,
            tokenizer,
            safety_prompts,
            epistemic_prompts,
            capability_prompts,
            gen_config,
            bootstrap_samples,
            bootstrap_seed + 30,
        )
    finally:
        capability_ablator.close()

    ablate_capability["condition"] = "ablate_capability"
    ablate_capability["ablated_direction"] = "capability"
    ablate_capability["ablated_layers"] = [r["name"] for r in capability_ranked]
    ablate_capability["strength"] = strength
    ablate_capability["elapsed_seconds"] = time.time() - t0
    results.append(ablate_capability)

    safety_cross = abs(
        ablate_safety["epistemic_abstention_rate"] - baseline["epistemic_abstention_rate"]
    )
    epistemic_cross = abs(
        ablate_epistemic["safety_refusal_rate"] - baseline["safety_refusal_rate"]
    )

    return {
        "config": {
            "top_k_layers": len(safety_ranked),
            "strength": strength,
            "target_layers": {
                "safety": [r["name"] for r in safety_ranked],
                "epistemic": [r["name"] for r in epistemic_ranked],
                "capability": [r["name"] for r in capability_ranked],
            },
        },
        "results": results,
        "summary": {
            "baseline_safety_refusal": baseline["safety_refusal_rate"],
            "baseline_epistemic_abstention": baseline["epistemic_abstention_rate"],
            "baseline_capability_refusal": baseline["capability_refusal_rate"],

            "on_target_safety_drop":
                baseline["safety_refusal_rate"]
                - ablate_safety["safety_refusal_rate"],

            "on_target_epistemic_drop":
                baseline["epistemic_abstention_rate"]
                - ablate_epistemic["epistemic_abstention_rate"],

            "on_target_capability_drop":
                baseline["capability_refusal_rate"]
                - ablate_capability["capability_refusal_rate"],

            "safety_to_epistemic":
                baseline["epistemic_abstention_rate"]
                - ablate_safety["epistemic_abstention_rate"],

            "safety_to_capability":
                baseline["capability_refusal_rate"]
                - ablate_safety["capability_refusal_rate"],

            "epistemic_to_safety":
                baseline["safety_refusal_rate"]
                - ablate_epistemic["safety_refusal_rate"],

            "epistemic_to_capability":
                baseline["capability_refusal_rate"]
                - ablate_epistemic["capability_refusal_rate"],

            "capability_to_safety":
                baseline["safety_refusal_rate"]
                - ablate_capability["safety_refusal_rate"],

            "capability_to_epistemic":
                baseline["epistemic_abstention_rate"]
                - ablate_capability["epistemic_abstention_rate"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-ablation experiment")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--safety-direction", required=True, help="Safety direction artifact JSON")
    parser.add_argument("--epistemic-direction", required=True, help="Epistemic direction artifact JSON")
    parser.add_argument("--capability-direction", required=True, help="Capability direction artifact JSON")
    parser.add_argument("--output", required=True, help="Output results JSON")
    parser.add_argument("--eval-prompts", type=int, default=100, help="Eval prompts per group")
    parser.add_argument("--top-k-layers", type=int, default=1, help="Top-K layers to ablate")
    parser.add_argument("--top-k-values", type=int, nargs="+", help="Optional sweep over top-K layer counts")
    parser.add_argument("--strength", type=float, default=1.0, help="Ablation strength")
    parser.add_argument("--strength-values", type=float, nargs="+", help="Optional sweep over ablation strengths")
    parser.add_argument("--module-type", nargs="+", default=["attn_out"], help="Module types to ablate")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load direction artifacts
    safety_art = read_json(args.safety_direction)
    epistemic_art = read_json(args.epistemic_direction)
    capability_art = read_json(args.capability_direction)

    top_k_values = args.top_k_values or [args.top_k_layers]
    strength_values = args.strength_values or [args.strength]

    if any(value <= 0 for value in top_k_values):
        raise ValueError(f"top-k values must be positive, got {top_k_values}")

    default_safety_ranked, _ = build_direction_map(safety_art, top_k_values[0])
    default_epistemic_ranked, _ = build_direction_map(epistemic_art, top_k_values[0])
    default_capability_ranked, _ = build_direction_map(capability_art, top_k_values[0])
    logger.info(
        "Default safety direction anchor: %s (sep=%.4f)",
        default_safety_ranked[0]["name"],
        default_safety_ranked[0]["score"],
    )
    logger.info(
        "Default epistemic direction anchor: %s (sep=%.4f)",
        default_epistemic_ranked[0]["name"],
        default_epistemic_ranked[0]["score"],
    )
    logger.info(
        "Default capability direction anchor: %s (sep=%.4f)",
        default_capability_ranked[0]["name"],
        default_capability_ranked[0]["score"],
    )

    # Load model
    config = ModelLoadConfig(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype="float16",
        attn_implementation="eager",
    )
    logger.info("Loading model: %s", args.model_id)
    model, tokenizer = load_model_and_tokenizer(config)

    gen_config = TextGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    # Load evaluation prompts (separate from training prompts to avoid leakage)
    safety_eval = load_safety_prompts(limit=args.eval_prompts, seed=args.seed + 1000)
    epistemic_eval = load_epistemic_prompts(limit=args.eval_prompts, seed=args.seed + 1000)
    capability_eval = load_capability_prompts(
        limit=120,
        seed=args.seed,
        split="expanded_eval",
    )

    # For safety eval, use harmful prompts (we want to see if refusal persists)
    safety_prompts = prompts_to_text_list(safety_eval["harmful"])
    # For epistemic eval, use unanswerable prompts (we want to see if abstention persists)
    epistemic_prompts = prompts_to_text_list(epistemic_eval["unanswerable"])
    capability_prompts = prompts_to_text_list(capability_eval["incapable"])

    logger.info(
        "Eval prompts: %d safety, %d epistemic, %d capability",
        len(safety_prompts),
        len(epistemic_prompts),
        len(capability_prompts),
    )

    # === Condition 1: Baseline (no ablation) ===
    logger.info("\n=== Condition 1: BASELINE (no ablation) ===")
    t0 = time.time()
    baseline = evaluate_condition(
        model,
        tokenizer,
        safety_prompts,
        epistemic_prompts,
        capability_prompts,
        gen_config,
        args.bootstrap_samples,
        args.seed,
    )
    baseline["condition"] = "baseline"
    baseline["ablated_direction"] = "none"
    baseline["elapsed_seconds"] = time.time() - t0
    logger.info(
        "Baseline: safety_refusal=%.3f, epistemic_abstention=%.3f, capability_refusal=%.3f",
        baseline["safety_refusal_rate"],
        baseline["epistemic_abstention_rate"],
        baseline["capability_refusal_rate"],
    )

    runs: List[Dict[str, Any]] = []
    for top_k_layers in top_k_values:
        safety_ranked, safety_directions = build_direction_map(safety_art, top_k_layers)
        epistemic_ranked, epistemic_directions = build_direction_map(epistemic_art, top_k_layers)
        capability_ranked, capability_directions = build_direction_map(capability_art, top_k_layers)
        logger.info("Safety target layers: %s", ", ".join(row["name"] for row in safety_ranked))
        logger.info("Epistemic target layers: %s", ", ".join(row["name"] for row in epistemic_ranked))
        logger.info("Capability target layers: %s", ", ".join(row["name"] for row in capability_ranked))

        for strength in strength_values:
            logger.info(
                "\n=== Cross-ablation run: top_k=%d strength=%.3f ===",
                top_k_layers,
                strength,
            )
            run = evaluate_ablation_run(
                model=model,
                tokenizer=tokenizer,
                safety_prompts=safety_prompts,
                epistemic_prompts=epistemic_prompts,
                capability_prompts=capability_prompts,
                gen_config=gen_config,
                baseline=baseline,
                safety_ranked=safety_ranked,
                safety_directions=safety_directions,
                epistemic_ranked=epistemic_ranked,
                epistemic_directions=epistemic_directions,
                capability_ranked=capability_ranked,
                capability_directions=capability_directions,
                strength=strength,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.seed + top_k_layers * 100 + int(strength * 1000),
            )
            runs.append(run)

            logger.info(
                "%-25s | Safety | Epistemic | Capability",
                "Condition",
            )
            logger.info("-" * 65)

            for row in run["results"]:
                logger.info(
                    "%-25s | %.3f  | %.3f     | %.3f",
                    row["condition"],
                    row["safety_refusal_rate"],
                    row["epistemic_abstention_rate"],
                    row["capability_refusal_rate"],
                )

            logger.info(
                "On-target drops: S=%.3f E=%.3f C=%.3f",
                run["summary"]["on_target_safety_drop"],
                run["summary"]["on_target_epistemic_drop"],
                run["summary"]["on_target_capability_drop"],
            )

            logger.info(
                "Cross: S->E=%.3f S->C=%.3f E->S=%.3f "
                "E->C=%.3f C->S=%.3f C->E=%.3f",
                run["summary"]["safety_to_epistemic"],
                run["summary"]["safety_to_capability"],
                run["summary"]["epistemic_to_safety"],
                run["summary"]["epistemic_to_capability"],
                run["summary"]["capability_to_safety"],
                run["summary"]["capability_to_epistemic"],
            )

    if len(runs) == 1:
        artifact = {
            "artifact_type": "three_way_cross_ablation_results",
            "model_id": args.model_id,
            "safety_direction_artifact": args.safety_direction,
            "epistemic_direction_artifact": args.epistemic_direction,
            "capability_direction_artifact": args.capability_direction,
            "config": {
                "eval_prompts": args.eval_prompts,
                "top_k_layers": runs[0]["config"]["top_k_layers"],
                "strength": runs[0]["config"]["strength"],
                "target_layers": runs[0]["config"]["target_layers"],
                "max_new_tokens": args.max_new_tokens,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
            },
            "results": runs[0]["results"],
            "summary": runs[0]["summary"],
        }
    else:
        artifact = {
            "artifact_type": "three_way_cross_ablation_sweep",
            "model_id": args.model_id,
            "safety_direction_artifact": args.safety_direction,
            "epistemic_direction_artifact": args.epistemic_direction,
            "capability_direction_artifact": args.capability_direction,
            "baseline": baseline,
            "config": {
                "eval_prompts": args.eval_prompts,
                "top_k_values": top_k_values,
                "strength_values": strength_values,
                "max_new_tokens": args.max_new_tokens,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
            },
            "runs": runs,
        }

    write_json(args.output, artifact)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()

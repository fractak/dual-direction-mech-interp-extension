#!/usr/bin/env python3
"""Experiment 4: Cross-model replication.

Runs direction extraction + comparison on multiple model families
to test whether the safety↔epistemic relationship generalizes.

Default models (all fit on T4x2 in 4-bit):
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen3-8B
    - unsloth/gemma-2-9b-it-bnb-4bit

Usage:
    python scripts/cross_model_replication.py \
        --output-dir artifacts/cross_model/ \
        --load-in-4bit --prompt-limit 200
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import (
    load_epistemic_prompts,
    load_safety_prompts,
    load_capability_prompts,
    prompts_to_text_list,
)
from ddmi.editing.directions import (
    angular_distance_degrees,
    cosine_similarity,
    direction_from_contrast,
    rank_layers_by_separability,
    separability_score,
)
from ddmi.models.hooks import collect_activations_batched, get_residual_stream_names
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, ensure_dir, set_seed, write_json

logger = configure_logging(name="cross_model")


DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "unsloth/gemma-2-9b-it-bnb-4bit",
]


def run_single_model(
    model_id: str,
    safety_a: List[str],
    safety_b: List[str],
    epistemic_a: List[str],
    epistemic_b: List[str],
    capability_a: List[str],
    capability_b: List[str],
    load_in_4bit: bool = True,
    max_input_length: int = 512,
) -> Dict[str, Any]:
    """Extract directions and compute geometry for a single model."""
    # Phi-4-mini ships custom modeling code
    needs_trust = "phi" in model_id.lower()
    config = ModelLoadConfig(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        torch_dtype="float16",
        attn_implementation="eager",
        trust_remote_code=needs_trust,
    )

    logger.info("Loading model: %s", model_id)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    layer_names = get_residual_stream_names(model)
    logger.info("Discovered %d layers", len(layer_names))

    # Collect activations
    logger.info("Collecting safety activations...")
    acts_s_a = collect_activations_batched(model, tokenizer, safety_a, layer_names, max_input_length)
    acts_s_b = collect_activations_batched(model, tokenizer, safety_b, layer_names, max_input_length)

    logger.info("Collecting epistemic activations...")
    acts_e_a = collect_activations_batched(model, tokenizer, epistemic_a, layer_names, max_input_length)
    acts_e_b = collect_activations_batched(model, tokenizer, epistemic_b, layer_names, max_input_length)

    logger.info("Collecting capability activations...")
    acts_c_a = collect_activations_batched(model, tokenizer, capability_a, layer_names, max_input_length)
    acts_c_b = collect_activations_batched(model, tokenizer, capability_b, layer_names, max_input_length)

    # Per-layer analysis
    comparisons = []
    safety_sep_scores = {}
    epistemic_sep_scores = {}
    capability_sep_scores = {}

    for layer_name in layer_names:
        s_dir = direction_from_contrast(
            acts_s_a[layer_name], acts_s_b[layer_name]
        )
        e_dir = direction_from_contrast(
            acts_e_a[layer_name], acts_e_b[layer_name]
        )
        c_dir = direction_from_contrast(
            acts_c_a[layer_name], acts_c_b[layer_name]
        )

        s_sep = separability_score(
            acts_s_a[layer_name], acts_s_b[layer_name]
        )
        e_sep = separability_score(
            acts_e_a[layer_name], acts_e_b[layer_name]
        )
        c_sep = separability_score(
            acts_c_a[layer_name], acts_c_b[layer_name]
        )

        se_cos = cosine_similarity(s_dir, e_dir)
        sc_cos = cosine_similarity(s_dir, c_dir)
        ec_cos = cosine_similarity(e_dir, c_dir)

        layer_idx = int(layer_name.split(".")[-1])

        comparisons.append({
            "layer_name": layer_name,
            "layer_index": layer_idx,
            "layer_depth_frac": layer_idx / len(layer_names),

            # Preserve original fields.
            "cosine_similarity": se_cos,
            "angular_distance_deg":
                angular_distance_degrees(s_dir, e_dir),

            "safety_epistemic_cosine": se_cos,
            "safety_capability_cosine": sc_cos,
            "epistemic_capability_cosine": ec_cos,

            "safety_epistemic_angle":
                angular_distance_degrees(s_dir, e_dir),
            "safety_capability_angle":
                angular_distance_degrees(s_dir, c_dir),
            "epistemic_capability_angle":
                angular_distance_degrees(e_dir, c_dir),

            "safety_separability": s_sep,
            "epistemic_separability": e_sep,
            "capability_separability": c_sep,
        })

        safety_sep_scores[layer_name] = s_sep
        epistemic_sep_scores[layer_name] = e_sep
        capability_sep_scores[layer_name] = c_sep

    # Top layer analysis
    safety_ranked = rank_layers_by_separability(safety_sep_scores)
    epistemic_ranked = rank_layers_by_separability(epistemic_sep_scores)
    capability_ranked = rank_layers_by_separability(capability_sep_scores)

    top_safety_layer = safety_ranked[0][0]
    top_epistemic_layer = epistemic_ranked[0][0]
    top_capability_layer = capability_ranked[0][0]

    top_safety_comparison = next(
        c for c in comparisons
        if c["layer_name"] == top_safety_layer
    )
    top_capability_comparison = next(
        c for c in comparisons
        if c["layer_name"] == top_capability_layer
    )

    mean_se = sum(
        c["safety_epistemic_cosine"] for c in comparisons
    ) / len(comparisons)
    mean_sc = sum(
        c["safety_capability_cosine"] for c in comparisons
    ) / len(comparisons)
    mean_ec = sum(
        c["epistemic_capability_cosine"] for c in comparisons
    ) / len(comparisons)

    logger.info(
        "Top safety layer: %s (sep=%.4f)",
        top_safety_layer, safety_ranked[0][1]
    )
    logger.info(
        "Top epistemic layer: %s (sep=%.4f)",
        top_epistemic_layer, epistemic_ranked[0][1]
    )
    logger.info(
        "Top capability layer: %s (sep=%.4f)",
        top_capability_layer, capability_ranked[0][1]
    )
    logger.info(
        "Mean cosines: S-E=%.4f S-C=%.4f E-C=%.4f",
        mean_se, mean_sc, mean_ec
    )
    logger.info(
        "At top capability layer: S-C=%.4f E-C=%.4f",
        top_capability_comparison["safety_capability_cosine"],
        top_capability_comparison["epistemic_capability_cosine"],
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "num_layers": len(layer_names),

        "top_safety_layer": top_safety_layer,
        "top_safety_sep": safety_ranked[0][1],

        "top_epistemic_layer": top_epistemic_layer,
        "top_epistemic_sep": epistemic_ranked[0][1],

        "top_capability_layer": top_capability_layer,
        "top_capability_sep": capability_ranked[0][1],

        # Original metric retained.
        "cosine_at_top_safety":
            top_safety_comparison["safety_epistemic_cosine"],
        "angle_at_top_safety":
            top_safety_comparison["safety_epistemic_angle"],

        "mean_safety_epistemic_cosine": mean_se,
        "mean_safety_capability_cosine": mean_sc,
        "mean_epistemic_capability_cosine": mean_ec,

        "safety_capability_at_top_capability":
            top_capability_comparison["safety_capability_cosine"],
        "epistemic_capability_at_top_capability":
            top_capability_comparison["epistemic_capability_cosine"],

        "safety_top5": [
            {"name": n, "score": s}
            for n, s in safety_ranked[:5]
        ],
        "epistemic_top5": [
            {"name": n, "score": s}
            for n, s in epistemic_ranked[:5]
        ],
        "capability_top5": [
            {"name": n, "score": s}
            for n, s in capability_ranked[:5]
        ],

        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model replication")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt-limit", type=int, default=200)
    parser.add_argument(
        "--capability-prompt-limit", type=int, default=160
    )
    parser.add_argument(
        "--capability-split",
        choices=["all", "train", "eval"],
        default="train",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Load prompts once (shared across all models)
    safety_groups = load_safety_prompts(
        limit=args.prompt_limit, seed=args.seed
    )
    epistemic_groups = load_epistemic_prompts(
        limit=args.prompt_limit, seed=args.seed
    )
    capability_groups = load_capability_prompts(
        limit=args.capability_prompt_limit,
        seed=args.seed,
        split=args.capability_split,
    )

    safety_a = prompts_to_text_list(safety_groups["harmful"])
    safety_b = prompts_to_text_list(safety_groups["benign"])
    epistemic_a = prompts_to_text_list(epistemic_groups["unanswerable"])
    epistemic_b = prompts_to_text_list(epistemic_groups["answerable"])
    capability_a = prompts_to_text_list(
        capability_groups["incapable"]
    )
    capability_b = prompts_to_text_list(
        capability_groups["capable"]
    )

    all_results = []

    for model_id in args.models:
        logger.info("\n{'='*60}")
        logger.info("Model: %s", model_id)
        logger.info("{'='*60}")

        t0 = time.time()
        result = run_single_model(
            model_id,
            safety_a, safety_b,
            epistemic_a, epistemic_b,
            capability_a, capability_b,
            load_in_4bit=args.load_in_4bit,
        )
        result["elapsed_seconds"] = time.time() - t0
        all_results.append(result)

        # Save per-model result
        safe_name = model_id.replace("/", "_").replace("-", "_")
        write_json(
            str(Path(args.output_dir) / f"{safe_name}_comparison.json"),
            result,
        )

    # Summary table
    logger.info("\n=== CROSS-MODEL SUMMARY ===")
    logger.info(
        "%-40s | Top Safety | Top Epist | Cosine | Angle",
        "Model",
    )
    logger.info("-" * 90)
    for r in all_results:
        logger.info(
            "%-40s | %-10s | %-9s | %.4f | %.1f°",
            r["model_id"],
            r["top_safety_layer"],
            r["top_epistemic_layer"],
            r["cosine_at_top_safety"],
            r["angle_at_top_safety"],
        )

    # Save combined
    combined = {
        "artifact_type": "cross_model_capability_replication",
        "models": args.models,
        "prompt_limit": args.prompt_limit,
        "capability_prompt_limit": args.capability_prompt_limit,
        "capability_split": args.capability_split,
        "seed": args.seed,
        "results": all_results,
        "summary": [
            {
                "model_id": r["model_id"],
                "top_safety_layer": r["top_safety_layer"],
                "top_epistemic_layer": r["top_epistemic_layer"],
                "cosine_at_top_safety":
                    r["cosine_at_top_safety"],
                "angle_at_top_safety":
                    r["angle_at_top_safety"],
                "top_capability_layer":
                    r["top_capability_layer"],
                "top_capability_sep":
                    r["top_capability_sep"],
                "mean_safety_epistemic_cosine":
                    r["mean_safety_epistemic_cosine"],
                "mean_safety_capability_cosine":
                    r["mean_safety_capability_cosine"],
                "mean_epistemic_capability_cosine":
                    r["mean_epistemic_capability_cosine"],
                "safety_capability_at_top_capability":
                    r["safety_capability_at_top_capability"],
                "epistemic_capability_at_top_capability":
                    r["epistemic_capability_at_top_capability"],
            }
            for r in all_results
        ],
    }

    write_json(str(Path(args.output_dir) / "combined_results.json"), combined)
    logger.info("Combined results saved to %s/combined_results.json", args.output_dir)


if __name__ == "__main__":
    main()

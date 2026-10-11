#!/usr/bin/env python3
"""Experiment 3: Quantization perturbation sweep.

Measures how different quantization methods perturb the safety and epistemic
directions. For each precision:
  1. Load model at that precision
  2. Extract both directions
  3. Compare to FP16 reference directions (cosine, norm ratio)
  4. Test cross-precision ablation: does FP16-extracted direction still work?

Usage:
    python scripts/quantization_sweep.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --fp16-safety-artifact artifacts/directions/llama31_8b_safety_fp16.json \
        --fp16-epistemic-artifact artifacts/directions/llama31_8b_epistemic_fp16.json \
        --output artifacts/quantization/llama31_8b_sweep.json
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
    load_capability_prompts,
    prompts_to_text_list,
)
from ddmi.editing.directions import (
    cosine_similarity,
    direction_from_contrast,
    l2_norm,
    separability_score,
)
from ddmi.models.hooks import collect_activations_batched, get_residual_stream_names
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, read_json, set_seed, write_json

logger = configure_logging(name="quantization_sweep")


PRECISION_CONFIGS = {
    "fp16": {"load_in_4bit": False, "load_in_8bit": False, "torch_dtype": "float16"},
    "bnb_nf4": {"load_in_4bit": True, "load_in_8bit": False, "torch_dtype": "float16"},
    "bnb_int8": {"load_in_4bit": False, "load_in_8bit": True, "torch_dtype": "float16"},
}


def load_reference_direction(artifact: Dict[str, Any], layer_name: str) -> Dict[str, Any]:
    """Extract the saved direction payload for one layer from a direction artifact."""
    directions = artifact.get("directions", {})
    if layer_name not in directions:
        raise KeyError(f"Layer {layer_name!r} not found in artifact")
    return directions[layer_name]


def extract_directions_at_layer(
    model: Any,
    tokenizer: Any,
    layer_name: str,
    incapable: List[str],
    capable: List[str],
    max_input_length: int = 512,
) -> Dict[str, Any]:
    """Extract capability direction at one residual-stream layer."""

    acts_incapable = collect_activations_batched(
        model,
        tokenizer,
        incapable,
        [layer_name],
        max_input_length,
    )
    acts_capable = collect_activations_batched(
        model,
        tokenizer,
        capable,
        [layer_name],
        max_input_length,
    )

    c_dir = direction_from_contrast(
        acts_incapable[layer_name],
        acts_capable[layer_name],
    )

    c_sep = separability_score(
        acts_incapable[layer_name],
        acts_capable[layer_name],
    )

    from ddmi.editing.directions import difference_of_means

    c_raw = l2_norm(
        difference_of_means(
            acts_incapable[layer_name],
            acts_capable[layer_name],
        )
    )

    return {
        "capability_direction": c_dir,
        "capability_separability": c_sep,
        "capability_raw_norm": c_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capability quantization perturbation sweep"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--fp16-capability-artifact",
        required=True,
        help="FP16 capability direction artifact",
    )
    parser.add_argument(
        "--target-layer",
        default="model.layers.24",
        help="Frozen capability layer",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--capability-split",
        choices=["all", "train", "eval"],
        default="train",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    groups = load_capability_prompts(
        limit=args.prompt_limit,
        seed=args.seed,
        split=args.capability_split,
    )

    incapable = prompts_to_text_list(groups["incapable"])
    capable = prompts_to_text_list(groups["capable"])

    target_layer = args.target_layer

    fp16_artifact = read_json(
        args.fp16_capability_artifact
    )

    capability_ref = load_reference_direction(
        fp16_artifact,
        target_layer,
    )

    results = {
        "fp16": {
            "capability_direction":
                capability_ref["direction"],
            "capability_separability":
                capability_ref.get("separability"),
            "capability_raw_norm":
                capability_ref.get("raw_norm"),
        }
    }

    logger.info(
        "Using provided FP16 capability artifact "
        "as baseline at %s",
        target_layer,
    )

    precisions_to_run = (
        (name, cfg)
        for name, cfg in PRECISION_CONFIGS.items()
        if name != "fp16"
    )

    for precision, precision_kwargs in precisions_to_run:
        logger.info("\n=== Precision: %s ===", precision)

        config = ModelLoadConfig(
            model_id=args.model_id,
            **precision_kwargs,
            attn_implementation="eager",
        )

        t0 = time.time()

        try:
            model, tokenizer = load_model_and_tokenizer(
                config
            )
        except Exception as e:
            logger.warning(
                "Failed to load at %s: %s",
                precision,
                e,
            )
            results[precision] = {
                "error": str(e)
            }
            continue

        logger.info(
            "Model loaded in %.1fs",
            time.time() - t0,
        )

        t0 = time.time()

        layer_results = extract_directions_at_layer(
            model,
            tokenizer,
            target_layer,
            incapable,
            capable,
        )

        logger.info(
            "Capability direction extracted in %.1fs",
            time.time() - t0,
        )

        results[precision] = layer_results

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fp16_c = results["fp16"]["capability_direction"]

    drift_table = []

    for precision, res in results.items():
        if "error" in res:
            continue

        c_cos = cosine_similarity(
            fp16_c,
            res["capability_direction"],
        )

        c_norm_ratio = (
            l2_norm(res["capability_direction"])
            / max(l2_norm(fp16_c), 1e-12)
        )

        row = {
            "precision": precision,
            "capability_cosine_vs_fp16": c_cos,
            "capability_norm_ratio": c_norm_ratio,
            "capability_separability":
                res["capability_separability"],
            "capability_raw_norm":
                res["capability_raw_norm"],
        }

        drift_table.append(row)

        sep = res["capability_separability"]
        sep_text = "NA" if sep is None else f"{sep:.4f}"

        logger.info(
            "%s: capability_cos=%.4f, "
            "norm_ratio=%.4f, sep=%s",
            precision,
            c_cos,
            c_norm_ratio,
            sep_text,
        )

    for precision in results:
        if "error" not in results[precision]:
            results[precision].pop(
                "capability_direction",
                None,
            )

    artifact = {
        "artifact_type":
            "capability_quantization_sweep",
        "model_id": args.model_id,
        "target_layer": target_layer,
        "prompt_limit": args.prompt_limit,
        "capability_split":
            args.capability_split,
        "seed": args.seed,
        "precisions":
            list(PRECISION_CONFIGS.keys()),
        "fp16_capability_artifact":
            args.fp16_capability_artifact,
        "drift_table": drift_table,
        "per_precision_results": results,
    }

    write_json(args.output, artifact)

    logger.info(
        "Sweep results saved to %s",
        args.output,
    )


if __name__ == "__main__":
    main()

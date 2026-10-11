#!/usr/bin/env python3

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import (
    load_capability_prompts,
    prompts_to_text_list,
)
from ddmi.evaluation.probes import fit_and_evaluate_binary_probe
from ddmi.models.hooks import (
    collect_activations_batched,
    get_residual_stream_names,
)
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, read_json, set_seed, write_json

logger = configure_logging(name="capability_probe")


def main():
    parser = argparse.ArgumentParser(
        description="Capability-refusal linear probe"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--direction-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    artifact = read_json(args.direction_artifact)
    selected_layer = artifact["ranked_layers"][0]["name"]

    train_groups = load_capability_prompts(
        limit=160,
        seed=args.seed,
        split="train",
    )
    eval_groups = load_capability_prompts(
        limit=40,
        seed=args.seed,
        split="eval",
    )

    train_pos = prompts_to_text_list(train_groups["incapable"])
    train_neg = prompts_to_text_list(train_groups["capable"])
    eval_pos = prompts_to_text_list(eval_groups["incapable"])
    eval_neg = prompts_to_text_list(eval_groups["capable"])

    config = ModelLoadConfig(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype="float16",
        attn_implementation="eager",
        trust_remote_code="phi" in args.model_id.lower(),
    )

    logger.info("Loading model: %s", args.model_id)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Loaded in %.1fs", time.time() - t0)

    layers = get_residual_stream_names(model)

    logger.info("Collecting train activations...")
    train_pos_acts = collect_activations_batched(
        model, tokenizer, train_pos, layers,
        max_input_length=args.max_input_length,
    )
    train_neg_acts = collect_activations_batched(
        model, tokenizer, train_neg, layers,
        max_input_length=args.max_input_length,
    )

    logger.info("Collecting held-out activations...")
    eval_pos_acts = collect_activations_batched(
        model, tokenizer, eval_pos, layers,
        max_input_length=args.max_input_length,
    )
    eval_neg_acts = collect_activations_batched(
        model, tokenizer, eval_neg, layers,
        max_input_length=args.max_input_length,
    )

    # fit_and_evaluate_binary_probe only does its own random split,
    # so import its underlying probe implementation through a small
    # explicit train/eval helper.
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    results = []

    def stack_acts(xs):
        """Convert nested activation outputs into [n_examples, hidden_dim]."""

        tensors = []

        def collect(x):
            if isinstance(x, torch.Tensor):
                t = x.detach().cpu().float()

                if t.ndim == 1:
                    t = t.unsqueeze(0)
                elif t.ndim > 2:
                    t = t.reshape(-1, t.shape[-1])

                tensors.append(t)
                return

            if isinstance(x, (list, tuple)):
                # Plain numeric vector.
                if x and all(
                    isinstance(v, (int, float))
                    for v in x
                ):
                    tensors.append(
                        torch.tensor(x, dtype=torch.float32)
                        .unsqueeze(0)
                    )
                    return

                for y in x:
                    collect(y)
                return

            raise TypeError(
                f"Unsupported activation type: {type(x)}"
            )

        collect(xs)

        if not tensors:
            raise ValueError("No activations found")

        hidden_dims = {t.shape[-1] for t in tensors}
        if len(hidden_dims) != 1:
            raise ValueError(
                f"Inconsistent hidden dimensions: {hidden_dims}"
            )

        return torch.cat(tensors, dim=0)

    for layer in layers:
        train_pos_tensor = stack_acts(train_pos_acts[layer])
        train_neg_tensor = stack_acts(train_neg_acts[layer])
        eval_pos_tensor = stack_acts(eval_pos_acts[layer])
        eval_neg_tensor = stack_acts(eval_neg_acts[layer])

        x_train = torch.cat([
            train_pos_tensor,
            train_neg_tensor,
        ], dim=0).float().cpu().numpy()

        y_train = np.array(
            [1] * len(train_pos_tensor)
            + [0] * len(train_neg_tensor)
        )

        x_eval = torch.cat([
            eval_pos_tensor,
            eval_neg_tensor,
        ], dim=0).float().cpu().numpy()

        y_eval = np.array(
            [1] * len(eval_pos_tensor)
            + [0] * len(eval_neg_tensor)
        )

        clf = LogisticRegression(
            max_iter=2000,
            random_state=args.seed,
        )
        clf.fit(x_train, y_train)

        train_acc = accuracy_score(
            y_train, clf.predict(x_train)
        )
        eval_acc = accuracy_score(
            y_eval, clf.predict(x_eval)
        )

        results.append({
            "layer_name": layer,
            "layer_index": int(layer.split(".")[-1]),
            "train_accuracy": float(train_acc),
            "eval_accuracy": float(eval_acc),
        })

    results.sort(
        key=lambda x: x["eval_accuracy"],
        reverse=True,
    )

    selected = next(
        x for x in results
        if x["layer_name"] == selected_layer
    )

    out = {
        "artifact_type": "capability_linear_probe",
        "model_id": args.model_id,
        "train_split": "train",
        "eval_split": "eval",
        "train_prompts": 160,
        "eval_prompts": 40,
        "selected_layer": selected_layer,
        "summary": {
            "best_layer": results[0]["layer_name"],
            "best_eval_accuracy": results[0]["eval_accuracy"],
            "selected_layer_train_accuracy":
                selected["train_accuracy"],
            "selected_layer_eval_accuracy":
                selected["eval_accuracy"],
        },
        "layers": results,
    }

    write_json(args.output, out)

    print("\n=== CAPABILITY PROBE ===")
    print("Best layer:", results[0]["layer_name"])
    print("Best eval accuracy:",
          f'{results[0]["eval_accuracy"]:.3f}')
    print("Selected layer:", selected_layer)
    print("Selected train accuracy:",
          f'{selected["train_accuracy"]:.3f}')
    print("Selected eval accuracy:",
          f'{selected["eval_accuracy"]:.3f}')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

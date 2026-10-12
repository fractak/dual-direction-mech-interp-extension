#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.data.loaders import (
    load_epistemic_prompts,
    load_safety_prompts,
    prompts_to_text_list,
)
from ddmi.evaluation.detectors import abstention_rate, refusal_rate
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Behavioral sanity check for Instruct vs safety-LoRA checkpoint."
    )
    p.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--adapter",
        default="artifacts/safety_finetuning/llama31_8b_instruct_safety_lora",
    )
    p.add_argument(
        "--output",
        default="artifacts/safety_finetuning/behavior_B_vs_C.json",
    )
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    return p.parse_args()


def load_prompts(limit, seed):
    safety = load_safety_prompts(limit=limit, seed=seed)
    epistemic = load_epistemic_prompts(limit=limit, seed=seed)

    return {
        "harmful": prompts_to_text_list(safety["harmful"]),
        "benign": prompts_to_text_list(safety["benign"]),
        "unanswerable": prompts_to_text_list(epistemic["unanswerable"]),
        "answerable": prompts_to_text_list(epistemic["answerable"]),
    }


def load_checkpoint(model_id, adapter=None):
    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id=model_id,
            load_in_4bit=True,
            torch_dtype="float16",
            attn_implementation="eager",
        )
    )

    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    return model, tokenizer


def evaluate(model, tokenizer, prompts, gen_config):
    responses = {}

    for name, texts in prompts.items():
        print(f"  generating {name}: {len(texts)}")
        responses[name] = generate_batch(
            model,
            tokenizer,
            texts,
            config=gen_config,
        )

    metrics = {
        "harmful_refusal_rate": refusal_rate(responses["harmful"]),
        "benign_refusal_rate": refusal_rate(responses["benign"]),
        "unanswerable_abstention_rate": abstention_rate(
            responses["unanswerable"]
        ),
        "answerable_abstention_rate": abstention_rate(
            responses["answerable"]
        ),
    }

    return metrics, responses


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    prompts = load_prompts(args.limit, args.seed)

    print("Prompt counts:")
    for k, v in prompts.items():
        print(f"  {k:14} {len(v)}")

    gen_config = TextGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    print("\nEvaluating B: Instruct...")
    model_b, tokenizer_b = load_checkpoint(args.model_id)
    metrics_b, responses_b = evaluate(
        model_b, tokenizer_b, prompts, gen_config
    )
    release_model(model_b)
    del tokenizer_b

    print("\nEvaluating C: Instruct + safety LoRA...")
    model_c, tokenizer_c = load_checkpoint(
        args.model_id,
        adapter=args.adapter,
    )
    metrics_c, responses_c = evaluate(
        model_c, tokenizer_c, prompts, gen_config
    )
    release_model(model_c)
    del tokenizer_c

    result = {
        "config": {
            "model_B": args.model_id,
            "model_C_adapter": args.adapter,
            "limit": args.limit,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
        },
        "metrics": {
            "B_instruct": metrics_b,
            "C_safety_lora": metrics_c,
        },
        "responses": {
            "B_instruct": responses_b,
            "C_safety_lora": responses_c,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("\nRESULTS")
    for name, metrics in result["metrics"].items():
        print(name)
        for key, value in metrics.items():
            print(f"  {key:32} {value:.3f}")

    print("\nSaved:", out)


if __name__ == "__main__":
    main()

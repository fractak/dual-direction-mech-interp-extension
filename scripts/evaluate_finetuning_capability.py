#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.data.loaders import load_capability_prompts, prompts_to_text_list
from ddmi.evaluation.detectors import is_capability_refusal
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        default="artifacts/safety_finetuning/llama31_8b_instruct_safety_lora",
    )
    p.add_argument(
        "--output",
        default="artifacts/safety_finetuning/capability_behavior_B_vs_C.json",
    )
    return p.parse_args()


def load_checkpoint(adapter=None):
    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            load_in_4bit=True,
            torch_dtype="float16",
            attn_implementation="eager",
        )
    )

    if adapter:
        peft = PeftModel.from_pretrained(model, adapter)
        model = peft.base_model.model
        model.eval()

    return model, tokenizer


def evaluate(model, tokenizer, prompts, config):
    responses = generate_batch(
        model,
        tokenizer,
        prompts,
        config=config,
    )
    labels = [is_capability_refusal(r) for r in responses]
    return responses, labels


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    data = load_capability_prompts(
        limit=120,
        seed=42,
        split="expanded_eval",
    )

    samples = data["incapable"]
    prompts = prompts_to_text_list(samples)

    config = TextGenerationConfig(
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    print("Capability prompts:", len(prompts))

    print("\nLoading B...")
    model_b, tokenizer_b = load_checkpoint()
    responses_b, labels_b = evaluate(
        model_b,
        tokenizer_b,
        prompts,
        config,
    )
    release(model_b)
    del tokenizer_b

    print("Loading C...")
    model_c, tokenizer_c = load_checkpoint(args.adapter)
    responses_c, labels_c = evaluate(
        model_c,
        tokenizer_c,
        prompts,
        config,
    )
    release(model_c)
    del tokenizer_c

    rows = []

    for sample, prompt, rb, lb, rc, lc in zip(
        samples,
        prompts,
        responses_b,
        labels_b,
        responses_c,
        labels_c,
    ):
        rows.append({
            "id": sample.id,
            "prompt": prompt,
            "B": {
                "detector_label": lb,
                "response": rb,
            },
            "C": {
                "detector_label": lc,
                "response": rc,
            },
        })

    result = {
        "config": {
            "n": len(prompts),
            "seed": 42,
            "split": "expanded_eval",
            "max_new_tokens": 128,
            "temperature": 0.0,
        },
        "rates": {
            "B": sum(labels_b) / len(labels_b),
            "C": sum(labels_c) / len(labels_c),
        },
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("\nDETECTOR RATES")
    print(f"B: {result['rates']['B']:.3f}")
    print(f"C: {result['rates']['C']:.3f}")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()

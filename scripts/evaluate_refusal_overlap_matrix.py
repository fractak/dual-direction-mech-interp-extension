#!/usr/bin/env python3

import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.evaluation.detectors import (
    is_refusal,
    is_capability_refusal,
    is_abstention,
)

from ddmi.models.generation import (
    TextGenerationConfig,
    generate_batch,
)

from ddmi.models.loader import (
    ModelLoadConfig,
    load_model_and_tokenizer,
)

DATA = Path(
    "data/refusal_overlap/"
    "refusal_overlap_matrix.json"
)

OUT = Path(
    "artifacts/refusal_overlap/"
    "baseline_behavior.json"
)

ADAPTER = Path(
    "artifacts/safety_finetuning/"
    "llama31_8b_instruct_safety_lora"
)

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
        peft = PeftModel.from_pretrained(
            model,
            adapter,
        )
        model = peft.base_model.model
        model.eval()

    return model, tokenizer


def release(model):
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_rows():
    d = json.loads(DATA.read_text())
    return d["rows"]


def evaluate(model, tokenizer, rows, config):
    prompts = [x["prompt"] for x in rows]

    responses = generate_batch(
        model,
        tokenizer,
        prompts,
        config=config,
    )

    out = []

    for row, response in zip(rows, responses):
        out.append({
            **row,
            "response": response,
            "refusal": is_refusal(response),
            "capability_refusal": (
                is_capability_refusal(response)
            ),
            "epistemic_abstention": (
                is_abstention(response)
            ),
        })

    return out

def main():
    rows = load_rows()

    print("prompts:", len(rows))

    config = TextGenerationConfig(
        max_new_tokens=128,
        temperature=0.0,
        do_sample=False,
    )

    results = {}

    for checkpoint, adapter in [
        ("B_instruct", None),
        ("C_safety_lora", str(ADAPTER)),
    ]:
        print("\nLoading", checkpoint)

        model, tokenizer = load_checkpoint(adapter)

        results[checkpoint] = evaluate(
            model,
            tokenizer,
            rows,
            config,
        )

        release(model)
        del tokenizer

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "n": len(rows),
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "do_sample": False,
                },
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )

    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()

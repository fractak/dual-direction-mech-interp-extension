import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.data.loaders import load_capability_prompts, prompts_to_text_list
from ddmi.editing.apply_edit import ActivationDirectionAblator, EditSpec
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer

ROOT = Path("artifacts/safety_finetuning")
LAYER = "model.layers.20"
STRENGTH = 1.5
OUT = ROOT / "capability_C_safety_orthogonal_layer20.json"

def get_direction(filename):
    data = json.loads((ROOT / filename).read_text())
    assert LAYER in data["directions"], filename
    vector = torch.tensor(
        data["directions"][LAYER]["direction"],
        dtype=torch.float32,
    )
    assert vector.ndim == 1
    assert torch.isfinite(vector).all()
    return vector / vector.norm()

capability = get_direction("C_capability.json")
safety = get_direction("C_safety.json")

safety_orthogonal = safety - torch.dot(safety, capability) * capability
safety_orthogonal = safety_orthogonal / safety_orthogonal.norm()

directions = {
    "safety_orthogonal": safety_orthogonal,
}

print("Cosine(safety, capability):",
      torch.dot(safety, capability).item())
print("Cosine(orthogonal safety, capability):",
      torch.dot(safety_orthogonal, capability).item())

for name, vector in directions.items():
    cosine = torch.dot(vector, capability).item()
    print(
        f"{name}: layer={LAYER}, "
        f"norm={vector.norm().item():.6f}, "
        f"cosine_with_capability={cosine:.6f}",
        flush=True,
    )

data = load_capability_prompts(
    limit=120, seed=42, split="expanded_eval"
)
prompts = prompts_to_text_list(data["incapable"])
assert len(prompts) == 60

model, tokenizer = load_model_and_tokenizer(
    ModelLoadConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        torch_dtype="float16",
        attn_implementation="eager",
    )
)

model = PeftModel.from_pretrained(
    model,
    str(ROOT / "llama31_8b_instruct_safety_lora"),
).base_model.model
model.eval()

config = TextGenerationConfig(
    max_new_tokens=128,
    temperature=0.0,
    do_sample=False,
)

results = {}

for name, vector in directions.items():
    print(f"\nGenerating C + {name} ablation...", flush=True)

    ablator = ActivationDirectionAblator(
        module_names=[LAYER],
        direction=vector.tolist(),
        spec=EditSpec(strength=STRENGTH),
    ).attach(model)

    try:
        with torch.inference_mode():
            responses = generate_batch(
                model, tokenizer, prompts, config=config
            )
    finally:
        ablator.close()

    assert len(responses) == 60

    results[name] = [
        {
            "id": f"C-{name}-{i+1:03d}",
            "source_index": i,
            "prompt": prompt,
            "response": response,
        }
        for i, (prompt, response) in enumerate(
            zip(prompts, responses)
        )
    ]

    print(f"Completed: {len(responses)} responses", flush=True)

    # Save after each condition so completed work is preserved.
    OUT.write_text(json.dumps({
        "config": {
            "checkpoint": "C_safety_lora",
            "layer": LAYER,
            "strength": STRENGTH,
            "prompt_seed": 42,
            "split": "expanded_eval",
            "n_per_condition": 60,
            "direction_artifacts": {
                "safety_orthogonal": "C_safety.json (orthogonalized to C capability)",
                "epistemic": "C_epistemic.json",
            },
        },
        "results": results,
    }, indent=2) + "\n")

print("\nSaved:", OUT)
print("Conditions:", list(results))

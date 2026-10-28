import json
from pathlib import Path

import torch
from peft import PeftModel

from ddmi.data.loaders import (
    load_capability_prompts,
    prompts_to_text_list,
)
from ddmi.editing.apply_edit import ActivationDirectionAblator, EditSpec
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer

ROOT = Path("artifacts/safety_finetuning")
OUT = ROOT / "capability_C_random_layer20.json"
SEED = 2026
STRENGTH = 1.5

direction_data = json.loads(
    (ROOT / "C_capability.json").read_text()
)
layer = direction_data["ranked_layers"][0]["name"]
assert layer == "model.layers.20", layer

original = torch.tensor(
    direction_data["directions"][layer]["direction"],
    dtype=torch.float32,
)

generator = torch.Generator(device="cpu").manual_seed(SEED)
random_direction = torch.randn(
    original.shape, generator=generator
)
random_direction = (
    random_direction
    / random_direction.norm()
    * original.norm()
)

print("Layer:", layer)
print("Random seed:", SEED)
print("Original direction norm:", original.norm().item())
print("Random direction norm:", random_direction.norm().item())
print(
    "Cosine similarity:",
    torch.nn.functional.cosine_similarity(
        original.flatten()[None],
        random_direction.flatten()[None],
    ).item(),
)

data = load_capability_prompts(
    limit=120,
    seed=42,
    split="expanded_eval",
)
samples = data["incapable"]
prompts = prompts_to_text_list(samples)
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

ablator = ActivationDirectionAblator(
    module_names=[layer],
    direction=random_direction.tolist(),
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

rows = [
    {
        "id": f"C-random-{i+1:03d}",
        "prompt": prompt,
        "response": response,
    }
    for i, (prompt, response) in enumerate(
        zip(prompts, responses)
    )
]

OUT.write_text(json.dumps({
    "config": {
        "checkpoint": "C_safety_lora",
        "layer": layer,
        "strength": STRENGTH,
        "random_seed": SEED,
        "prompt_seed": 42,
        "split": "expanded_eval",
        "n": len(rows),
        "direction_norm": original.norm().item(),
    },
    "rows": rows,
}, indent=2) + "\n")

print("\nSaved:", OUT)
print("Generated responses:", len(rows))
print("First response:", repr(responses[0][:300]))

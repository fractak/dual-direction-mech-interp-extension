"""One configurable runner for capability edits, controls, and overlap tests.

Planning uses the standard library. GPU/model imports occur only when executing.
New responses are unannotated and never overwrite the frozen paper evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ddmi.checkpoints import CHECKPOINTS, MODEL_ID
from ddmi.paper.evidence import read_json

OVERLAP_CONDITIONS = {
    "safety_only": [(), ("safety",)],
    "capability_only": [(), ("capability",)],
    "epistemic_only": [(), ("epistemic",)],
    "safety_capability": [(), ("safety",), ("capability",), ("safety", "capability")],
    "safety_epistemic": [(), ("safety",), ("epistemic",), ("safety", "epistemic")],
    "capability_epistemic": [(), ("capability",), ("epistemic",), ("capability", "epistemic")],
    "triple_overlap": [
        (),
        ("safety",),
        ("capability",),
        ("epistemic",),
        ("safety", "capability"),
        ("safety", "epistemic"),
        ("capability", "epistemic"),
        ("safety", "capability", "epistemic"),
    ],
}


def direction_spec(root, checkpoint, name, layer):
    artifact_path = Path(root) / f"evidence/directions/{checkpoint}/{name}.json"
    artifact = read_json(artifact_path)
    if layer not in artifact["directions"]:
        raise ValueError(f"Missing layer {layer} in {artifact_path}")
    return {
        "name": name,
        "checkpoint": checkpoint,
        "layer": layer,
        "artifact": str(artifact_path),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }


def build_plan(
    root,
    checkpoints,
    conditions,
    strengths,
    random_seeds,
    layer,
    direction_checkpoint,
    study="capability",
):
    if any(not math.isfinite(s) or s < 0 for s in strengths):
        raise ValueError("Strengths must be finite and nonnegative")
    plan = []
    for checkpoint in checkpoints:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint: {checkpoint}")
        if study == "overlap":
            for category, edits in OVERLAP_CONDITIONS.items():
                for names in edits:
                    for strength in strengths if names else [0.0]:
                        directions = []
                        for name in names:
                            peak = {
                                "safety": 22,
                                "epistemic": 9,
                                "capability": 24 if checkpoint == "original_instruct" else 20,
                            }[name]
                            directions.append(
                                direction_spec(root, checkpoint, name, f"model.layers.{peak}")
                            )
                        condition = "ablate_" + "_".join(names) if names else "baseline"
                        plan.append(
                            {
                                "checkpoint": checkpoint,
                                "prompt_type": category,
                                "condition": condition,
                                "strength": strength,
                                "directions": directions,
                            }
                        )
            continue
        for condition in conditions:
            if condition not in (
                "baseline",
                "capability",
                "safety",
                "epistemic",
                "safety_orthogonal",
                "random",
            ):
                raise ValueError(f"Unknown condition: {condition}")
            for strength in strengths if condition != "baseline" else [0.0]:
                seeds = random_seeds if condition == "random" else [None]
                for seed in seeds:
                    names = (
                        []
                        if condition in ("baseline", "random")
                        else (
                            ["safety", "capability"]
                            if condition == "safety_orthogonal"
                            else [condition]
                        )
                    )
                    directions = [
                        direction_spec(root, direction_checkpoint, name, layer) for name in names
                    ]
                    # A random vector has the archived capability vector's dimension.
                    if condition == "random":
                        directions = [
                            direction_spec(root, direction_checkpoint, "capability", layer)
                        ]
                    plan.append(
                        {
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "strength": strength,
                            "layer": layer,
                            "random_seed": seed,
                            "directions": directions,
                        }
                    )
    if not plan:
        raise ValueError("No experiment conditions selected")
    return plan


def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def execute(plan, root, output, adapter_path, seed=42, study="capability"):
    import gc

    import torch

    from ddmi.editing.apply_edit import ActivationDirectionAblator, EditSpec
    from ddmi.models.generation import TextGenerationConfig, generate_batch
    from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
    from ddmi.utils.io import set_seed

    set_seed(seed)
    if study == "capability":
        pairs = read_json(Path(root) / "data/capability_eval_expanded.json")
        prompts_by_type = {None: [{"id": row["id"], "prompt": row["incapable"]} for row in pairs]}
    else:
        matrix = read_json(Path(root) / "data/refusal_overlap/refusal_overlap_matrix.json")
        prompts_by_type = {
            category: [row for row in matrix["rows"] if row["prompt_type"] == category]
            for category in OVERLAP_CONDITIONS
        }
    result = {
        "schema_version": 1,
        "study": study,
        "model_id": MODEL_ID,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "generation": {
            "seed": seed,
            "max_new_tokens": 128,
            "max_input_length": 512,
            "do_sample": False,
            "precision": "bnb_nf4",
            "compute_dtype": "float16",
            "attention": "eager",
        },
        "annotation_status": "unreviewed_not_paper_results",
        "conditions": [],
    }
    for checkpoint in dict.fromkeys(item["checkpoint"] for item in plan):
        model, tokenizer = load_model_and_tokenizer(
            ModelLoadConfig(
                model_id=MODEL_ID,
                load_in_4bit=True,
                torch_dtype="float16",
                attn_implementation="eager",
            )
        )
        if CHECKPOINTS[checkpoint]["requires_adapter"]:
            from peft import PeftModel

            wrapped = PeftModel.from_pretrained(model, str(adapter_path))
            model = wrapped.base_model.model
            del wrapped
            model.eval()
        try:
            for item in plan:
                if item["checkpoint"] != checkpoint:
                    continue
                print(f"{checkpoint}: {item['condition']}, strength={item['strength']}", flush=True)
                hooks = []
                try:
                    vectors = []
                    for spec in item["directions"]:
                        path = Path(spec["artifact"])
                        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                            raise ValueError(f"Direction changed after planning: {path}")
                        data = read_json(path)
                        vector = torch.tensor(
                            data["directions"][spec["layer"]]["direction"], dtype=torch.float32
                        )
                        if not torch.isfinite(vector).all() or vector.norm() == 0:
                            raise ValueError("Invalid direction vector")
                        vectors.append(vector / vector.norm())
                    if item["condition"] == "safety_orthogonal":
                        safety, capability = vectors
                        vector = safety - torch.dot(safety, capability) * capability
                        if vector.norm() < 1e-8:
                            raise ValueError("Orthogonalized safety direction is degenerate")
                        vectors = [vector / vector.norm()]
                    if item["condition"] == "random":
                        generator = torch.Generator(device="cpu").manual_seed(item["random_seed"])
                        vector = torch.randn(vectors[0].shape, generator=generator)
                        vectors = [vector / vector.norm()]
                    for spec, vector in zip(item["directions"], vectors):
                        hook = ActivationDirectionAblator(
                            [spec["layer"]], vector.tolist(), EditSpec(strength=item["strength"])
                        ).attach(model)
                        hooks.append(hook)
                    prompt_rows = prompts_by_type[item.get("prompt_type")]
                    responses = generate_batch(
                        model,
                        tokenizer,
                        [r["prompt"] for r in prompt_rows],
                        config=TextGenerationConfig(),
                    )
                finally:
                    for hook in reversed(hooks):
                        hook.close()
                result["conditions"].append(
                    {
                        **item,
                        "rows": [
                            {
                                "prompt_id": row["id"],
                                "source_index": i,
                                "prompt": row["prompt"],
                                "response": response,
                                "annotation": None,
                            }
                            for i, (row, response) in enumerate(zip(prompt_rows, responses))
                        ],
                    }
                )
                write_atomic(output, result)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return result

"""Frozen prospective plans and measured generation, separate from archived evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from ddmi.checkpoints import MODEL_ID
from ddmi.paper.evidence import read_json, verify_checksums
from ddmi.provenance import adapter_manifest, runtime_metadata, sha256_file


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def create_plan(root, config, prompt_file=None, direction_root=None):
    root = Path(root)
    model_id = config.get("model_id", MODEL_ID)
    width = config.get("hidden_size", 4096)
    layer_count = config.get("num_hidden_layers", 32)
    verify_checksums(root)
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("model_revision must be an immutable 40-character Hub commit SHA")
    if config["precision"] not in ("nf4", "fp16", "bf16"):
        raise ValueError("precision must be nf4, fp16 or bf16")
    if config["purpose"] not in ("pilot", "confirmatory"):
        raise ValueError("purpose must be pilot or confirmatory")
    if config["purpose"] == "confirmatory" and (prompt_file is None or direction_root is None):
        raise ValueError("Confirmatory plans require new prompt and direction files")
    prompt_path = Path(prompt_file) if prompt_file else root / "data/capability_eval_expanded.json"
    raw = read_json(prompt_path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Prompt file must be a nonempty JSON list")
    prompts = []
    for row in raw:
        text = row.get("prompt", row.get("incapable"))
        if not isinstance(text, str) or not text.strip() or not isinstance(row["id"], str):
            raise ValueError("Each prompt needs a string id and nonempty prompt")
        prompts.append({"id": row["id"], "prompt": text, "domain": row.get("domain")})
    if len({r["id"] for r in prompts}) != len(prompts) or len(
        {normalized(r["prompt"]) for r in prompts}
    ) != len(prompts):
        raise ValueError("Prompt IDs and normalized prompt strings must be unique")
    frozen = set()
    for filename in ("capability_train_pairs.json", "capability_eval_expanded.json"):
        for row in read_json(root / "data" / filename):
            frozen.update(normalized(row[key]) for key in ("capable", "incapable"))
    frozen.update(
        normalized(r["prompt"])
        for r in read_json(root / "data/safety_finetuning/train_sft.json")["rows"]
    )
    overlap = sum(normalized(r["prompt"]) in frozen for r in prompts)
    if config["purpose"] == "confirmatory" and (
        overlap or any(not isinstance(r["domain"], str) or not r["domain"] for r in prompts)
    ):
        raise ValueError(
            "Confirmatory prompts need domains and no exact overlap with frozen prompts"
        )
    if not config["layers"] or len(set(config["layers"])) != len(config["layers"]):
        raise ValueError("layers must be nonempty and unique")
    for layer in config["layers"]:
        if type(layer) is not int or not 0 <= layer < layer_count:
            raise ValueError(f"Layer indices must be integers in [0, {layer_count - 1}]")
    for key in ("projection_strengths", "relative_budgets"):
        values = config[key]
        if (
            not values
            or len(set(values)) != len(values)
            or any(not math.isfinite(x) or x <= 0 for x in values)
        ):
            raise ValueError(f"{key} must contain unique finite positive values")
    seeds = config["random_seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("random_seeds must be unique nonnegative integers")
    source_root = Path(direction_root) if direction_root else root / "evidence/directions"
    vectors, source_hashes, direction_metadata = {}, {}, {}
    for source in ("original_instruct", "safety_lora"):
        path = source_root / source / "capability.json"
        artifact = read_json(path)
        source_hashes[source] = sha256_file(path)
        direction_metadata[source] = {
            k: artifact.get(k)
            for k in ("checkpoint", "model_id", "model_revision", "adapter_manifest")
        }
        if config["purpose"] == "confirmatory":
            if (
                artifact.get("checkpoint") != source
                or artifact.get("model_id") != model_id
                or artifact.get("enable_thinking") != config.get("enable_thinking")
                or artifact.get("model_revision") != config["model_revision"]
            ):
                raise ValueError("Direction checkpoint/model revision does not match plan")
            if source == "safety_lora" and not artifact.get("adapter_manifest"):
                raise ValueError("Adapted directions must identify adapter file hashes")
            hashes = artifact.get("prompt_sha256", {})
            training_hashes = set(hashes.get("positive", []) + hashes.get("negative", []))
            if not hashes.get("positive") or not hashes.get("negative"):
                raise ValueError("New directions must record extraction prompt hashes")
            if training_hashes.intersection(
                hashlib.sha256(r["prompt"].encode()).hexdigest() for r in prompts
            ):
                raise ValueError("Evaluation prompts overlap direction extraction prompts")
        for layer in config["layers"]:
            vector = artifact["directions"][f"model.layers.{layer}"]["direction"]
            if len(vector) != width or not all(math.isfinite(x) for x in vector):
                raise ValueError("Expected finite directions matching model hidden size")
            norm = math.sqrt(sum(x * x for x in vector))
            if norm <= 0:
                raise ValueError("Zero direction")
            vectors[f"{source}:L{layer}"] = [x / norm for x in vector]
    conditions = []
    for receiver in ("original_instruct", "safety_lora"):
        for layer in config["layers"]:
            # Baseline observes both source vectors without modifying activations.
            conditions.append(
                {
                    "receiver": receiver,
                    "layer": layer,
                    "mode": "observe",
                    "amount": 0,
                    "source": "both",
                    "random_seed": None,
                }
            )
            for source in ("original_instruct", "safety_lora"):
                for mode, values in (
                    ("projection", config["projection_strengths"]),
                    ("relative_l2", config["relative_budgets"]),
                ):
                    for amount in values:
                        conditions.append(
                            {
                                "receiver": receiver,
                                "layer": layer,
                                "mode": mode,
                                "amount": amount,
                                "source": source,
                                "random_seed": None,
                            }
                        )
            for seed in seeds:
                for amount in config["relative_budgets"]:
                    conditions.append(
                        {
                            "receiver": receiver,
                            "layer": layer,
                            "mode": "relative_l2",
                            "amount": amount,
                            "source": "random",
                            "random_seed": seed,
                        }
                    )
    if "fixed_direction_receiver_files" in config:
        conditions = [
            c for c in conditions
            if (c["receiver"] == "safety_lora" or config["include_original_receiver"])
            and (c["mode"] == "observe" or (
                c["mode"] == "relative_l2" and c["source"] == "safety_lora"
            ))
        ]
    for index, condition in enumerate(conditions):
        condition["id"] = f"c{index:04d}"
    payload = {
        "schema_version": 1,
        "config": config,
        "model_id": model_id,
        "prompt_file_sha256": sha256_file(prompt_path),
        "exact_frozen_prompt_overlap": overlap,
        "direction_file_sha256": source_hashes,
        "direction_metadata": direction_metadata,
        "vectors": vectors,
        "prompts": prompts,
        "conditions": conditions,
        "planned_responses": len(prompts) * len(conditions),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Prospective only if frozen before inspecting new outcomes; "
        "exact-string guards do not exclude paraphrases or prior exposure.",
    }
    plan = {"plan_sha256": digest(payload), "payload": payload}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if digest(plan["payload"]) != plan["plan_sha256"]:
        raise ValueError("Plan checksum mismatch")
    if plan["payload"]["schema_version"] != 1:
        raise ValueError("Unsupported plan schema")
    config = plan["payload"]["config"]
    if "fixed_direction_receiver_files" in config:
        receiver_files = config["fixed_direction_receiver_files"]
        if (
            config["purpose"] != "confirmatory"
            or type(config.get("include_original_receiver")) is not bool
            or not isinstance(receiver_files, dict)
            or set(receiver_files) != {"adapter_config.json", "adapter_model.safetensors"}
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                   for v in receiver_files.values())
        ):
            raise ValueError("Fixed-direction plans require a pinned receiver adapter")
    token_limit = config.get("max_new_tokens", 128)
    if type(token_limit) is not int or token_limit < 1:
        raise ValueError("max_new_tokens must be a positive integer")


def execute_plan(plan, output, adapter_path, resume=False, check_only=False):
    """Run or resume under an OS-released exclusive directory lock (Linux/Unix)."""
    import fcntl

    validate_plan(plan)
    output = Path(output)
    if check_only and not resume:
        raise ValueError("--check-only requires --resume")
    if resume:
        if not output.is_dir():
            raise ValueError("Resume output directory does not exist")
    else:
        if output.exists():
            raise ValueError("Output exists; use --resume for this frozen run")
        output.mkdir(parents=True)
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process holds this run's lock") from exc
        try:
            return _execute_plan(plan, output, adapter_path, resume, check_only)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _execute_plan(plan, output, adapter_path, resume, check_only):
    import gc

    import torch
    from peft import PeftModel

    from ddmi.editing.measured import MeasuredIntervention, TOKEN_LOCALIZATION_MODES
    from ddmi.models.positions import POSITION_CONTROL_MODES
    from ddmi.experiments import write_atomic
    from ddmi.models.generation import TextGenerationConfig, generate_text
    from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
    from ddmi.utils.io import set_seed

    validate_plan(plan)
    output = Path(output)
    payload, records = plan["payload"], []
    config = payload["config"]
    adapter = adapter_manifest(adapter_path)
    if config["purpose"] == "confirmatory":
        extraction_files = payload["direction_metadata"]["safety_lora"]["adapter_manifest"]["files"]
        receiver_files = config.get("fixed_direction_receiver_files", extraction_files)
        if receiver_files != adapter["files"]:
            if "fixed_direction_receiver_files" in config:
                raise ValueError("Inference adapter differs from the frozen receiver adapter")
            raise ValueError("Inference adapter differs from direction-extraction adapter")
    set_seed(config["seed"])
    current_runtime = runtime_metadata()
    journal_path = output / "responses.jsonl"
    if resume:
        from ddmi.recovery import commit_recovery, recover_run

        result, details = recover_run(plan, output, adapter, current_runtime)
        print("Validated resume state: " + json.dumps(details, sort_keys=True), flush=True)
        if check_only:
            print("Check only: no model generation or progress files changed.", flush=True)
            return result
        backup = commit_recovery(output, result, details)
        print(f"Original progress backed up to {backup}", flush=True)
        records = result["conditions"]
        if result["complete"]:
            print("All planned responses are already saved; no generation needed.", flush=True)
            return result
    else:
        result = {
            "schema_version": 2,
            "plan_sha256": plan["plan_sha256"],
            "annotation_status": "unreviewed",
            "runtime": current_runtime,
            "adapter": adapter,
            "complete": False,
            "conditions": records,
            "response_journal": "responses.jsonl",
        }
        write_atomic(output / "plan.json", plan)
        write_atomic(output / "run.json", result)
        journal_path.touch(exist_ok=False)
    by_condition = {condition["id"]: condition for condition in records}
    prompt_count = len(payload["prompts"])
    for receiver in ("original_instruct", "safety_lora"):
        pending = [
            c
            for c in payload["conditions"]
            if c["receiver"] == receiver
            and len(by_condition.get(c["id"], {}).get("rows", [])) < prompt_count
        ]
        if not pending:
            continue
        model, tokenizer = load_model_and_tokenizer(
            ModelLoadConfig(
                model_id=payload["model_id"],
                revision=config["model_revision"],
                load_in_4bit=config["precision"] == "nf4",
                torch_dtype="bfloat16" if config["precision"] == "bf16" else "float16",
                attn_implementation="eager",
            )
        )
        try:
            if receiver == "safety_lora":
                wrapped = PeftModel.from_pretrained(model, str(adapter_path))
                model = wrapped.base_model.model
                del wrapped
            model.eval()
            for condition in payload["conditions"]:
                if condition["receiver"] != receiver:
                    continue
                record = by_condition.get(condition["id"])
                if record is not None and len(record["rows"]) == prompt_count:
                    continue
                saved_indices = {row["source_index"] for row in record["rows"]} if record else set()
                print(
                    f"Saved {len(saved_indices)}/{prompt_count} prompts; "
                    f"generating {prompt_count - len(saved_indices)}",
                    flush=True,
                )
                print(
                    f"{condition['id']}: {receiver} {condition['mode']} {condition['amount']}",
                    flush=True,
                )
                layer = condition["layer"]
                sources = (
                    ("original_instruct", "safety_lora")
                    if condition["source"] == "both"
                    else (condition["source"],)
                )
                hooks = []
                try:
                    for source in sources:
                        if source == "random":
                            vector = torch.randn(
                                len(payload["vectors"][f"safety_lora:L{layer}"]),
                                generator=torch.Generator().manual_seed(condition["random_seed"]),
                            ).tolist()
                        else:
                            vector = payload["vectors"][f"{source}:L{layer}"]
                        hook = MeasuredIntervention(
                            f"model.layers.{layer}", vector, condition["mode"], condition["amount"]
                        ).attach(model)
                        hooks.append((source, hook))
                    if record is None:
                        record = {**condition, "rows": []}
                        records.append(record)
                        by_condition[condition["id"]] = record
                        order = {c["id"]: i for i, c in enumerate(payload["conditions"])}
                        records.sort(key=lambda c: order[c["id"]])
                    for index, row in enumerate(payload["prompts"]):
                        if index in saved_indices:
                            continue
                        for _, hook in hooks:
                            hook.reset()
                        generated = generate_text(
                            model,
                            tokenizer,
                            row["prompt"],
                            TextGenerationConfig(
                                max_new_tokens=config.get("max_new_tokens", 128),
                                enable_thinking=config.get("enable_thinking"),
                                record_token_ids=(condition["mode"] in TOKEN_LOCALIZATION_MODES
                                                  or config.get("record_position_controls", False)),
                                position_control_mode=(condition["mode"]
                                    if config.get("record_position_controls", False) else None),
                            ),
                            return_metadata=True,
                            prefill_callback=(hooks[0][1].set_prefill_position
                                if condition["mode"] in POSITION_CONTROL_MODES else None),
                        )
                        telemetry = {source: hook.calls for source, hook in hooks}
                        if any(not calls for calls in telemetry.values()):
                            raise ValueError("Intervention hook did not execute")
                        record["rows"].append(
                            {
                                "prompt_id": row["id"],
                                "domain": row["domain"],
                                "source_index": index,
                                "prompt": row["prompt"],
                                **generated,
                                "telemetry": telemetry,
                                "annotation": None,
                            }
                        )
                        # Append one completed response instead of serializing the growing
                        # full telemetry collection after every prompt.
                        with journal_path.open("a", encoding="utf-8") as journal:
                            json.dump(
                                {
                                    "condition_id": condition["id"],
                                    "row": record["rows"][-1],
                                    "plan_sha256": plan["plan_sha256"],
                                },
                                journal,
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            journal.write("\n")
                    record["rows"].sort(key=lambda row: row["source_index"])
                finally:
                    for _, hook in reversed(hooks):
                        hook.close()
                write_atomic(output / "run.json", result)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result["complete"] = True
    write_atomic(output / "run.json", result)
    return result

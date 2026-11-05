"""Validate and recover measured responses without trusting an incomplete checkpoint."""

from __future__ import annotations

import copy
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ddmi.paper.evidence import read_json
from ddmi.publication import validate_plan


def recover_run(plan, output, adapter, runtime):
    """Read-only recovery. No file is changed until all validation succeeds."""
    import json

    output = Path(output)
    validate_plan(plan)
    saved_plan = read_json(output / "plan.json")
    validate_plan(saved_plan)
    if saved_plan["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Resume plan differs from the saved frozen plan")
    saved = read_json(output / "run.json")
    if saved.get("schema_version") != 2 or saved.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Checkpoint schema or plan hash does not match")
    if saved.get("annotation_status") != "unreviewed":
        raise ValueError("Cannot resume a reviewed dataset")
    if saved.get("adapter") != adapter:
        raise ValueError("Adapter config or weight hashes changed; refusing to mix checkpoints")
    if saved.get("runtime") != runtime:
        raise ValueError("Runtime/package/device metadata changed; refusing to mix environments")
    specs = {c["id"]: c for c in plan["payload"]["conditions"]}
    if len(specs) != len(plan["payload"]["conditions"]):
        raise ValueError("Duplicate condition IDs in frozen plan")
    prompts = plan["payload"]["prompts"]
    recovered = {key: {} for key in specs}

    def add(condition_id, row):
        if not isinstance(row, dict):
            raise ValueError("Saved journal row is not an object")
        if condition_id not in specs:
            raise ValueError("Saved response references an unknown condition")
        index = row.get("source_index")
        if type(index) is not int or index not in range(len(prompts)):
            raise ValueError("Saved response has an invalid prompt index")
        prompt = prompts[index]
        if any(
            row.get(key) != prompt[source]
            for key, source in (("prompt_id", "id"), ("prompt", "prompt"), ("domain", "domain"))
        ):
            raise ValueError("Saved prompt identity or text differs from the frozen plan")
        if not isinstance(row.get("response"), str) or row.get("annotation") is not None:
            raise ValueError("Saved response is missing text or has acquired annotations")
        if type(row.get("generated_tokens")) is not int or row["generated_tokens"] <= 0:
            raise ValueError("Saved response has invalid generated-token metadata")
        telemetry = row.get("telemetry", {})
        spec = specs[condition_id]
        sources = (
            {"original_instruct", "safety_lora"} if spec["source"] == "both" else {spec["source"]}
        )
        if set(telemetry) != sources:
            raise ValueError("Saved response has missing or unexpected telemetry sources")
        for calls in telemetry.values():
            if not calls or len(calls) != row["generated_tokens"]:
                raise ValueError("Saved telemetry is incomplete")
            for n, call in enumerate(calls):
                if call.get("phase") != ("prefill" if n == 0 else "decode"):
                    raise ValueError("Saved telemetry has invalid phase ordering")
                if type(call.get("tokens")) is not int or call["tokens"] < 1:
                    raise ValueError("Saved telemetry has an invalid token count")
                if n and call["tokens"] != 1:
                    raise ValueError("Saved decoding telemetry does not use single-token caching")
                for metric in (
                    "activation_norm",
                    "absolute_projection",
                    "displacement_l2",
                    "relative_displacement_l2",
                ):
                    values = call.get(metric, {})
                    if set(values) != {"mean", "rms", "min", "max", "last"} or any(
                        not isinstance(v, (float, int)) or not math.isfinite(v)
                        for v in values.values()
                    ):
                        raise ValueError(
                            "Saved telemetry contains invalid or nonfinite measurements"
                        )
        existing = recovered[condition_id].get(index)
        if existing is not None and existing != row:
            raise ValueError("Conflicting saved responses for the same condition and prompt")
        recovered[condition_id][index] = row

    seen_conditions = set()
    checkpoint_responses = 0
    for condition in saved.get("conditions", []):
        key = condition.get("id")
        if key not in specs or key in seen_conditions:
            raise ValueError("Unknown or duplicate checkpoint condition")
        seen_conditions.add(key)
        if any(condition.get(field) != value for field, value in specs[key].items()):
            raise ValueError("Checkpoint condition settings differ from the frozen plan")
        indices = set()
        for row in condition["rows"]:
            if row.get("source_index") in indices:
                raise ValueError("Duplicate prompt in checkpoint condition")
            indices.add(row.get("source_index"))
            add(key, row)
            checkpoint_responses += 1

    journal_path = output / "responses.jsonl"
    if not journal_path.is_file():
        raise ValueError("Response journal is missing; preserve this run for manual inspection")
    journal_records, trim_offset, needs_newline = 0, None, False
    with journal_path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if line.endswith(b"\n"):
                    raise ValueError(
                        "Malformed complete journal line; refusing automatic repair"
                    ) from exc
                # readline only returns a non-newline-terminated line at EOF.
                trim_offset = offset
                break
            if not isinstance(event, dict) or "condition_id" not in event or "row" not in event:
                raise ValueError("Invalid response journal event")
            if event.get("plan_sha256", plan["plan_sha256"]) != plan["plan_sha256"]:
                raise ValueError("Journal event belongs to a different frozen plan")
            add(event["condition_id"], event["row"])
            journal_records += 1
            needs_newline = not line.endswith(b"\n")

    result = {key: copy.deepcopy(value) for key, value in saved.items() if key != "conditions"}
    result["conditions"] = [
        {**spec, "rows": [recovered[spec["id"]][i] for i in sorted(recovered[spec["id"]])]}
        for spec in plan["payload"]["conditions"]
        if recovered[spec["id"]]
    ]
    total = sum(len(rows) for rows in recovered.values())
    expected = len(specs) * len(prompts)
    if saved.get("complete") and total != expected:
        raise ValueError("Run marked complete is missing responses")
    result["complete"] = total == expected
    details = {
        "checkpoint_responses": checkpoint_responses,
        "valid_journal_events": journal_records,
        "recovered_responses": total,
        "remaining_responses": expected - total,
        "complete_conditions": sum(len(rows) == len(prompts) for rows in recovered.values()),
        "trim_offset": trim_offset,
        "needs_newline": needs_newline if trim_offset is None else False,
    }
    return result, details


def commit_recovery(output, result, details):
    """Back up original files before checkpoint promotion or final-tail repair."""
    from ddmi.experiments import write_atomic

    output = Path(output)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    backup = output / "resume_backups" / stamp
    backup.mkdir(parents=True)
    for name in ("plan.json", "run.json", "responses.jsonl"):
        shutil.copy2(output / name, backup / name)
    journal = output / "responses.jsonl"
    if details["trim_offset"] is not None:
        with journal.open("r+b") as handle:
            handle.truncate(details["trim_offset"])
        # A preceding valid line was complete and newline-terminated.
    elif details["needs_newline"]:
        with journal.open("ab") as handle:
            handle.write(b"\n")
    result.setdefault("resume_history", []).append(
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "recovery_protocol": "journal_resume_v1",
            "backup": str(backup.relative_to(output)),
            **details,
        }
    )
    write_atomic(output / "run.json", result)
    return backup

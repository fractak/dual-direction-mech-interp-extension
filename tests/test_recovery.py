"""Recovery preserves completed work and rejects mixed or corrupt provenance."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest

from ddmi.publication import create_plan, digest, execute_plan
from ddmi.recovery import commit_recovery, recover_run

ROOT = Path(__file__).resolve().parents[1]


def tiny_plan():
    plan = create_plan(
        ROOT,
        {
            "purpose": "pilot",
            "model_revision": "a" * 40,
            "precision": "nf4",
            "seed": 42,
            "layers": [20],
            "projection_strengths": [1],
            "relative_budgets": [0.03],
            "random_seeds": [2026],
        },
    )
    plan["payload"]["conditions"] = plan["payload"]["conditions"][:1]
    plan["payload"]["prompts"] = plan["payload"]["prompts"][:3]
    plan["payload"]["planned_responses"] = 3
    plan["plan_sha256"] = digest(plan["payload"])
    return plan


def event(plan, index):
    prompt = plan["payload"]["prompts"][index]
    metric = {key: 0.0 for key in ("mean", "rms", "min", "max", "last")}
    return {
        "condition_id": "c0000",
        "row": {
            "prompt_id": prompt["id"],
            "prompt": prompt["prompt"],
            "domain": prompt["domain"],
            "source_index": index,
            "response": f"saved response {index}",
            "annotation": None,
            "generated_tokens": 1,
            "telemetry": {
                source: [
                    {
                        "phase": "prefill",
                        "tokens": 2,
                        **{
                            name: dict(metric)
                            for name in (
                                "activation_norm",
                                "absolute_projection",
                                "displacement_l2",
                                "relative_displacement_l2",
                            )
                        },
                    }
                ]
                for source in ("original_instruct", "safety_lora")
            },
        },
    }


@pytest.fixture
def crashed(tmp_path):
    plan = tiny_plan()
    adapter, runtime = {"files": {"weights": "hash"}}, {"packages": {"torch": "version"}}
    output = tmp_path / "run"
    output.mkdir()
    result = {
        "schema_version": 2,
        "plan_sha256": plan["plan_sha256"],
        "annotation_status": "unreviewed",
        "adapter": adapter,
        "runtime": runtime,
        "complete": False,
        "conditions": [{**plan["payload"]["conditions"][0], "rows": [event(plan, 0)["row"]]}],
    }
    (output / "plan.json").write_text(json.dumps(plan))
    (output / "run.json").write_text(json.dumps(result))
    (output / "responses.jsonl").write_text(
        "\n".join(json.dumps(event(plan, i)) for i in [0, 1]) + "\n"
    )
    return plan, output, adapter, runtime


def test_recovery_unions_checkpoint_and_journal_without_repeating_responses(crashed):
    plan, output, adapter, runtime = crashed
    original = (output / "run.json").read_bytes()
    result, details = recover_run(plan, output, adapter, runtime)
    assert details["recovered_responses"] == 2
    assert details["remaining_responses"] == 1
    assert [r["source_index"] for r in result["conditions"][0]["rows"]] == [0, 1]
    assert (output / "run.json").read_bytes() == original
    backup = commit_recovery(output, result, details)
    assert (backup / "run.json").read_bytes() == original
    assert len(json.loads((output / "run.json").read_text())["conditions"][0]["rows"]) == 2


@pytest.mark.parametrize(
    "mismatch", ["adapter", "runtime", "plan", "prompt", "telemetry", "conflict"]
)
def test_recovery_rejects_changes_without_modifying_saved_progress(crashed, mismatch):
    plan, output, adapter, runtime = crashed
    if mismatch == "adapter":
        adapter = {"files": {"weights": "different"}}
    elif mismatch == "runtime":
        runtime = {"packages": {"torch": "changed"}}
    elif mismatch == "plan":
        plan = copy.deepcopy(plan)
        plan["payload"]["config"]["seed"] = 99
        plan["plan_sha256"] = digest(plan["payload"])
    else:
        item = event(plan, 1 if mismatch != "conflict" else 0)
        if mismatch == "prompt":
            item["row"]["prompt"] += " changed"
        elif mismatch == "telemetry":
            item["row"]["telemetry"] = {}
        else:
            item["row"]["response"] = "conflicting response"
        with (output / "responses.jsonl").open("a") as handle:
            handle.write(json.dumps(item) + "\n")
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        recover_run(plan, output, adapter, runtime)
    assert before == {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}


def test_partial_final_journal_write_is_backed_up_before_repair(crashed):
    plan, output, adapter, runtime = crashed
    journal = output / "responses.jsonl"
    with journal.open("ab") as handle:
        handle.write(b'{"condition_id":"c0000","row":')
    original = journal.read_bytes()
    result, details = recover_run(plan, output, adapter, runtime)
    assert details["recovered_responses"] == 2
    assert details["trim_offset"] is not None
    assert journal.read_bytes() == original
    backup = commit_recovery(output, result, details)
    assert (backup / "responses.jsonl").read_bytes() == original
    assert len([json.loads(line) for line in journal.read_text().splitlines()]) == 2


def test_complete_bad_journal_line_is_not_silently_removed(crashed):
    plan, output, adapter, runtime = crashed
    with (output / "responses.jsonl").open("a") as handle:
        handle.write("not valid JSON\n")
    with pytest.raises(ValueError, match="Malformed complete"):
        recover_run(plan, output, adapter, runtime)


def test_valid_final_record_without_newline_is_preserved(crashed):
    plan, output, adapter, runtime = crashed
    journal = output / "responses.jsonl"
    journal.write_bytes(journal.read_bytes().rstrip(b"\n"))
    result, details = recover_run(plan, output, adapter, runtime)
    assert details["needs_newline"]
    commit_recovery(output, result, details)
    with journal.open("a") as handle:
        handle.write(json.dumps(event(plan, 2)) + "\n")
    final, details = recover_run(plan, output, adapter, runtime)
    assert final["complete"]
    assert details["recovered_responses"] == 3


def test_runner_resume_generates_only_missing_prompts_and_checks_only_is_read_only(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    from ddmi import publication
    from ddmi.models import generation, loader

    plan = tiny_plan()
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(21)])
    model_loads = []

    def load(config):
        model_loads.append(config)
        return model, object()

    monkeypatch.setattr(loader, "load_model_and_tokenizer", load)
    monkeypatch.setattr(publication, "adapter_manifest", lambda path: {"files": {}})
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    calls = []

    def generate(model, tokenizer, prompt, *args, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("interrupted second response")
        model.model.layers[20](torch.ones(1, 2, 4096))
        return {
            "response": prompt,
            "generated_tokens": 1,
            "input_tokens": 2,
            "untruncated_input_tokens": 2,
            "input_truncated": False,
            "hit_token_limit": False,
        }

    monkeypatch.setattr(generation, "generate_text", generate)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        execute_plan(plan, output, "fake")
    before = {
        name: (output / name).read_bytes() for name in ["plan.json", "run.json", "responses.jsonl"]
    }
    checked = execute_plan(plan, output, "fake", resume=True, check_only=True)
    assert not checked["complete"]
    assert len(model_loads) == 1
    assert before == {name: (output / name).read_bytes() for name in before}
    final = execute_plan(plan, output, "fake", resume=True)
    assert final["complete"]
    first, second, third = [r["prompt"] for r in plan["payload"]["prompts"]]
    assert calls == [first, second, second, third]
    journal = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    assert [e["row"]["source_index"] for e in journal] == [0, 1, 2]
    assert not model.model.layers[20]._forward_hooks
    loads = len(model_loads)
    execute_plan(plan, output, "fake", resume=True)
    assert len(model_loads) == loads


def test_run_lock_blocks_a_second_writer(tmp_path):
    import fcntl

    output = tmp_path / "run"
    output.mkdir()
    with (output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another process"):
            execute_plan(tiny_plan(), output, "fake", resume=True)


def test_recovery_matches_687_response_interruption_shape(tmp_path):
    plan = create_plan(
        ROOT,
        {
            "purpose": "pilot",
            "model_revision": "a" * 40,
            "precision": "nf4",
            "seed": 42,
            "layers": [20],
            "projection_strengths": [1, 1.5],
            "relative_budgets": [0.03, 0.1, 0.3],
            "random_seeds": list(range(2026, 2032)),
        },
    )
    adapter, runtime = {"files": {}}, {"packages": {}}
    output = tmp_path / "large_run"
    output.mkdir()
    saved = {
        "schema_version": 2,
        "plan_sha256": plan["plan_sha256"],
        "annotation_status": "unreviewed",
        "adapter": adapter,
        "runtime": runtime,
        "complete": False,
        "conditions": [],
    }
    with (output / "responses.jsonl").open("w") as journal:
        for n, spec in enumerate(plan["payload"]["conditions"][:12]):
            rows = []
            for index in range(60 if n < 11 else 27):
                item = event(plan, index)
                item["condition_id"] = spec["id"]
                sources = (
                    ["original_instruct", "safety_lora"]
                    if spec["source"] == "both"
                    else [spec["source"]]
                )
                sample = next(iter(item["row"]["telemetry"].values()))
                item["row"]["telemetry"] = {source: sample for source in sources}
                rows.append(item["row"])
                journal.write(json.dumps(item) + "\n")
            if n < 11:
                saved["conditions"].append({**spec, "rows": rows})
    (output / "plan.json").write_text(json.dumps(plan))
    (output / "run.json").write_text(json.dumps(saved))
    result, details = recover_run(plan, output, adapter, runtime)
    assert details["recovered_responses"] == 687
    assert details["remaining_responses"] == 2793
    assert details["complete_conditions"] == 11
    partial = result["conditions"][-1]
    assert partial["id"] == "c0011"
    assert len(partial["rows"]) == 27
    assert partial["rows"][-1]["source_index"] == 26

"""Scientific controls, prospective plan integrity, and review independence."""

from __future__ import annotations

import copy
import csv
import json
import sys
import types
from pathlib import Path

import pytest

from ddmi.prospective_analysis import analyze_reviewed, cluster_summary
from ddmi.publication import create_plan, digest, execute_plan, validate_plan
from ddmi.review import export_review, merge_review

ROOT = Path(__file__).resolve().parents[1]


def config():
    return {
        "purpose": "pilot",
        "model_revision": "a" * 40,
        "precision": "nf4",
        "seed": 42,
        "layers": [20, 24],
        "projection_strengths": [1, 1.5],
        "relative_budgets": [0.1],
        "random_seeds": [2026, 2027],
    }


@pytest.fixture
def plan():
    return create_plan(ROOT, config())


def test_plan_full_same_layer_matrix_and_equal_budget_randoms(plan):
    validate_plan(plan)
    p = plan["payload"]
    assert p["planned_responses"] == 36 * 60
    for layer in (20, 24):
        for source in ("original_instruct", "safety_lora"):
            matched = [
                c
                for c in p["conditions"]
                if c["layer"] == layer and c["source"] == source and c["mode"] == "relative_l2"
            ]
            assert {c["receiver"] for c in matched} == {"original_instruct", "safety_lora"}
            assert {c["amount"] for c in matched} == {0.1}
    changed = copy.deepcopy(plan)
    changed["payload"]["prompts"][0]["prompt"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        validate_plan(changed)


@pytest.mark.parametrize("revision", ["main", "", "not-a-revision"])
def test_mutable_model_revision_refused(revision):
    c = config()
    c["model_revision"] = revision
    with pytest.raises(ValueError, match="immutable"):
        create_plan(ROOT, c)


def test_confirmatory_cannot_reuse_archival_prompts(tmp_path):
    c = config()
    c["purpose"] = "confirmatory"
    with pytest.raises(ValueError, match="require new"):
        create_plan(ROOT, c)
    with pytest.raises(ValueError, match="no exact overlap"):
        create_plan(
            ROOT, c, ROOT / "data/capability_eval_expanded.json", ROOT / "evidence/directions"
        )


def test_new_directions_require_extraction_provenance(tmp_path):
    c = config()
    c["purpose"] = "confirmatory"
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps([{"id": "new", "prompt": "a newly authored prompt", "domain": "x"}])
    )
    with pytest.raises(ValueError, match="revision"):
        create_plan(ROOT, c, prompts, ROOT / "evidence/directions")


def test_relative_displacement_is_equal_despite_unequal_projections():
    torch = pytest.importorskip("torch")
    from ddmi.editing.measured import MeasuredIntervention

    model = torch.nn.Sequential(torch.nn.Identity())
    inputs = [torch.tensor([[[3.0, 4.0]]]), torch.tensor([[[0.05, 5.0]]])]
    for value in inputs:
        hook = MeasuredIntervention("0", [1, 0], "relative_l2", 0.1).attach(model)
        result = model(value)
        hook.close()
        ratio = (result - value).norm() / value.norm()
        assert ratio.item() == pytest.approx(0.1, abs=1e-7)
        assert hook.calls[0]["relative_displacement_l2"]["last"] == pytest.approx(0.1, abs=1e-7)
    assert not model[0]._forward_hooks


def test_measured_projection_sign_flip_and_tuple_preserved():
    torch = pytest.importorskip("torch")
    from ddmi.editing.measured import MeasuredIntervention

    class TupleLayer(torch.nn.Module):
        def forward(self, x):
            return x, "cache"

    model = torch.nn.Sequential(TupleLayer())
    hook = MeasuredIntervention("0", [1, 0], "projection", 1.5).attach(model)
    value = torch.tensor([[[3.0, 4.0], [2.0, 1.0]]])
    result, cache = model(value)
    assert cache == "cache"
    assert torch.allclose(result[..., 0], -0.5 * value[..., 0])
    assert torch.allclose(result[..., 1], value[..., 1])
    model(torch.ones(1, 1, 2))
    assert [c["phase"] for c in hook.calls] == ["prefill", "decode"]
    with pytest.raises(ValueError, match="cache"):
        model(value)
    hook.close()


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_relative_rounding_telemetry_is_measured_from_cast_output(dtype_name):
    torch = pytest.importorskip("torch")
    from ddmi.editing.measured import MeasuredIntervention

    model = torch.nn.Sequential(torch.nn.Identity())
    hook = MeasuredIntervention("0", [1, 0], "relative_l2", 1e-6).attach(model)
    value = torch.tensor([[[3.0, 4.0]]], dtype=getattr(torch, dtype_name))
    try:
        value.float()
    except RuntimeError as exc:
        if "cpuinfo" in str(exc):
            pytest.skip("Sandbox hides CPU information needed by this PyTorch dtype conversion")
        raise
    edited = model(value)
    assert edited.dtype == value.dtype
    assert hook.calls[0]["displacement_l2"]["last"] == float(
        (edited.float() - value.float()).norm()
    )
    hook.close()


def test_zero_projection_policy_is_explicit_and_observe_is_identity():
    torch = pytest.importorskip("torch")
    from ddmi.editing.measured import MeasuredIntervention

    model = torch.nn.Sequential(torch.nn.Identity())
    value = torch.tensor([[[0.0, 5.0]]])
    for mode in ("observe", "relative_l2"):
        hook = MeasuredIntervention("0", [1, 0], mode, 0.1).attach(model)
        edited = model(value)
        assert hook.calls[0]["zero_projection_tokens"] == 1
        if mode == "observe":
            assert edited is value
        else:
            assert edited[0, 0, 0] == -0.5
        hook.close()


def fill_review(path, labels):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row, label in zip(rows, labels):
        row["outcome"] = label
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_blind_export_deduplicates_and_merge_requires_adjudication(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "rows": [
                    {"prompt": "p", "response": "r", "outcome": "other", "checkpoint": "secret"},
                    {
                        "prompt": "p",
                        "response": "r",
                        "outcome": "other",
                        "checkpoint": "other-secret",
                    },
                    {"prompt": "q", "response": "s", "outcome": "other", "checkpoint": "secret"},
                ]
            }
        )
    )
    review = tmp_path / "review"
    assert export_review(source, review) == {"unique_text_pairs": 2, "source_rows": 3}
    assert "secret" not in (review / "reviewer_A.csv").read_text()
    fill_review(review / "reviewer_A.csv", ["acknowledgment", "unsupported"])
    fill_review(review / "reviewer_B.csv", ["acknowledgment", "other"])
    report = merge_review(
        review / "PRIVATE_review_map.json",
        review / "reviewer_A.csv",
        review / "reviewer_B.csv",
        tmp_path / "merged",
    )
    assert report["agreement"] == 0.5
    assert report["unresolved"] == 1
    assert report["cohen_kappa"] == pytest.approx(1 / 3)
    fill_review(tmp_path / "merged/adjudications.csv", ["unsupported"])
    report = merge_review(
        review / "PRIVATE_review_map.json",
        review / "reviewer_A.csv",
        review / "reviewer_B.csv",
        tmp_path / "resolved",
        tmp_path / "merged/adjudications.csv",
    )
    assert report["all_resolved"]
    restored = json.loads((tmp_path / "resolved/reviewed_source.json").read_text())["rows"]
    assert restored[0]["independent_review"] == restored[1]["independent_review"]
    assert all(r["outcome"] == "other" for r in restored)
    # Source mutation must invalidate reviews.
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="Source changed"):
        merge_review(
            review / "PRIVATE_review_map.json",
            review / "reviewer_A.csv",
            review / "reviewer_B.csv",
            tmp_path / "invalid",
        )


def test_cluster_intervals_preserve_domain_dependence():
    delta = [-1.0] * 10 + [1.0] * 10
    domains = ["a"] * 10 + ["b"] * 10
    result = cluster_summary(delta, domains, samples=1000)
    assert result["bootstrap_95_ci"] == [-1, 1]
    assert cluster_summary([1, 2], ["a", "a"]) is None


def tiny_reviewed_run(plan):
    p = copy.deepcopy(plan)
    p["payload"]["prompts"] = [
        {"id": "a", "prompt": "A", "domain": "a"},
        {"id": "b", "prompt": "B", "domain": "b"},
    ]
    p["plan_sha256"] = digest(p["payload"])
    conditions = []
    for condition in p["payload"]["conditions"]:
        label = (
            "unsupported"
            if condition["receiver"] == "safety_lora" and condition["mode"] != "observe"
            else "acknowledgment"
        )
        rows = [
            {
                "prompt_id": r["id"],
                "prompt": r["prompt"],
                "domain": r["domain"],
                "independent_review": {"outcome": label},
                "hit_token_limit": False,
                "input_truncated": False,
                "telemetry": {
                    source: [
                        {
                            "phase": "prefill",
                            "absolute_projection": {"last": 1},
                            "displacement_l2": {"last": 0},
                            "relative_displacement_l2": {"last": 0, "max": 0},
                            "zero_projection_tokens": 0,
                        }
                    ]
                    for source in (
                        ["original_instruct", "safety_lora"]
                        if condition["source"] == "both"
                        else [condition["source"]]
                    )
                },
            }
            for r in p["payload"]["prompts"]
        ]
        conditions.append({**condition, "rows": rows})
    return p, {"complete": True, "plan_sha256": p["plan_sha256"], "conditions": conditions}


def test_prospective_analysis_pairs_receivers_and_rejects_unresolved_labels(plan):
    p, run = tiny_reviewed_run(plan)
    result = analyze_reviewed(p, run)
    effects = result["receiver_differences_in_changes"][0]["effects"]
    assert effects["acknowledgment"]["prompt_paired"]["change"] == -1
    assert effects["unsupported"]["prompt_paired"]["change"] == 1
    run["conditions"][0]["rows"][0]["independent_review"]["outcome"] = None
    with pytest.raises(ValueError, match="resolved"):
        analyze_reviewed(p, run)


def test_measured_runner_closes_hooks_on_generation_failure(plan, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from ddmi import publication
    from ddmi.models import generation, loader

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(25)])
    monkeypatch.setattr(loader, "load_model_and_tokenizer", lambda config: (model, object()))

    def fail(*args, **kwargs):
        model.model.layers[20](torch.ones(1, 2, 4096))
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(generation, "generate_text", fail)
    monkeypatch.setattr(publication, "adapter_manifest", lambda path: {"files": {}})
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        execute_plan(plan, tmp_path / "run", "fake")
    assert not model.model.layers[20]._forward_hooks
    assert not json.loads((tmp_path / "run/run.json").read_text())["complete"]


def test_measured_runner_success_keeps_new_responses_unannotated(plan, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from ddmi import publication
    from ddmi.models import generation, loader

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(25)])
    monkeypatch.setattr(loader, "load_model_and_tokenizer", lambda config: (model, object()))

    def generate(*args, **kwargs):
        model.model.layers[20](torch.ones(1, 2, 4096))
        model.model.layers[20](torch.ones(1, 1, 4096))
        return {
            "response": "unreviewed output",
            "input_tokens": 2,
            "untruncated_input_tokens": 2,
            "input_truncated": False,
            "generated_tokens": 2,
            "hit_token_limit": False,
        }

    monkeypatch.setattr(generation, "generate_text", generate)
    monkeypatch.setattr(publication, "adapter_manifest", lambda path: {"files": {}})
    wrapper = types.SimpleNamespace(
        from_pretrained=lambda model, path: types.SimpleNamespace(
            base_model=types.SimpleNamespace(model=model)
        )
    )
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=wrapper))
    plan = copy.deepcopy(plan)
    plan["payload"]["conditions"] = [
        c
        for c in plan["payload"]["conditions"]
        if c["layer"] == 20 and c["mode"] in ("observe", "projection") and c["amount"] in (0, 1)
    ]
    plan["payload"]["prompts"] = plan["payload"]["prompts"][:1]
    plan["plan_sha256"] = digest(plan["payload"])
    result = execute_plan(plan, tmp_path / "run", "fake")
    assert result["complete"]
    journal = [
        json.loads(line) for line in (tmp_path / "run/responses.jsonl").read_text().splitlines()
    ]
    assert len(journal) == sum(len(c["rows"]) for c in result["conditions"])
    for c, event in zip(result["conditions"], journal):
        assert event["condition_id"] == c["id"]
        assert event["row"] == c["rows"][0]
    for c in result["conditions"]:
        assert c["rows"][0]["annotation"] is None
        for calls in c["rows"][0]["telemetry"].values():
            assert [v["phase"] for v in calls] == ["prefill", "decode"]
    assert not model.model.layers[20]._forward_hooks


def test_measured_runner_journals_completed_response_before_later_failure(
    plan, monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    from ddmi import publication
    from ddmi.models import generation, loader

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(25)])
    monkeypatch.setattr(loader, "load_model_and_tokenizer", lambda config: (model, object()))
    monkeypatch.setattr(publication, "adapter_manifest", lambda path: {"files": {}})
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=object))
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("failure on second response")
        model.model.layers[20](torch.ones(1, 2, 4096))
        return {"response": "first completed response", "generated_tokens": 1}

    monkeypatch.setattr(generation, "generate_text", generate)
    plan = copy.deepcopy(plan)
    plan["payload"]["conditions"] = [plan["payload"]["conditions"][0]]
    plan["payload"]["prompts"] = plan["payload"]["prompts"][:2]
    plan["plan_sha256"] = digest(plan["payload"])
    with pytest.raises(RuntimeError, match="second response"):
        execute_plan(plan, tmp_path / "run", "fake")
    checkpoint = json.loads((tmp_path / "run/run.json").read_text())
    assert not checkpoint["complete"]
    journal = [
        json.loads(line) for line in (tmp_path / "run/responses.jsonl").read_text().splitlines()
    ]
    assert len(journal) == 1
    assert journal[0]["row"]["response"] == "first completed response"
    assert journal[0]["condition_id"] == plan["payload"]["conditions"][0]["id"]
    assert not model.model.layers[20]._forward_hooks

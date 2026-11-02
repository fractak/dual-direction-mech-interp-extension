"""Experiment planning and cleanup of hooks, using synthetic CPU models."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from ddmi.experiments import build_plan

ROOT = Path(__file__).resolve().parents[1]


def test_main_plan_shares_the_same_vector():
    plan = build_plan(
        ROOT,
        ["original_instruct", "safety_lora"],
        ["baseline", "capability"],
        [1.5],
        [],
        "model.layers.20",
        "safety_lora",
    )
    edited = [row for row in plan if row["condition"] == "capability"]
    assert len(plan) == 4
    assert edited[0]["directions"] == edited[1]["directions"]
    assert edited[0]["directions"][0]["checkpoint"] == "safety_lora"


def test_baseline_is_not_repeated_for_every_strength():
    plan = build_plan(
        ROOT,
        ["safety_lora"],
        ["baseline", "capability"],
        [0, 0.42, 0.79, 1.5],
        [],
        "model.layers.20",
        "safety_lora",
    )
    assert len(plan) == 5


def test_overlap_plan_preserves_52_groups():
    plan = build_plan(
        ROOT,
        ["original_instruct", "safety_lora"],
        [],
        [1.5],
        [],
        "unused",
        "safety_lora",
        study="overlap",
    )
    assert len(plan) == 52
    triple = next(
        row
        for row in plan
        if row["checkpoint"] == "safety_lora"
        and row["condition"] == "ablate_safety_capability_epistemic"
    )
    assert [spec["layer"] for spec in triple["directions"]] == [
        "model.layers.22",
        "model.layers.20",
        "model.layers.9",
    ]


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf")])
def test_invalid_strength_rejected(strength):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_plan(
            ROOT,
            ["original_instruct"],
            ["capability"],
            [strength],
            [],
            "model.layers.20",
            "safety_lora",
        )


def fake_model_and_generation(monkeypatch, fail=False):
    torch = pytest.importorskip("torch")
    from ddmi.models import generation, loader

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(21)])
    monkeypatch.setattr(loader, "load_model_and_tokenizer", lambda config: (model, object()))

    def generate(model, tokenizer, prompts, config):
        if fail:
            raise RuntimeError("synthetic generation failure")
        value = model.model.layers[20](torch.ones(1, 2, 4096))[0, -1, 0].item()
        return [str(value)] * len(prompts)

    monkeypatch.setattr(generation, "generate_batch", generate)
    return model


def test_new_runs_do_not_inherit_annotations(monkeypatch, tmp_path):
    from ddmi.experiments import execute

    model = fake_model_and_generation(monkeypatch)
    plan = build_plan(
        ROOT,
        ["original_instruct"],
        ["baseline", "capability"],
        [1.5],
        [],
        "model.layers.20",
        "safety_lora",
    )
    output = tmp_path / "run.json"
    result = execute(plan, ROOT, output, None)
    assert len(result["conditions"]) == 2
    assert all(
        row["annotation"] is None for condition in result["conditions"] for row in condition["rows"]
    )
    assert not model.model.layers[20]._forward_hooks
    assert output.is_file()


def test_generation_failure_removes_hooks(monkeypatch, tmp_path):
    from ddmi.experiments import execute

    model = fake_model_and_generation(monkeypatch, fail=True)
    plan = build_plan(
        ROOT, ["original_instruct"], ["capability"], [1.5], [], "model.layers.20", "safety_lora"
    )
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        execute(plan, ROOT, tmp_path / "run.json", None)
    assert not model.model.layers[20]._forward_hooks


def test_preparer_reads_advbench_instruction_and_preserves_alpaca_input(monkeypatch, tmp_path):
    import runpy

    datasets = types.ModuleType("datasets")

    def load_dataset(name, *args, **kwargs):
        if name == "ivnle/advbench_harmful_behaviors":
            return [{"instruction": "exclude this instruction"}]
        if name == "iNLP-Lab/multilingual-safety":
            return [
                {"prompt": "exclude this instruction", "output": "safe response"},
                {"prompt": "remaining prompt", "output": "safe response"},
            ]
        return [{"instruction": "summarize", "input": "the passage", "output": "a summary"}]

    datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    output = tmp_path / "data.json"
    monkeypatch.setattr(sys, "argv", ["prepare_training_data.py", "--output", str(output)])
    runpy.run_path(str(ROOT / "scripts/prepare_training_data.py"), run_name="__main__")
    rows = json.loads(output.read_text())["rows"]
    assert len(rows) == 2
    assert all(row["prompt"] != "exclude this instruction" for row in rows)
    assert (
        next(row["prompt"] for row in rows if row["kind"] == "benign") == "summarize\n\nthe passage"
    )

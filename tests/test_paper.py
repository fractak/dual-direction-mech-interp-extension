"""Scientific regression, pairing, provenance, and terminal-only reproduction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ddmi.paper.analysis import analyze
from ddmi.paper.evidence import load_rows, validate_rows, verify_checksums
from ddmi.paper.stats import bootstrap_indices, exact_p, holm, paired

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return analyze(ROOT)


def test_main_counts_and_paired_intervals(result):
    conditions = result["capability"]["conditions"]
    outcomes = ("acknowledgment", "instructions", "unsupported", "other")
    expected = {
        ("original_instruct", "baseline"): [58, 2, 0, 0],
        ("safety_lora", "baseline"): [58, 2, 0, 0],
        ("original_instruct", "capability_from_safety_lora"): [49, 10, 1, 0],
        ("safety_lora", "capability_from_safety_lora"): [6, 8, 40, 6],
        ("original_instruct", "capability_from_original_instruct"): [49, 11, 0, 0],
        ("safety_lora", "capability_from_original_instruct"): [58, 2, 0, 0],
        ("safety_lora", "safety"): [43, 5, 11, 1],
        ("safety_lora", "epistemic"): [52, 4, 3, 1],
        ("safety_lora", "safety_orthogonal"): [59, 1, 0, 0],
        ("safety_lora", "capability_low"): [49, 3, 8, 0],
        ("safety_lora", "capability_medium"): [25, 7, 24, 4],
    }
    for (checkpoint, condition), counts in expected.items():
        assert [conditions[checkpoint][condition]["counts"][key] for key in outcomes] == counts
    difference = result["capability"]["difference_in_changes"]
    assert difference["acknowledgment"]["change"] == pytest.approx(-43 / 60)
    assert difference["acknowledgment"]["bootstrap_95_ci"] == pytest.approx([-50 / 60, -36 / 60])
    assert difference["unsupported"]["change"] == pytest.approx(39 / 60)
    assert difference["unsupported"]["bootstrap_95_ci"] == pytest.approx([32 / 60, 46 / 60])
    assert len(result["capability"]["exploratory_tests"]) == 16


def test_random_controls_are_six_paired_runs(result):
    controls = result["capability"]["random_controls"]
    assert len(controls) == 6
    assert sum(row["changed_text"] for row in controls.values()) == 65
    assert all(row["changed_outcome"] == 0 for row in controls.values())
    assert all(row["counts"]["acknowledgment"] == 58 for row in controls.values())


def test_overlap_joint_outcomes_and_transitions(result):
    overlap = result["overlap"]
    assert overlap["joint_outcomes_original"] == {
        "baseline": [4, 16, 0, 0],
        "ablate_safety": [3, 5, 10, 2],
    }
    assert (overlap["new_epistemic_mentions"], overlap["new_mentions_with_refusal"]) == (9, 1)
    triple = overlap["exploratory_tests"]["adapted_triple_add_epistemic"]
    assert (triple["lost"], triple["gained"]) == (1, 1)
    assert triple["bootstrap_95_ci"] == pytest.approx([-0.15, 0.15])


def test_geometry_uses_the_papers_model_named_original_collections(result):
    assert result["geometry"]["mean_between_checkpoints"] == pytest.approx(
        {
            "safety": 0.7883943623886085,
            "epistemic": 0.8190315353984496,
            "capability": 0.6954370834377954,
        }
    )
    assert result["geometry"]["adapted_layer20"]["capability_safety"] == pytest.approx(0.5269509760)


def test_frozen_evidence_and_group_sizes():
    assert verify_checksums(ROOT) == 15
    assert len(load_rows(ROOT)) == 1020
    assert len(load_rows(ROOT, overlap=True)) == 1040


def test_duplicate_prompt_indices_are_rejected():
    rows = load_rows(ROOT)
    rows[1]["source_index"] = rows[0]["source_index"]
    with pytest.raises(ValueError, match="duplicate prompt indices"):
        validate_rows(rows)


def test_mispaired_prompt_strings_are_rejected():
    rows = load_rows(ROOT)
    rows[0]["prompt"] += " mismatched context"
    with pytest.raises(ValueError, match="pairing mismatch"):
        validate_rows(rows)


def test_non_boolean_overlap_annotations_are_rejected():
    rows = load_rows(ROOT, overlap=True)
    rows[0]["refusal"] = "false"
    with pytest.raises(ValueError, match="booleans"):
        validate_rows(rows, overlap=True)


def test_unknown_outcomes_are_rejected():
    rows = load_rows(ROOT)
    rows[0]["outcome"] = "fabricated_answer"
    with pytest.raises(ValueError, match="Unknown final outcome"):
        validate_rows(rows)


def test_checksum_change_is_rejected(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.json").write_text(json.dumps({"files": {"sample.json": "bad-hash"}}))
    (tmp_path / "sample.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_checksums(tmp_path)


def test_exact_test_and_holm():
    assert exact_p(0, 0) == 1
    assert exact_p(8, 0) == pytest.approx(0.0078125)
    assert holm({"one": 0.01, "two": 0.03, "three": 0.9}) == pytest.approx(
        {"one": 0.03, "two": 0.06, "three": 0.9}
    )
    result = paired([True, False], [False, True], bootstrap_indices(2))
    assert result["lost"] == result["gained"] == 1
    assert result["change"] == 0


def test_all_figures_generate_from_another_directory_without_display(tmp_path):
    pytest.importorskip("matplotlib")
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_paper_figures.py"),
            "--output-dir",
            str(tmp_path),
            "--check-paper",
            "--dpi",
            "100",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    for path in tmp_path.glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    for path in tmp_path.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "data/results.json").is_file()


def test_missing_inputs_report_a_path_without_traceback(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_paper.py"), "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "evidence/manifest.json" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.py")), ids=lambda p: p.name)
def test_cli_help_does_not_load_models(script):
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout

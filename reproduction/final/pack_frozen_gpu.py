"""Pack frozen GPU plans, adapter files, training inputs, and local runner source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from server_inputs import file_sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = HERE / "reference"
ARTIFACTS = ROOT / "artifacts"
ADAPTER_NAMES = ("mixed42", "mixed43", "mixed44", "benign42", "benign43", "benign44",
                 "safety42", "safety43", "safety44")


def load(path):
    return json.loads(path.read_text())


def plan_ids():
    results = load(REF / "receiver_robustness_results.json")
    ids = dict(results["plan_sha256s"])
    for name in ("family_replication", "position_control", "interleaved_replication"):
        result = load(REF / f"{name}_results.json")
        component = result.get("component_plans", result.get("plan_sha256"))
        ids.update({f"{name}/{key}": value for key, value in component.items() if "smoke" not in key})
    ids["qwen_interleaved"] = load(REF / "qwen_interleaved_results.json")["plan_sha256"]
    return ids


def plans():
    ids = plan_ids()
    wanted = set(ids.values())
    found = {}
    candidates = set((ARTIFACTS / "plans").rglob("*.json"))
    candidates.update((ARTIFACTS / "runs").rglob("plan.json"))
    candidates.update((ARTIFACTS / "receiver_robustness/plans").glob("*.json"))
    for path in sorted(candidates):
        text = path.read_text()
        if not text.lstrip().startswith("{"):
            continue
        plan = json.loads(text)
        if not isinstance(plan, dict):
            continue
        digest = plan.get("plan_sha256")
        if not isinstance(digest, str) or digest not in wanted or digest in found:
            continue
        payload = json.dumps(plan["payload"], sort_keys=True, ensure_ascii=False, allow_nan=False)
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise ValueError(f"Plan content hash mismatch: {path}")
        found[digest] = path
    for label, digest in sorted(ids.items()):
        print(f"{label}: {found.get(digest, 'MISSING')} ({digest})")
    if found.keys() != wanted:
        raise ValueError("Restore the missing plan JSONs under artifacts/plans before packaging")
    return {label: found[digest] for label, digest in ids.items()}


def sources():
    prior = load(ROOT / "reproduction/inputs.json")["gpu"]
    files = {name: ROOT / name for name in prior}
    for name, size in prior.items():
        if not files[name].is_file() or files[name].stat().st_size != size:
            raise ValueError(f"Missing or size-mismatched historical GPU input: {name}")
    late = plans()
    for path in late.values():
        files[str(path.relative_to(ROOT))] = path
    localization = "artifacts/plans/token_localization/full.json"
    files[localization] = ROOT / localization
    prefix = "artifacts/receiver_robustness"
    for arm in ("benign", "safety", "mixed"):
        name = f"{prefix}/training/{arm}.json"
        files[name] = ROOT / name
    for label in ADAPTER_NAMES:
        for filename in ("adapter_config.json", "adapter_model.safetensors", "training_metadata.json"):
            name = f"{prefix}/adapters/{label}/{filename}"
            files[name] = ROOT / name
    for name in ("data/safety_finetuning/train_sft.json", "scripts/run_publication_study.py",
                 "scripts/train_safety_lora.py", "scripts/review_responses.py"):
        files[name] = ROOT / name
    for path in (ROOT / "src/ddmi").rglob("*.py"):
        files[str(path.relative_to(ROOT))] = path
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f"Missing GPU input: {name}")
    for label in ADAPTER_NAMES:
        path = late[label]
        expected = load(path)["payload"]["config"]["fixed_direction_receiver_files"]
        for filename, digest in expected.items():
            name = f"{prefix}/adapters/{label}/{filename}"
            if file_sha(files[name]) != digest:
                raise ValueError(f"Adapter fails its plan hash: {name}")
    index = {label: str(path.relative_to(ROOT)) for label, path in late.items()}
    index["token_localization"] = localization
    return files, index


def bundle_files(arguments):
    files = {}
    for directory in arguments:
        directory = directory.resolve()
        if not directory.is_dir() or not directory.name.isidentifier():
            raise ValueError(f"Expected a named experiment runner directory: {directory}")
        for p in sorted(directory.iterdir()):
            if (p.is_file() and not p.name.startswith(".") and "private" not in p.name.lower()
                    and p.suffix in {".py", ".sh", ".patch", ".md", ".txt"}):
                key = f"runner_bundles/{directory.name}/{p.name}"
                if key in files:
                    raise ValueError(f"Two runner bundles share the same filename: {key}")
                files[key] = p
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runner", action="append", type=Path, default=[])
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = out.with_name(out.name + ".json")
    if out.exists() or manifest.exists():
        parser.error("Choose a new archive path")
    files, index = sources()
    files.update(bundle_files(args.runner))
    checksums = {name: file_sha(path) for name, path in sorted(files.items())}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with out.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1, dereference=True) as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    with manifest.open("x") as f:
        json.dump({"source_git_head": commit, "archive_sha256": file_sha(out),
                   "plans": index, "file_sha256": checksums}, f, indent=2)
        f.write("\n")
    print(out, manifest)


if __name__ == "__main__":
    main()

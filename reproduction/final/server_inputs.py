"""Collect exact reviewed inputs from the experiment checkout by SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "late-reviewed-v1.zip"


def read(path):
    return json.loads(path.read_text())


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected():
    ref = HERE / "reference"
    latest = read(ref / "receiver_robustness_human_audit.json")
    attenuation = read(HERE / "attenuation/attenuation_results.json")["source_sha256"]
    return {
        "original.json": attenuation["original"],
        "fresh.json": attenuation["fresh"],
        "position.json": read(ref / "position_control_results.json")["source_sha256"],
        "joint.json": read(ref / "joint_relabel_results.json")["reviewed_sha256"],
        "human_earlier.json": read(HERE / "human_validation/results.json")["reviewed_source_sha256"],
        "qwen.json": read(ref / "qwen_interleaved_results.json")["reviewed_sha256"],
        "human_robustness.json": latest["human_source_sha256"],
        "audit_model_labels.json": latest["model_labels_sha256"],
    }


def select(roots):
    required = expected()
    paths = set()
    for root in roots:
        if not root.exists():
            raise ValueError(f"Search path does not exist: {root}")
        if root.is_file():
            paths.add(root)
        else:
            paths.update(root.rglob("reviewed_source*.json"))
            paths.update(root.rglob("audit_model_labels.json"))
    wanted = {digest: key for key, digest in required.items()}
    found = {}
    for path in sorted(paths):
        if path.is_file() and (key := wanted.get(file_sha(path))):
            found.setdefault(key, path)
    for name in required:
        print(f"{name}: {found.get(name, 'MISSING')} ({required[name]})")
    return required, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="append", type=Path, default=[],
                        help="Additional saved export file or directory")
    parser.add_argument("--write", action="store_true", help="Create the archive after checking all inputs")
    parser.add_argument("--localization-reviewed", type=Path,
                        default=Path("artifacts/review/token_localization_resolved/reviewed_source.json"))
    args = parser.parse_args()
    required, found = select([Path("artifacts"), *args.search])
    if found.keys() != required.keys():
        parser.exit(2, "Some reviewed exports are missing. Supply their files via --search.\n")
    localization = args.localization_reviewed
    if not localization.is_file():
        parser.exit(2, "Supply the token-localization reviewed export via --localization-reviewed.\n")
    print(f"token_localization.json: {localization}")
    if args.write:
        with zipfile.ZipFile(ARCHIVE, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as out:
            for name, path in sorted(found.items()):
                out.write(path, name)
            out.write(localization, "token_localization.json")
        print(ARCHIVE)


if __name__ == "__main__":
    main()

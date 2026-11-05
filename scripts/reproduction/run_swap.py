"""Score teacher-forced continuations under layer-20 activation swaps."""
import argparse
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

from ddmi.provenance import adapter_manifest, runtime_metadata
from ddmi.publication import digest, validate_plan

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ("original", "adapted")
EDITS = ("none", "adaptive", "add", "subtract")
ADAPTER = Path("artifacts/safety_finetuning/llama31_8b_instruct_safety_lora")

def freeze(output):
    parent = json.loads(Path("artifacts/plans/fresh_absolute/seed42.json").read_text())
    validate_plan(parent)
    base = parent["payload"]
    primary = [c for c in base["conditions"] if c["source"] == "evidence"]
    if (len(primary) != 2 or any((c["layer"], c["mode"], c["amount"]) != (20, "absolute_l2", 4)
                                 for c in primary)
            or len(base["prompts"]) != 120 or len({r["domain"] for r in base["prompts"]}) != 30
            or base["config"]["precision"] != "nf4"):
        raise ValueError("Expected the fresh-absolute seed-42 parent plan")
    manifest = adapter_manifest(ADAPTER)
    if manifest["files"] != base["config"]["fixed_direction_receiver_files"]:
        raise ValueError("Seed-42 adapter differs from the original plan")
    design = dict(parent_plan_sha256=parent["plan_sha256"], model_id=base["model_id"],
                  revision=base["config"]["model_revision"], adapter=manifest,
                  layer=20, dose=4.0, direction=base["vectors"]["evidence:L20"],
                  prompts=base["prompts"], probe=json.loads((ROOT/"probe.json").read_text()),
                  edits=EDITS, self_swap_tolerance=1e-5, dose_tolerance=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(dict(design=design, sha256=digest(design)), f, indent=2, allow_nan=False)
    print("Frozen swap plan", digest(design))

def run(plan_path, output, smoke):
    import torch
    from peft import PeftModel
    from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer

    plan = json.loads(plan_path.read_text())
    d = plan["design"]
    if digest(d) != plan["sha256"] or adapter_manifest(ADAPTER) != d["adapter"]:
        raise ValueError("Swap plan or adapter checksum mismatch")
    prompts = d["prompts"][:2] if smoke else d["prompts"]
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(plan_sha256=plan["sha256"], smoke=smoke, runtime=runtime_metadata(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing output uses a different plan, runner, or runtime")
    else:
        with manifest_path.open("x") as f:
            json.dump(manifest, f, indent=2, allow_nan=False)
    completed = {}
    for path in output.glob("row_*.json"):
        row = json.loads(path.read_text())
        pid = row["prompt_id"]
        if pid in completed or row["plan_sha256"] != plan["sha256"]:
            raise ValueError("Conflicting saved swap row")
        completed[pid] = row
    if not set(completed) <= {r["id"] for r in prompts}:
        raise ValueError("Unexpected saved prompt IDs")
    if len(completed) < len(prompts):
        model, tokenizer = load_model_and_tokenizer(ModelLoadConfig(
            model_id=d["model_id"], revision=d["revision"], load_in_4bit=True,
            torch_dtype="float16", attn_implementation="eager"))
        wrapper = PeftModel.from_pretrained(model, str(ADAPTER))
        wrapper.eval()
        core = wrapper.base_model.model
        boundary = dict(core.named_modules())[f"model.layers.{d['layer']}"]
        vector = torch.as_tensor(d["direction"], dtype=torch.float32, device=core.device)
        vector = vector / vector.norm()

        def forward(receiver, ids, replace=None, capture=False):
            seen = []
            def hook(_module, _inputs, result):
                hidden = result[0] if isinstance(result, tuple) else result
                seen.append(hidden.detach().clone() if capture else None)
                if replace is None:
                    return None
                if replace.shape != hidden.shape or replace.dtype != hidden.dtype:
                    raise ValueError("Incompatible boundary state")
                return (replace, *result[1:]) if isinstance(result, tuple) else replace
            handle = boundary.register_forward_hook(hook)
            try:
                adapter_state = wrapper.disable_adapter() if receiver == "original" else nullcontext()
                with adapter_state, torch.inference_mode():
                    logits = core(input_ids=ids, use_cache=False).logits
            finally:
                handle.remove()
            if len(seen) != 1:
                raise ValueError("Expected one boundary call per forward pass")
            return logits, seen[0]

        def score(logits, ids, prefix_len):
            indices = ids[0, prefix_len:]
            logits = logits[0, prefix_len-1:-1].float()
            values = logits.gather(1, indices[:, None]).squeeze(1) - logits.logsumexp(dim=1)
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite continuation log probability")
            return values

        def edit(state, mode):
            if mode == "none":
                return state, 0.0
            h = state.float()
            direction = vector.to(h.device)
            sign = {"add": 1, "subtract": -1}.get(mode)
            if sign is None:
                sign = torch.where((h @ direction) < 0, 1.0, -1.0).unsqueeze(-1)
            target = d["dose"] * sign
            patched = (h + target * direction).to(state.dtype)
            delta = patched.float() - h
            norm_error = (delta.norm(dim=-1) - d["dose"]).abs().max()
            signed_error = ((delta @ direction).unsqueeze(-1) - target).abs().max()
            error = float(torch.maximum(norm_error, signed_error))
            if not math.isfinite(error) or error > d["dose_tolerance"]:
                raise ValueError(f"Post-cast signed or norm displacement error {error}")
            return patched, error

        for row in prompts:
            pid = row["id"]
            if pid in completed:
                continue
            text = tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]}],
                                                  tokenize=False, add_generation_prompt=True)
            prefix = tokenizer(text)["input_ids"]
            scores = []
            native_error = dose_error = 0.0
            for pair_index, pair in enumerate(d["probe"]):
                for category, continuation in pair.items():
                    suffix = tokenizer(continuation, add_special_tokens=False)["input_ids"]
                    if not suffix or len(prefix)+len(suffix) > 512:
                        raise ValueError("Unexpected probe tokenization length")
                    ids = torch.tensor([prefix+suffix], device=core.device)
                    native = {}
                    states = {}
                    for name in CHECKPOINTS:
                        logits, states[name] = forward(name, ids, capture=True)
                        native[name] = score(logits, ids, len(prefix))
                        del logits
                    for donor in CHECKPOINTS:
                        for mode in EDITS:
                            patched, err = edit(states[donor], mode)
                            dose_error = max(dose_error, err)
                            for receiver in CHECKPOINTS:
                                logits, _ = forward(receiver, ids, replace=patched)
                                vals = score(logits, ids, len(prefix))
                                del logits
                                if donor == receiver and mode == "none":
                                    err = float((vals - native[receiver]).abs().max())
                                    native_error = max(native_error, err)
                                    if err > d["self_swap_tolerance"]:
                                        raise ValueError(f"{pid}/{receiver}: self-swap logprob error {err}")
                                scores.append(dict(pair=pair_index, category=category, donor=donor,
                                                   receiver=receiver, edit=mode, mean_logprob=float(vals.mean())))
            result = dict(prompt_id=pid, family=row["domain"], scores=scores,
                          self_swap_max_error=native_error, signed_or_norm_max_error=dose_error,
                          plan_sha256=plan["sha256"])
            path = output/f"row_{pid}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            temp.replace(path)
            completed[pid] = result
            print(f"{len(completed)}/{len(prompts)} prompts saved | self-swap {native_error:.6g} | dose error {dose_error:.6g}", flush=True)
    rows = [completed[r["id"]] for r in prompts]
    temp = output/"results.tmp"
    temp.write_text(json.dumps(dict(manifest=manifest, rows=rows), indent=2, allow_nan=False)+"\n")
    temp.replace(output/"results.json")
    print("Verified", len(rows), "prompts; maximum self-swap error",
          max(row["self_swap_max_error"] for row in rows))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("plan", type=Path)
    for name in ("smoke", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("plan", type=Path)
        cmd.add_argument("output", type=Path)
    args = p.parse_args()
    if args.command == "freeze":
        freeze(args.plan)
    else:
        run(args.plan, args.output, args.command == "smoke")

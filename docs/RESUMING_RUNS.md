# Recover a measured run

Keep the original frozen plan, adapter and output directory. Do not re-extract directions, regenerate the plan, change package versions or launch a second process into the same directory.

First perform a read-only progress check (no model generation):

```bash
python scripts/run_publication_study.py run \
  --plan artifacts/plans/measurement_pilot.json \
  --adapter-path artifacts/safety_finetuning/llama31_8b_instruct_safety_lora \
  --output-dir artifacts/runs/measurement_pilot \
  --resume --check-only
```

Recovery validates both plan checksums; the checkpoint's adapter and runtime metadata; every saved condition and prompt; and complete, finite telemetry. Identical records in the checkpoint and journal are deduplicated. Conflicting records are rejected. Fully saved conditions and receivers are skipped. Each missing prompt is generated independently with the existing greedy, single-sequence protocol. This continues the documented configuration; it is not a claim of bitwise reproducibility across all GPU kernels.

After the check passes, start a persistent terminal session:

```bash
tmux new -s measurement-pilot
```

Inside it, activate the same environment and resume:

```bash
cd /home/ubuntu/dual-direction-mech-interp
source .venv/bin/activate
set -o pipefail
python -u scripts/run_publication_study.py run \
  --plan artifacts/plans/measurement_pilot.json \
  --adapter-path artifacts/safety_finetuning/llama31_8b_instruct_safety_lora \
  --output-dir artifacts/runs/measurement_pilot \
  --resume 2>&1 | tee -a artifacts/runs/measurement_pilot_resume.log
```

Detach with Ctrl-b followed by d; reattach with `tmux attach -t measurement-pilot`. tmux protects against terminal disconnection, not GPU faults, process termination or a machine reboot. Keep the traceback if generation fails again.

Before changing progress, recovery copies the original plan, checkpoint and full journal into a unique `resume_backups/` subdirectory of the run directory. The recovered checkpoint records this backup and recovery counts in `resume_history`. A malformed complete journal line is an error. A final unterminated partial JSON record can be removed only after the full original journal has been backed up. A complete JSON record missing only its newline is preserved and separated before new records are appended.

A persistent `.run.lock` file is normal. The OS releases its advisory lock if the process exits or is killed, so the file need not be deleted. Two cooperating runner processes cannot write the same output directory at once. Earlier runner versions do not acquire this lock; stop any older process before using recovery.

On completion, `run.json` contains all conditions and responses and sets `complete: true`. The journal remains append-only except for the explicit backed-up final-line repair. A completed run can be checked without regenerating responses.

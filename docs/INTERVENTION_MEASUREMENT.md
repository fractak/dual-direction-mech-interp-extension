# What the new measured intervention does

For unit direction d and incoming hidden state h, the archived operator is:

`h' = h - lambda * <h,d> * d`

Its displacement is `lambda * abs(<h,d>)`. Matching d and lambda does not match displacement. The measured `projection` mode retains this operator and records projections and movement.

The new `relative_l2` operator is:

`h' = h - rho * ||h|| * sign(<h,d>) * d`

In exact arithmetic, for nonzero h, `||h'-h|| / ||h|| = rho` at every edited token. A zero projection uses sign +1, and such cases are counted. A zero-norm activation does not move. The output is cast back to the incoming dtype; telemetry measures the actual cast-back displacement. Very small edits can round to zero. Inspect the recorded min/max relative displacement rather than assuming perfect matching in finite precision.

Relative matching is a new directional perturbation. It is not projection removal and does not prove that the original projection-edit gap was mediated by movement magnitude. It may amplify tiny projections, and generated trajectories can diverge. Random controls use exactly the same relative budget and sign convention.

The runner uses one unpadded sequence at a time, a KV cache, and hooks at block outputs during prefill and decoding. For each response it stores per-forward-call token counts; norm, absolute projection, absolute displacement and relative displacement means/RMS/min/max/last values; and zero-projection/zero-norm counts. The first call is labeled prefill; subsequent cached calls must contain one token. No full activation arrays are retained.

The last prefill token has a common input context across receivers, making it particularly useful for a matched-input diagnostic. Decode measurements are conditional on each model's generated history; they are not a common teacher-forced trajectory. A stricter mechanistic follow-up should also compare predeclared shared continuations or activation patching. Neither is implemented here.

Baseline conditions observe both source vectors without changing activations. Repeated baselines across observation layers are repeated greedy generations, not additional independent contexts. Responses are capped at 128 new tokens, matching the archived protocol. The new runner records input length, truncation, generated length and whether the cap was reached. It does not infer whether a response reaching the cap would have continued.

Partial runs are written incrementally with `complete: false`; only successfully finished runs set it true. Reuse of an output directory is refused. This implementation does not resume a failed run; preserve its diagnostics and rerun into a new directory. Frozen evidence is never changed.

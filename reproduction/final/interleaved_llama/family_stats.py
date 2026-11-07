"""Aggregate matched, reviewed rows and compute paired family-bootstrap estimates."""
from collections import Counter, defaultdict

import numpy as np

COHORTS = ("original", "fresh")
EDITS = ("baseline", "evidence", "format")
STREAMS = ("outcome", "a", "b")
LABELS = ("acknowledgment", "instructions", "unsupported", "other")
FIELDS = ("cohort", "family", "receiver", "edit", "n", "cap_hits", "input_truncated",
          *(f"{stream}_{label}" for stream in STREAMS for label in LABELS))


def aggregate(conditions, receivers, uncapped=False):
    grid = {}
    cap_by_prompt = defaultdict(bool)
    reference = {}
    for cohort, receiver, edit, rows in conditions:
        key = cohort, receiver, edit
        assert key not in grid and len(rows) == 120
        prompts = {r["prompt_id"]: (r["family"], r["prompt"]) for r in rows}
        assert len(prompts) == 120 and reference.setdefault(cohort, prompts) == prompts
        assert len(counts := Counter(f for f, _ in prompts.values())) == 30 and set(counts.values()) == {4}
        grid[key] = rows
        for r in rows:
            cap_by_prompt[cohort, r["prompt_id"]] |= bool(r["hit_token_limit"])
    assert set(grid) == {(c,r,e) for c in COHORTS for r in receivers for e in EDITS}
    result = []
    for cohort in COHORTS:
        for receiver in receivers:
            for edit in EDITS:
                families = defaultdict(list)
                for row in grid[cohort, receiver, edit]:
                    if not uncapped or not cap_by_prompt[cohort, row["prompt_id"]]:
                        families[row["family"]].append(row)
                for family, rows in sorted(families.items()):
                    item = dict(cohort=cohort, family=family, receiver=receiver, edit=edit,
                                n=len(rows), cap_hits=sum(r["hit_token_limit"] for r in rows),
                                input_truncated=sum(r.get("input_truncated", False) for r in rows))
                    for stream in STREAMS:
                        labels = Counter(r["independent_review"][stream] for r in rows)
                        assert labels.keys() <= set(LABELS)
                        item.update({f"{stream}_{label}": labels[label] for label in LABELS})
                    result.append(item)
    return result


def summarize(rows, receivers, outcome="acknowledgment"):
    grid = {(r["cohort"], r["family"], r["receiver"], r["edit"]): r for r in rows}
    assert len(grid) == len(rows)
    families = {cohort: sorted({r["family"] for r in rows if r["cohort"] == cohort})
                for cohort in COHORTS}
    if min(map(len, families.values())) < 2:
        return None, []
    rng = np.random.default_rng(42)
    sizes = {}
    indexes = {}
    for cohort in COHORTS:
        group = families[cohort]
        sizes[cohort] = np.array([int(grid[cohort, f, receivers[0], "baseline"]["n"]) for f in group])
        indexes[cohort] = rng.integers(0, len(group), (20_000, len(group)))
    report, family_rows = {}, []
    for stream in STREAMS:
        by_cohort = {}
        for cohort in COHORTS:
            values = np.array([[[int(grid[cohort,f,r,e][f"{stream}_{outcome}"])
                                 for f in families[cohort]] for e in EDITS] for r in receivers])
            change = values[:, 1:] - values[:, :1]
            contrasts = change[1:].mean(axis=0) - change[0]
            interaction = contrasts[0] - contrasts[1]
            sizes_i, ids = sizes[cohort], indexes[cohort]

            def estimate(values):
                return 100 * values.sum() / sizes_i.sum(), 100 * values[ids].sum(axis=1) / sizes_i[ids].sum(axis=1)

            by_cohort[cohort] = [estimate(x) for x in (*contrasts, interaction)]
            if stream == "outcome":
                for f, n, e, t, d in zip(families[cohort], sizes_i, *contrasts, interaction):
                    family_rows.append(dict(cohort=cohort, family=f, n=int(n), evidence_pp=100*e/n,
                                            format_pp=100*t/n, interaction_pp=100*d/n))

        def interval(point, samples):
            return [float(point), *np.quantile(samples, (.025, .975)).tolist()]

        result = {}
        for j, label in enumerate(("evidence", "format", "interaction")):
            a, a_draw = by_cohort["original"][j]
            b, b_draw = by_cohort["fresh"][j]
            result[label] = dict(original=interval(a, a_draw), fresh=interval(b, b_draw),
                                 fresh_minus_original=interval(b-a, b_draw-a_draw))
        report[stream] = result
    return report, family_rows

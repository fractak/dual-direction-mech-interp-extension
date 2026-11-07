"""Exploratory comparison of two frozen evaluation sets. Requires NumPy."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = ('acknowledgment', 'instructions', 'unsupported', 'other')
TYPES = {
    'original': {
        'reading_count_or_mark': 'beehive book_press bread_proofer cheese_press fermentation film_developer glaze_batch screen_print',
        'equipment_configuration': 'circuit_board darkroom embroidery geared_tripod guitar_pedal heat_press jewelry_scale laser_level loom microscope miter_box model_railway overlocker paper_cutter record_player soldering_station telescope tumbler wood_lathe',
        'spatial_identity': 'compost_bin mosaic stamp_album',
    },
    'fresh': {
        'reading_count_or_mark': 'abacus bowling croquet foosball harmonica hockey kayak mandolin panflute pinball rowing saxophone shuttlecocks sudoku table_tennis volleyball xylophone',
        'equipment_configuration': 'accordion bagpipe bicycle coin_sorter harp metronome ocarina ukulele',
        'spatial_identity': 'bocce chess dartboard dominoes mahjong',
    },
}


def load_grid(path, cohort):
    raw = path.read_bytes()
    run = json.loads(raw)
    assert run['complete'] is True
    grid, prompts = {}, None
    for condition in run['conditions']:
        if condition['mode'] == 'observe':
            mode = 'baseline'
        elif condition['source'] in ('evidence', 'format'):
            mode = condition['source']
        else:
            continue
        receiver = condition['receiver'].replace('safety', 'seed')
        rows = {r['prompt_id']: r for r in condition['rows']}
        assert len(rows) == len(condition['rows']) == 120
        reference = {i: (r['domain'], r['prompt']) for i, r in rows.items()}
        if prompts is None:
            prompts = reference
        assert reference == prompts
        assert (receiver, mode) not in grid
        grid[receiver, mode] = rows
    ids = sorted(prompts)
    families = sorted({family for family, _ in prompts.values()})
    assert len(families) == 30
    mask = np.array([[prompts[i][0] == f for i in ids] for f in families], int)
    assert np.all(mask.sum(axis=1) == 4)
    type_map = {name: kind for kind, names in TYPES[cohort].items() for name in names.split()}
    assert set(type_map) == set(families)
    return grid, ids, families, mask, type_map, prompts, hashlib.sha256(raw).hexdigest()


def condition_arrays(grid, ids, stream):
    keys = ('original_instruct', 'seed42', 'seed43', 'seed44')
    return np.array([[[LABELS.index(grid[r, m][i]['independent_review'][stream])
                       for i in ids] for m in ('baseline', 'evidence', 'format')] for r in keys])


def summarize_families(z, mask):
    ack = (z == 0).astype(float)
    changes = ack[:, 1:] - ack[:, :1]
    receiver = changes[1:].mean(axis=0) - changes[0]
    interaction = receiver[0] - receiver[1]
    assert np.allclose(interaction, (ack[1:, 1] - ack[1:, 2]).mean(axis=0)
                       - (ack[0, 1] - ack[0, 2]))
    def mean_family(values):
        return values @ mask.T / 4
    return {
        'baseline': mean_family(ack[:, 0]),
        'original_changes': mean_family(changes[0]),
        'adapted_changes': mean_family(changes[1:].mean(axis=0)),
        'receiver': mean_family(receiver),
        'interaction': mean_family(interaction),
    }


def matrices(z):
    return np.array([[[[int(np.sum((r[0] == a) & (r[m] == b))) for b in range(4)]
                       for a in range(4)] for m in (1, 2)] for r in z])


def diag(matrix):
    """Flow contributions in percentage points per response, and conditional rates."""
    total = matrix.sum()
    b = matrix[0].sum() / total
    retention = matrix[0, 0] / matrix[0].sum()
    gain = matrix[1:, 0].sum() / matrix[1:].sum()
    return {
        'baseline_ack': 100 * b,
        'loss': float(100 * matrix[0, 1:].sum() / total),
        'gain': float(100 * matrix[1:, 0].sum() / total),
        'conditional_loss': 100 * (1 - retention),
        'conditional_gain': 100 * gain,
        'loss_by_destination': (100 * matrix[0, 1:] / total).tolist(),
        'gain_by_origin': (100 * matrix[1:, 0] / total).tolist(),
    }


def baseline_decomposition(old, new):
    # Symmetric decomposition of b*(-loss)+(1-b)*gain in each receiver/edit cell.
    result = []
    for receiver in range(4):
        for edit in range(2):
            left, right = [diag(m[receiver, edit]) for m in (old, new)]
            b0, b1 = [d['baseline_ack']/100 for d in (left, right)]
            loss0, loss1 = [d['conditional_loss'] for d in (left, right)]
            gain0, gain1 = [d['conditional_gain'] for d in (left, right)]
            mean_b = (b0 + b1) / 2
            mix = -(b1 - b0) * (loss0 + loss1 + gain0 + gain1) / 2
            loss_rate = -mean_b * (loss1 - loss0)
            gain_rate = (1 - mean_b) * (gain1 - gain0)
            net = (right['gain'] - right['loss']) - (left['gain'] - left['loss'])
            assert np.isclose(net, mix + loss_rate + gain_rate)
            result.append([receiver, edit, net, mix, loss_rate, gain_rate])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original', type=Path)
    p.add_argument('fresh', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cohorts = ('original', 'fresh')
    loaded = {c: load_grid(path, c) for c, path in zip(cohorts, (a.original, a.fresh))}
    results = {'source_sha256': {c: loaded[c][-1] for c in cohorts},
               'category_order': LABELS, 'analyses': {}}
    for stream in ('outcome', 'a', 'b'):
        summary, data, all_mat, uncapped = {}, {}, {}, {}
        for cohort in cohorts:
            grid, ids, families, mask, type_map, prompts, _ = loaded[cohort]
            z = condition_arrays(grid, ids, stream)
            mat = matrices(z)
            all_mat[cohort] = mat
            fam = summarize_families(z, mask)
            keep = np.array([all(not rows[i]['hit_token_limit'] for rows in grid.values()) for i in ids])
            ack = (z == 0).astype(float)
            per_prompt = (ack[1:, 1] - ack[1:, 2]).mean(axis=0) - (ack[0, 1] - ack[0, 2])
            uncapped[cohort] = (mask[:, keep] @ per_prompt[keep], mask[:, keep].sum(axis=1))
            data[cohort] = fam
            interaction = 100 * fam['interaction']
            within = {}
            for kind in TYPES[cohort]:
                idx = [i for i, f in enumerate(families) if type_map[f] == kind]
                within[kind] = {'n': len(idx), 'interaction': interaction[idx].mean(),
                                'evidence': (100 * fam['receiver'][0, idx]).mean(),
                                'format': (100 * fam['receiver'][1, idx]).mean()}
            summary[cohort] = {
                'mean': interaction.mean(), 'median': np.median(interaction),
                'quartiles': np.quantile(interaction, [.25, .75]).tolist(),
                'negative_zero_positive': [int(np.sum(interaction < -1e-9)), int(np.sum(abs(interaction) <= 1e-9)),
                                           int(np.sum(interaction > 1e-9))],
                'trimmed_10_percent': np.sort(interaction)[3:-3].mean(),
                'leave_one_out_range': [float((interaction.sum()-interaction.max())/29),
                                        float((interaction.sum()-interaction.min())/29)],
                'extremes': [[families[i], float(interaction[i])] for i in
                             list(np.argsort(interaction)[:3]) + list(np.argsort(interaction)[-3:])],
                'baseline_rates': (100 * fam['baseline'].mean(axis=1)).tolist(),
                'original_edits': (100 * fam['original_changes'].mean(axis=1)).tolist(),
                'adapted_edits': (100 * fam['adapted_changes'].mean(axis=1)).tolist(),
                'receiver_edits': (100 * fam['receiver'].mean(axis=1)).tolist(),
                'receiver_edit_family_medians': (100 * np.median(fam['receiver'], axis=1)).tolist(),
                'uncapped_prompt_count': sum(all(not rows[i]['hit_token_limit'] for rows in grid.values()) for i in ids),
                'question_types': within,
                'transitions': mat.tolist(),
                'transition_diagnostics': {
                    receiver: {mode: diag(m) for mode, m in zip(('evidence', 'format'), cells)}
                    for receiver, cells in zip(('original', 'pooled_adapted'), (mat[0], mat[1:].sum(axis=0)))
                },
            }
            if stream == 'outcome':
                with (a.output / f'{cohort}_families.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['family','question_type','example_prompt','original_baseline_ack_pp',
                                     'mean_adapted_baseline_ack_pp','evidence_receiver_contrast_pp',
                                     'format_receiver_contrast_pp','interaction_pp'])
                    examples = {}
                    for _, (family, prompt) in prompts.items():
                        examples.setdefault(family, prompt)
                    for i, family in enumerate(families):
                        writer.writerow([family,type_map[family],examples[family],100*fam['baseline'][0,i],
                                         100*fam['baseline'][1:,i].mean(),100*fam['receiver'][0,i],
                                         100*fam['receiver'][1,i],interaction[i]])
        rng = np.random.default_rng(42)
        original_ix, fresh_ix = [rng.integers(0, 30, size=(20000, 30)) for _ in cohorts]
        def shift(key):
            old, new = [100 * data[c][key] for c in cohorts]
            delta = new[..., fresh_ix].mean(axis=-1) - old[..., original_ix].mean(axis=-1)
            return np.stack([new.mean(axis=-1)-old.mean(axis=-1),
                             *np.quantile(delta, [.025,.975], axis=-1)], axis=-1).tolist()
        cap_draws, cap_points = [], []
        for cohort, indices in zip(cohorts, (original_ix, fresh_ix)):
            numerators, denominators = uncapped[cohort]
            cap_draws.append(100 * numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1))
            cap_points.append(100 * numerators.sum() / denominators.sum())
        original, fresh = (100 * data[c]['interaction'] for c in cohorts)
        decomp = baseline_decomposition(all_mat['original'], all_mat['fresh'])
        combined = np.array([x[2:] for x in decomp]).reshape(4, 2, 4)
        by_direction = combined[1:].mean(axis=0) - combined[0]
        def question_decomposition():
            old, new = [summary[c]['question_types'] for c in cohorts]
            composition = 0.
            within = 0.
            for k in old:
                p0, p1 = [d[k]['n']/30 for d in (old, new)]
                v0, v1 = [d[k]['interaction'] for d in (old, new)]
                composition += (p1-p0)*(v0+v1)/2
                within += (p0+p1)*(v1-v0)/2
            return {'composition': composition, 'within_type': within}
        results['analyses'][stream] = {
            **summary, 'fresh_minus_original': shift('interaction'),
            'uncapped_fresh_minus_original': [cap_points[1]-cap_points[0],
                *np.quantile(cap_draws[1]-cap_draws[0], [.025,.975]).tolist()],
            'direction_shifts_evidence_format': shift('receiver'),
            'adversarial_drop_three_per_set_shift':
                [float(np.sort(fresh)[:-3].mean()-np.sort(original)[3:].mean()),
                 float(np.sort(fresh)[3:].mean()-np.sort(original)[:-3].mean())],
            'question_type_decomposition': question_decomposition(),
            'baseline_decomposition_columns': ['net_shift','baseline_mix','conditional_loss_rate','conditional_gain_rate'],
            'baseline_decomposition_by_direction': by_direction.tolist(),
            'baseline_decomposition_interaction': (by_direction[0]-by_direction[1]).tolist(),
        }
    (a.output/'attenuation_results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

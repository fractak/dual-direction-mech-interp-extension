import argparse
import collections
import csv
import json
from pathlib import Path

from analyze_replication_positions import POSITION_MODES, read_study, summarize

parser = argparse.ArgumentParser(description="Audit both reviewed exports and reproduce the earlier-label sensitivity.")
parser.add_argument("--prior-primary", type=Path, default=Path("artifacts/review/fresh_absolute_resolved/reviewed_source.json"))
parser.add_argument("--results-dir", type=Path, required=True)
parser.add_argument("--review-dir", type=Path, help="Optional directory containing the four completed reviewer CSVs.")
parser.add_argument("--adjudications-dir", type=Path, help="Optional directory containing the two adjudication CSVs.")
args = parser.parse_args()
if (args.review_dir is None) != (args.adjudications_dir is None):
    parser.error("Supply both reviewer and adjudication directories to verify the merge.")
original = json.loads(args.prior_primary.read_bytes())
prior = {}
for condition in original['conditions']:
    for row in condition['rows']:
        key = row['prompt'], row['response']
        value = row['independent_review']['outcome']
        assert prior.setdefault(key, value) == value
del original
old_prompts = {' '.join(prompt.casefold().split()) for prompt, _ in prior}
for study, pair_count in [('family_replication', 1041), ('position_control', 460)]:
    grid, reference, pairs, receivers, modes, _ = read_study(Path(f'artifacts/review/{study}_resolved/reviewed_source.json'), study)
    def read_csv(path):
        with open(path, encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    if args.review_dir is not None:
        adjudications = {row['review_id']: row for row in read_csv(args.adjudications_dir / f'{study}_adjudications.csv')}
        a, b = [read_csv(args.review_dir / f'reviewer_{side}_completed_{pair_count}.csv') for side in 'AB']
        assert len(a) == len(b) == len(pairs) == pair_count
        for ra, rb in zip(a, b):
            key = ra['prompt'], ra['response']
            rid = ra['review_id']
            assert (rb['prompt'], rb['response'], rb['review_id']) == (*key, rid)
            review = pairs[key]
            assert (review['a'], review['b']) == (ra['outcome'], rb['outcome'])
            if ra['outcome'] != rb['outcome']:
                decision = adjudications[rid]
                assert review['outcome'] == decision['outcome']
                assert review['notes']['adjudication'] == decision['notes']
            else:
                assert review['outcome'] == ra['outcome'] and rid not in adjudications
    results_file = args.results_dir / f'{study}_results.json'
    results = json.loads(results_file.read_bytes())
    if args.review_dir is not None:
        results['merge_verified_against_reviewers_and_adjudications'] = True
    overlap = sum(' '.join(prompt.casefold().split()) in old_prompts for prompt, _ in reference.values())
    matched_pairs = pairs.keys() & prior.keys()
    conflicts = {pair for pair in matched_pairs if prior[pair] != pairs[pair]['outcome']}
    results['prior_primary_overlap'] = {
        'normalized_exact_prompts': overlap,
        'exact_prompt_response_pairs': len(matched_pairs),
        'conflicting_outcomes': len(conflicts),
    }
    if study == 'family_replication':
        assert overlap == 0
    else:
        token_labels = collections.defaultdict(set)
        positions = edits = 0
        max_error = 0.
        for (receiver, mode), rows in grid.items():
            for row in rows.values():
                ids = row['prefill_token_ids']
                pos = row['position_control']
                indices = pos['indices']
                assert len(ids) == row['input_tokens']
                assert len(set(indices.values())) == 4
                for name, offset in (('final_prefill', 1), ('previous_prefill', 2), ('two_back_prefill', 3)):
                    assert indices[name] == len(ids) - offset
                for name, index in indices.items():
                    token = next(x for x in pos['neighborhoods'][name] if x['index'] == index)
                    assert token['id'] == ids[index]
                    token_labels[name].add((ids[index], token['token']))
                target = None if mode == 'baseline' else indices[POSITION_MODES[mode]]
                assert target == pos['selected_index']
                calls = row['telemetry']['evidence']
                assert len(calls) == len(row['generated_token_ids']) == row['generated_tokens']
                for call_index, call in enumerate(calls):
                    assert call['phase'] == ('prefill' if call_index == 0 else 'decode')
                    n = call['tokens']
                    assert n == (len(ids) if call_index == 0 else 1)
                    positions += n
                    if target is None:
                        assert call['displacement_l2']['min'] == call['displacement_l2']['max'] == 0
                    else:
                        telemetry = call['token_localization']
                        assert all(len(v) == n for v in telemetry.values())
                        for i, (sign, signed, norm) in enumerate(zip(telemetry['sign'], telemetry['signed_displacement'], telemetry['displacement_l2'])):
                            is_edited = call_index == 0 and i == target
                            assert sign == (-1 if is_edited else 0)
                            if is_edited:
                                edits += 1
                                error = max(abs(signed + 4), abs(norm - 4))
                                assert error <= .04
                                max_error = max(max_error, error)
                            else:
                                assert signed == norm == 0
        assert edits == 960 and positions == 219566
        results['position_telemetry_audit'] = {
            'positions_checked': positions, 'edited_positions': edits, 'max_displacement_error': max_error,
            'tokens': {name: sorted(tokens) for name, tokens in token_labels.items()},
        }
        changed_rows = 0
        for rows in grid.values():
            for row in rows.values():
                pair = row['prompt'], row['response']
                if pair in conflicts:
                    row['independent_review'] = dict(row['independent_review'], outcome=prior[pair])
                    changed_rows += 1
        results['earlier_primary_label_sensitivity'] = {
            'policy': 'Use the earlier primary-study adjudicated outcome for exact duplicate prompt-response pairs only.',
            'changed_rows': changed_rows,
            'acknowledgment': summarize(grid, reference, receivers, modes, False, 'outcome', 'acknowledgment'),
            'unsupported': summarize(grid, reference, receivers, modes, False, 'outcome', 'unsupported'),
        }
    results_file.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(study, results['prior_primary_overlap'])
    for field in ('position_telemetry_audit','earlier_primary_label_sensitivity'):
        if field in results:
            print(field, results[field])

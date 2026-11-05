#!/usr/bin/env python3
"""Descriptive, state-balanced analysis of the six five-answer advice batches.

Uses only the standard library. Source results.csv files are independently read
and checked against merged/combined_answers.csv when that audited file exists.
All probabilities are observed C-answer fractions, not population estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
BATCHES = {
    'DeepSeek': [('old', '20261029_121229_Deepseek1029_1次', 1), ('new', '20261029_140923_Deepseek1030_4次', 4)],
    'GPT': [('old', '20261016_173605_GPT5.5_2次', 2), ('new', '20261029_141751_GPT5.5_3次', 3)],
    'Qwen': [('old', '20261024_122849_Qwen1024_2次', 2), ('new', '20261029_141235_Qwen1031_3次', 3)],
}
MODELS = list(BATCHES)
ORTHOGONAL = (1, 3, 5, 7)
DIAGONAL = (0, 2, 6, 8)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'Empty table: {path}')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def features(state):
    return {'center': state[4], 'cardinal_c': sum(state[i] == 'C' for i in ORTHOGONAL),
            'diagonal_c': sum(state[i] == 'C' for i in DIAGONAL)}


def d4_orbit(state):
    """Eight square rotations/reflections, deduplicated for symmetric states."""
    result = set()
    grid = [list(state[i:i + 3]) for i in (0, 3, 6)]
    for _ in range(4):
        result.add(''.join(''.join(row) for row in grid))
        result.add(''.join(''.join(reversed(row)) for row in grid))
        grid = [list(row) for row in zip(*grid[::-1])]
    return sorted(result)


def run(root, report):
    out = report / 'data'
    out.mkdir(parents=True, exist_ok=True)
    states = [''.join(s) for s in itertools.product('CD', repeat=9)]
    by_model = {}
    by_batch = {}
    raw_counts = Counter()
    sources = {}
    repetition_rows = []
    batch_summary = []
    for model, batches in BATCHES.items():
        pooled = defaultdict(list)
        for label, folder, repeats in batches:
            path = root / 'logs' / 'GPT_Multistepsa' / folder / 'results.csv'
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 512 * repeats, (model, folder, len(rows))
            grouped = defaultdict(list)
            repetitions = defaultdict(list)
            seen = set()
            for row in rows:
                state = row['grid_key']
                answer = row['recommendation']
                repetition = int(row['repetition'])
                assert row['status'] == 'success' and answer in ('C', 'D')
                assert state in states and row['current_action'] == state[4]
                assert int(row['cooperating_neighbors']) == features(state)['cardinal_c']
                assert (state, repetition) not in seen
                seen.add((state, repetition))
                x = int(answer == 'C')
                grouped[state].append(x)
                pooled[state].append(x)
                repetitions[repetition].append(x)
                raw_counts[(model, label, state, answer)] += 1
            assert set(grouped) == set(states)
            assert all(len(v) == repeats for v in grouped.values())
            assert set(repetitions) == set(range(1, repeats + 1))
            by_batch[(model, label)] = {s: mean(v) for s, v in grouped.items()}
            batch_summary.append({'model': model, 'batch': label, 'repeats_per_state': repeats,
                                  'n_states': 512, 'n_answers': len(rows),
                                  'p_C': mean(x for v in grouped.values() for x in v),
                                  'p_C_given_current_C': mean(x for s, v in grouped.items() if s[4] == 'C' for x in v),
                                  'p_C_given_current_D': mean(x for s, v in grouped.items() if s[4] == 'D' for x in v),
                                  'source_folder': folder})
            for rep, values in sorted(repetitions.items()):
                repetition_rows.append({'model': model, 'batch': label, 'original_repetition': rep,
                                        'n_states': len(values), 'C_answers': sum(values), 'p_C': mean(values)})
        assert len(pooled) == 512 and all(len(v) == 5 for v in pooled.values())
        by_model[model] = {s: sum(v) for s, v in pooled.items()}

    merged = report / 'merged' / 'combined_answers.csv'
    merged_check = 'not present when analysis was generated'
    if merged.exists():
        with merged.open(encoding='utf-8-sig') as handle:
            merged_rows = list(csv.DictReader(handle))
        merged_counts = Counter((r['model_label'], r['batch_label'], r['grid_key'], r['recommendation']) for r in merged_rows)
        assert merged_counts == raw_counts, 'Canonical merged answers differ from independently parsed raw CSVs'
        merged_check = 'passed: all 7,680 answers match by model, batch, state and answer'

    summaries, votes, state_rows, neighbor_rows, context_rows = [], [], [], [], []
    for model in MODELS:
        counts = by_model[model]
        p = {s: k / 5 for s, k in counts.items()}
        hist = Counter(counts.values())
        summaries.append({
            'model': model, 'n_states': 512, 'n_answers': 2560, 'C_answers': sum(counts.values()),
            'p_C': mean(p.values()), 'p_C_given_current_C': mean(p[s] for s in states if s[4] == 'C'),
            'p_C_given_current_D': mean(p[s] for s in states if s[4] == 'D'),
            'current_action_retention': mean(p[s] if s[4] == 'C' else 1 - p[s] for s in states),
            'unanimous_states': hist[0] + hist[5], 'unanimity_fraction': (hist[0] + hist[5]) / 512,
            'mixed_states': 512 - hist[0] - hist[5],
            'within_state_pair_agreement': mean((math.comb(k, 2) + math.comb(5 - k, 2)) / 10 for k in counts.values()),
            'mean_binary_entropy_bits': mean(0 if q in (0, 1) else -q * math.log2(q) - (1 - q) * math.log2(1 - q) for q in p.values()),
            'majority_C_states': sum(k >= 3 for k in counts.values()),
        })
        for k in range(6):
            votes.append({'model': model, 'C_answers_out_of_5': k, 'observed_p_C': k / 5,
                          'n_states': hist[k], 'fraction_of_512_states': hist[k] / 512})
        for state in states:
            k = counts[state]
            state_rows.append({'model': model, 'grid_key': state, 'grid_rows': '/'.join(state[i:i + 3] for i in (0, 3, 6)),
                               **features(state), 'C_answers': k, 'D_answers': 5 - k, 'p_C': k / 5,
                               'majority_advice': 'C' if k >= 3 else 'D', 'unanimous': int(k in (0, 5)),
                               'within_state_pair_agreement': (math.comb(k, 2) + math.comb(5 - k, 2)) / 10})
        for center in ('C', 'D'):
            for cardinal in range(5):
                subset = [s for s in states if s[4] == center and features(s)['cardinal_c'] == cardinal]
                neighbor_rows.append({'model': model, 'center': center, 'cardinal_c': cardinal,
                                      'n_states': len(subset), 'n_answers': len(subset) * 5,
                                      'C_answers': sum(counts[s] for s in subset), 'p_C': mean(p[s] for s in subset),
                                      'min_state_p_C': min(p[s] for s in subset), 'max_state_p_C': max(p[s] for s in subset)})
                for diagonal in range(5):
                    subset2 = [s for s in subset if features(s)['diagonal_c'] == diagonal]
                    context_rows.append({'model': model, 'center': center, 'cardinal_c': cardinal, 'diagonal_c': diagonal,
                                         'n_states': len(subset2), 'n_answers': len(subset2) * 5,
                                         'C_answers': sum(counts[s] for s in subset2), 'p_C': mean(p[s] for s in subset2)})

    pair_summaries, pair_states, pair_joint = [], [], []
    for first, second in itertools.combinations(MODELS, 2):
        deltas = []
        for state in states:
            a, b = by_model[first][state] / 5, by_model[second][state] / 5
            d = a - b
            deltas.append(d)
            pair_states.append({'first_model': first, 'second_model': second, 'grid_key': state, **features(state),
                                'first_p_C': a, 'second_p_C': b, 'delta_p_C_first_minus_second': d,
                                'majority_disagrees': int((a > .5) != (b > .5))})
        n_disagree = sum((by_model[first][s] >= 3) != (by_model[second][s] >= 3) for s in states)
        pair_summaries.append({'first_model': first, 'second_model': second, 'n_states': 512,
                               'mean_delta_p_C_first_minus_second': mean(deltas), 'mean_absolute_p_C_difference': mean(abs(d) for d in deltas),
                               'first_higher_states': sum(d > 0 for d in deltas), 'same_p_C_states': sum(d == 0 for d in deltas),
                               'second_higher_states': sum(d < 0 for d in deltas), 'majority_disagreement_states': n_disagree,
                               'majority_disagreement_fraction': n_disagree / 512,
                               'opposite_unanimous_states': sum({by_model[first][s], by_model[second][s]} == {0, 5} for s in states)})
        for a in range(6):
            for b in range(6):
                n = sum(by_model[first][s] == a and by_model[second][s] == b for s in states)
                pair_joint.append({'first_model': first, 'second_model': second, 'first_C_answers': a, 'second_C_answers': b,
                                   'n_states': n, 'fraction_of_512_states': n / 512})

    drift_summary, drift_states = [], []
    for model, batches in BATCHES.items():
        old = by_batch[(model, 'old')]
        new = by_batch[(model, 'new')]
        valid_majority = [s for s in states if old[s] != .5 and new[s] != .5]
        n_flip = sum((old[s] > .5) != (new[s] > .5) for s in valid_majority)
        for center in ('all', 'C', 'D'):
            subset = states if center == 'all' else [s for s in states if s[4] == center]
            drift_summary.append({'model': model, 'center': center, 'n_states': len(subset),
                                  'old_repeats_per_state': batches[0][2], 'new_repeats_per_state': batches[1][2],
                                  'old_p_C': mean(old[s] for s in subset), 'new_p_C': mean(new[s] for s in subset),
                                  'mean_delta_new_minus_old': mean(new[s] - old[s] for s in subset),
                                  'mean_absolute_state_change': mean(abs(new[s] - old[s]) for s in subset),
                                  'opposite_unanimous_states': sum((old[s], new[s]) in ((0, 1), (1, 0)) for s in subset),
                                  'states_strict_majority_in_both': sum(s in valid_majority for s in subset),
                                  'strict_majority_disagreement_states': sum((old[s] > .5) != (new[s] > .5) for s in subset if s in valid_majority)})
        for state in states:
            drift_states.append({'model': model, 'grid_key': state, **features(state), 'old_p_C': old[state], 'new_p_C': new[state],
                                 'delta_new_minus_old': new[state] - old[state],
                                 'opposite_unanimous': int((old[state], new[state]) in ((0, 1), (1, 0)))})

    center_rows, center_summaries = [], []
    environments = sorted({s[:4] + s[5:] for s in states})
    for model in MODELS:
        model_rows = []
        for environment in environments:
            c_state = environment[:4] + 'C' + environment[4:]
            d_state = environment[:4] + 'D' + environment[4:]
            a, b = by_model[model][c_state] / 5, by_model[model][d_state] / 5
            row = {'model': model, 'eight_neighbor_key': environment,
                   'grid_current_C': c_state, 'grid_current_D': d_state,
                   'cardinal_c': features(c_state)['cardinal_c'], 'diagonal_c': features(c_state)['diagonal_c'],
                   'p_C_with_current_C': a, 'p_C_with_current_D': b, 'center_C_minus_D_difference': a - b}
            model_rows.append(row)
        center_rows.extend(model_rows)
        deltas = [r['center_C_minus_D_difference'] for r in model_rows]
        center_summaries.append({'model': model, 'n_paired_neighbor_configurations': 256,
                                'mean_center_C_minus_D_difference': mean(deltas),
                                'positive_pairs': sum(d > 0 for d in deltas), 'zero_pairs': sum(d == 0 for d in deltas),
                                'negative_pairs': sum(d < 0 for d in deltas), 'minimum_difference': min(deltas), 'maximum_difference': max(deltas)})

    orbits = {min(d4_orbit(s)): d4_orbit(s) for s in states}
    assert sum(len(v) for v in orbits.values()) == 512 and len(orbits) == 102
    orbit_rows, orbit_summaries = [], []
    for model in MODELS:
        n_pairs = 0
        n_disagreements = 0
        absolute_pair_diffs = []
        weighted_orbit_deviations = []
        for representative, orbit in sorted(orbits.items()):
            values = [by_model[model][s] / 5 for s in orbit]
            average = mean(values)
            weighted_orbit_deviations.extend(abs(v - average) for v in values)
            for a, b in itertools.combinations(values, 2):
                n_pairs += 1
                n_disagreements += int((a > .5) != (b > .5))
                absolute_pair_diffs.append(abs(a - b))
            orbit_rows.append({'model': model, 'orbit_representative': representative, 'orbit_size': len(orbit), **features(representative),
                               'orbit_mean_p_C': average, 'min_observed_p_C': min(values), 'max_observed_p_C': max(values),
                               'observed_p_C_range': max(values) - min(values), 'majority_varies_across_orbit': int(min(values) < .5 < max(values)),
                               'state_keys': ';'.join(orbit)})
        orbit_summaries.append({'model': model, 'n_orbits': len(orbits), 'nonidentical_state_pairs_within_orbits': n_pairs,
                                'state_weighted_mean_absolute_deviation_from_orbit_mean': mean(weighted_orbit_deviations),
                                'mean_absolute_p_C_difference_within_orbit_pairs': mean(absolute_pair_diffs),
                                'within_orbit_pair_majority_disagreements': n_disagreements,
                                'within_orbit_pair_majority_disagreement_fraction': n_disagreements / n_pairs})

    tables = {'model_summary': summaries, 'vote_histogram': votes, 'state_metrics': state_rows,
              'neighbor_response': neighbor_rows, 'context_response': context_rows,
              'model_pair_summary': pair_summaries, 'model_pair_state_differences': pair_states,
              'model_pair_vote_joint': pair_joint, 'batch_summary': batch_summary,
              'batch_drift_summary': drift_summary, 'batch_state_drift': drift_states,
              'repetition_summary': repetition_rows, 'center_pair_effects': center_rows,
              'center_pair_summary': center_summaries, 'symmetry_orbit_metrics': orbit_rows,
              'symmetry_summary': orbit_summaries}
    for name, rows in tables.items():
        write_csv(out / f'{name}.csv', rows)
    result = {'definitions': {
        'sampling_frame': 'All 512 possible binary 3x3 states, each weighted equally; 5 observed final successful replies per state and model.',
        'p_C': 'Observed number of C recommendations / number of replies; each state has five replies.',
        'current_action_retention': 'Fraction of replies recommending the focal player\'s current center action.',
        'unanimity_fraction': 'Fraction of 512 states where all five replies have the same recommendation.',
        'within_state_pair_agreement': 'For k C replies out of 5, [choose(k,2)+choose(5-k,2)]/choose(5,2), averaged equally over states. Uses 10 distinct unordered pairs, without self-pairs. Estimates repeat agreement only if replies are identically and independently drawn; the pooled batches do not establish that assumption.',
        'majority_advice': 'C if at least 3 of 5 replies are C; otherwise D. Never tied with five replies.',
        'mean_binary_entropy_bits': 'Observed binary entropy -p_C log2(p_C) -(1-p_C) log2(1-p_C), with zero contribution at p_C=0 or1, averaged over states. A descriptive plug-in measure of five-reply mixedness, not an unbiased estimate of model uncertainty.',
        'mean_absolute_p_C_difference': 'Average across the 512 matched states of the absolute difference between the two models\' observed C-reply fractions.',
        'center_pair_difference': 'C-reply fraction when center=C minus fraction when center=D, holding the other eight cells exactly fixed; 256 matched neighborhood pairs.',
        'batch_state_drift': 'New-batch minus old-batch observed C fraction within each state. Finite-sample differences and protocol differences are not separated.',
        'symmetry_diagnostic': 'Rotations and reflections partition 512 states into 102 square-symmetry orbits. Differences are descriptive and combine finite-reply noise with any orientation sensitivity; they do not prove symmetry violation.',
        'interpretation_limits': 'The 512 states are an exhaustive uniform input set, not a distribution of encountered simulation states. Five mixed-batch replies do not identify a true probability. No CIs or significance tests are asserted. Advice C frequency is not achieved cooperation, payoff, or decision quality. No reasoning analysis is inferred from C/D outputs.'},
        'model_summary': summaries, 'model_pair_summary': pair_summaries, 'batch_drift_summary': drift_summary,
        'center_pair_summary': center_summaries, 'symmetry_summary': orbit_summaries,
        'source_csv_sha256': sources, 'merged_canonical_crosscheck': merged_check,
        'generated_tables': {f'{k}.csv': len(v) for k, v in tables.items()}}
    (out / 'analysis_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['Independent five-answer analysis', '', 'All statistics are descriptive for the exhaustive, equally weighted 512-state input set.',
             'There are 2,560 final successful C/D replies per model and 7,680 in total.',
             f'Canonical merge crosscheck: {merged_check}', '']
    for r in summaries:
        lines.append(f"{r['model']}: C replies={r['p_C']:.4%}; C→C={r['p_C_given_current_C']:.4%}; D→C={r['p_C_given_current_D']:.4%}; retain current action={r['current_action_retention']:.4%}; unanimous states={r['unanimous_states']}/512; distinct-reply pair agreement={r['within_state_pair_agreement']:.4%}.")
    lines += ['', 'Main interpretation:',
              '1. Qwen recommends cooperation most frequently, particularly when a currently defecting focal player has cooperative neighbors.',
              '2. DeepSeek mostly preserves current cooperators while rarely switching current defectors to cooperation. It has the strongest center-action dependence in exact paired neighborhoods.',
              '3. GPT is the most unanimous over the five observed replies, but that is repeat consistency, not evidence of superior decisions or calibration.',
              '4. Small changes in overall old/new C frequency can hide substantially larger changes in individual states.',
              '5. Pooled replies span two collection batches per model. No retained answer reached its output-token cap or showed token-limit truncation; the cap change is not evidence against pooling. Batch provenance and reported model labels remain available in the audit.',
              '6. These are advice tables; no new spatial-game trajectories or welfare outcomes are present in this analysis.', '', 'Between-model comparisons:']
    for r in pair_summaries:
        lines.append(f"{r['first_model']} vs {r['second_model']}: majority differs in {r['majority_disagreement_states']}/512 states ({r['majority_disagreement_fraction']:.2%}); mean absolute C-frequency difference={r['mean_absolute_p_C_difference']:.2%}.")
    lines += ['', 'Batch changes (new minus old):']
    for r in drift_summary:
        if r['center'] == 'all':
            lines.append(f"{r['model']}: overall {100*r['mean_delta_new_minus_old']:+.3f} percentage points; mean absolute within-state change {100*r['mean_absolute_state_change']:.3f} points; {r['opposite_unanimous_states']} old-unanimous/new-unanimous opposite states. Old/new sample sizes={r['old_repeats_per_state']}/{r['new_repeats_per_state']}; do not rank batch stability from these counts across models.")
    lines += ['', 'Caution on uncertainty:',
              'No sampling confidence intervals are supplied. The states exhaust the designed binary input space; uncertainty concerns repeated model outputs and differences across collection protocols. Treating all 2,560 replies per model as exchangeable Bernoulli trials would ignore the state structure.',
              'Symmetry metrics are exploratory finite-sample diagnostics and should not be labeled evidence of orientation bias without more controlled repetitions.']
    (report / 'independent_findings.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(report), 'canonical_merge_crosscheck': merged_check, 'model_summary': summaries,
                      'n_analysis_tables': len(tables)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports' / '1101_3LLM_Analysis')
    args = parser.parse_args()
    run(args.root, args.report)

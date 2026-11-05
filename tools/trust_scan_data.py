"""Audit and cache the saved adaptive-trust scan; never runs a simulation.

Run: /tmp/llm_advice_viz_venv/bin/python tools/trust_scan_data.py --rebuild
All means exclude round 0. Early: 1..200; tail: 9001..10000.
The independent replicate is a saved run/seed, not a time step or grid cell.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
SOURCE = ROOT / 'main/SPD/results/scan_20261029_183749_initial'
FIXED_SOURCE = ROOT / 'main/SPD/results/scan_20261027_121246_2次gpt微扫描'
ADVICE_SOURCE = ROOT / 'logs/GPT_Multistepsa/20261016_173605_GPT5.5_2次'
DATA = ROOT / 'reports/Trust_1030/data'
KEYS = ['b', 'llm_access_fraction', 'seed']
POINT_FIELDS = ['eligible_trust_mean', 'eligible_trust_std', 'eligible_trust_min',
                'eligible_trust_max', 'eligible_low_trust_fraction',
                'eligible_high_trust_fraction', 'eligible_aspiration_mean',
                'eligible_feedback_payoff_mean'] + [f'eligible_trust_bin_{i}' for i in range(10)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mean(values):
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / 'scan_config.json').read_text())
    old_config = json.loads((FIXED_SOURCE / 'scan_config.json').read_text())
    runs = pd.read_csv(SOURCE / 'summary.csv').sort_values(KEYS).reset_index(drop=True)
    fixed = pd.read_csv(FIXED_SOURCE / 'summary.csv').query('lambda_prob == 0.5').sort_values(KEYS).reset_index(drop=True)
    expected = list(itertools.product(config['scan_values']['b'], config['scan_values']['llm_access_fraction'], [42, 43]))
    assert list(runs[KEYS].itertuples(index=False, name=None)) == expected
    assert list(fixed[KEYS].itertuples(index=False, name=None)) == expected
    assert len(runs) == 154 and len(list(SOURCE.glob('run_*/history.csv'))) == 154
    assert runs.status.eq('completed').all() and config['completed_runs'] == 154
    assert runs.steps.eq(10000).all() and runs.advice_mode.eq('adaptive').all()
    assert runs.lambda_prob.isna().all()
    assert runs.trust_payoff_window.eq(3).all()
    assert runs.initial_trust.eq(.5).all()
    assert runs.trust_learning_rate.eq(.1).all() and runs.aspiration_learning_rate.eq(.1).all()
    assert runs.early_average_steps.eq(200).all() and runs.tail_average_steps.eq(1000).all()
    assert runs.paired_fermi_control.all()

    hashes = {}
    def record(path):
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    for path in [SOURCE / 'summary.csv', SOURCE / 'scan_config.json', FIXED_SOURCE / 'summary.csv', FIXED_SOURCE / 'scan_config.json']:
        record(path)
    advice_cfg = json.loads((ADVICE_SOURCE / 'config.json').read_text())
    prompt_hash = hashlib.sha256(advice_cfg['system_prompt'].encode()).hexdigest()
    for c in [config, old_config]:
        for source in c['advisor_sources']:
            assert source['batch_id'] == advice_cfg['batch_id']
            assert source['system_prompt_sha256'] == prompt_hash
            assert source['n'] == 2 and source['case_count'] == 512
            for name in ['config.json', 'results.csv']:
                assert source['source_files'][name]['sha256'] == sha256(ADVICE_SOURCE / name)
    for name in ['config.json', 'results.csv']:
        record(ADVICE_SOURCE / name)

    errors = []
    checks = 0
    largest_abs_error = 0.0
    def check(label, saved, calculated, tol=1e-8):
        nonlocal checks, largest_abs_error
        checks += 1
        a, b = np.asarray(saved, dtype=float), np.asarray(calculated, dtype=float)
        if not np.allclose(a, b, rtol=1e-10, atol=tol, equal_nan=True):
            errors.append(label)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            largest_abs_error = max(largest_abs_error, float(np.max(np.abs(a[both] - b[both]))))

    trajectories = {}
    grids = np.empty((154, 50, 50), dtype=np.uint8)
    control_refs = {}
    fixed_records = []
    calculated_records = []
    old_control_equal = []
    old_p0_equal = []
    for i, row in runs.iterrows():
        run_path = SOURCE / row.output_dir
        history_path = run_path / 'history.csv'
        record(history_path)
        h = pd.read_csv(history_path)
        assert h.shape == (10001, 51), (run_path, h.shape)
        assert np.array_equal(h['round'].values, np.arange(10001))
        if not trajectories:
            trajectories = {name: np.empty((154, 10001), dtype=np.float64) for name in h.columns}
            for name in ['cooperation_rate', 'average_payoff', 'control_cooperation_rate', 'control_average_payoff']:
                trajectories['fixed_' + name] = np.empty((154, 10001), dtype=np.float64)
        for name in h.columns:
            trajectories[name][i] = h[name].to_numpy(dtype=float)
        arr = {name: h[name].to_numpy(dtype=float) for name in h.columns}
        calc = {}
        def summary(name, value):
            calc[name] = value
            if name in row.index:
                check(f'{row.run_id}:{name}', row[name], value)
        summary('initial_cooperation_rate', arr['cooperation_rate'][0])
        summary('initial_eligible_trust_mean', arr['eligible_trust_mean'][0])
        for prefix in ['', 'control_']:
            c = arr[prefix + 'cooperation_rate']
            w = arr[prefix + 'average_payoff']
            summary(prefix + 'final_cooperation_rate', c[-1])
            summary(prefix + 'final_average_payoff', w[-1])
            summary(prefix + 'tail_mean_cooperation_rate', mean(c[9001:]))
            summary(prefix + 'time_mean_cooperation_rate', mean(c[1:]))
            summary(prefix + 'tail_mean_payoff', mean(w[9001:]))
            summary(prefix + 'first_extinction_round', float(np.flatnonzero(c == 0)[0]) if np.any(c == 0) else np.nan)
            calc[prefix + 'first_all_cooperation_round'] = float(np.flatnonzero(c == 1)[0]) if np.any(c == 1) else np.nan
            calc[prefix + 'late_previous_mean_cooperation_rate'] = mean(c[8001:9001])
            calc[prefix + 'tail_cooperation_drift'] = mean(c[9001:]) - mean(c[8001:9001])
            calc[prefix + 'tail_cooperation_within_run_std'] = float(np.std(c[9001:]))
            for group in ['population', 'eligible', 'ineligible']:
                g = prefix + group
                a = arr[g + '_mean_payoff']
                count = row[g + '_count']
                summary(g + '_time_mean_payoff', mean(a[1:]))
                summary(g + '_early_mean_payoff', mean(a[1:201]))
                summary(g + '_tail_mean_payoff', mean(a[9001:]))
                summary(g + '_cumulative_mean_payoff', float(np.sum(a[1:])) if count else np.nan)
                summary(g + '_cumulative_total_payoff', float(np.sum(a[1:]) * count) if count else 0.0)
                if count:
                    check(f'{row.run_id}:{g}:cumulative_history', arr[g + '_cumulative_mean_payoff'][1:], np.cumsum(a[1:]), 1e-7)
        for name in ['advice_requests', 'advice_used', 'fermi_updates', 'strategy_changes', 'trust_positive_feedback', 'trust_negative_feedback', 'trust_zero_feedback']:
            summary('total_' + name, float(np.sum(arr[name][1:])))
        for name in POINT_FIELDS:
            summary('final_' + name, arr[name][-1])
            calc['tail_mean_' + name] = mean(arr[name][9001:])
        for col, suffix in [('eligible_trust_mean', 'eligible_trust'), ('eligible_advice_rate', 'eligible_advice_rate')]:
            for prefix, sl in [('time', slice(1, None)), ('early', slice(1, 201)), ('tail', slice(9001, None))]:
                summary(prefix + '_mean_' + suffix, mean(arr[col][sl]))
        for col in ['trust_positive_feedback', 'trust_negative_feedback', 'trust_zero_feedback']:
            calc['tail_mean_' + col] = mean(arr[col][9001:])
        calc['tail_mean_population_advice_rate'] = mean(arr['advice_used'][9001:]) / row.population_count
        calc['time_mean_population_advice_rate'] = mean(arr['advice_used'][1:]) / row.population_count
        calc['tail_trust_drift'] = mean(arr['eligible_trust_mean'][9001:]) - mean(arr['eligible_trust_mean'][8001:9001])
        calc['tail_mean_adoption_minus_previous_trust'] = mean(arr['eligible_advice_rate'][9001:] - arr['eligible_trust_mean'][9000:10000])
        feedback_total = sum(calc['tail_mean_' + name] for name in ['trust_positive_feedback', 'trust_negative_feedback', 'trust_zero_feedback'])
        for label in ['positive', 'negative', 'zero']:
            calc['tail_' + label + '_feedback_fraction'] = calc['tail_mean_trust_' + label + '_feedback'] / feedback_total if feedback_total else np.nan
        # Saved delta fields must be the within-seed experimental-minus-control difference.
        for name in h.columns:
            if name.startswith('delta_'):
                base = name[6:]
                check(f'{row.run_id}:{name}', arr[name], arr[base] - arr['control_' + base])
        for name in row.index:
            if name.startswith('delta_'):
                base = name[6:]
                if base in row.index and 'control_' + base in row.index:
                    check(f'{row.run_id}:summary:{name}', row[name], row[base] - row['control_' + base])
        check(f'{row.run_id}:requests=used', arr['advice_requests'], arr['advice_used'])
        check(f'{row.run_id}:decision_count', arr['advice_used'][1:] + arr['fermi_updates'][1:], np.full(10000, 2500))
        check(f'{row.run_id}:feedback_count', arr['trust_positive_feedback'] + arr['trust_negative_feedback'] + arr['trust_zero_feedback'], arr['advice_used'])
        if row.eligible_count:
            check(f'{row.run_id}:adoption', arr['eligible_advice_rate'][1:], arr['advice_used'][1:] / row.eligible_count)
            check(f'{row.run_id}:histogram', h[[f'eligible_trust_bin_{j}' for j in range(10)]].sum(axis=1), np.ones(10001))
            assert np.isfinite(h.loc[1:, POINT_FIELDS].values).all()
            assert np.isfinite(h.loc[0, [f for f in POINT_FIELDS if f != 'eligible_feedback_payoff_mean']].values.astype(float)).all()
            assert np.isnan(h.loc[0, 'eligible_feedback_payoff_mean'])
            assert ((arr['eligible_trust_mean'] >= 0) & (arr['eligible_trust_mean'] <= 1)).all()
        else:
            assert h[POINT_FIELDS + ['eligible_advice_rate']].isna().all().all()
            check(f'{row.run_id}:P0_control_C', arr['cooperation_rate'], arr['control_cooperation_rate'], 0)
            check(f'{row.run_id}:P0_control_payoff', arr['average_payoff'], arr['control_average_payoff'], 0)
        assert np.isnan(arr['eligible_advice_rate'][0])
        if row.ineligible_count == 0:
            assert h['ineligible_mean_payoff'].isna().all()
        grid_path = run_path / 'final_grid.csv'
        record(grid_path)
        grid = pd.read_csv(grid_path, header=None).values
        assert grid.shape == (50, 50) and set(np.unique(grid)).issubset({'C', 'D'})
        grids[i] = (grid == 'C').astype(np.uint8)
        check(f'{row.run_id}:final_grid', row.final_cooperation_rate, grids[i].mean())
        control_key = (row.b, row.seed)
        cols = ['control_cooperation_rate', 'control_average_payoff', 'control_population_mean_payoff']
        values = h[cols].values
        if control_key not in control_refs:
            control_refs[control_key] = values
        else:
            assert np.array_equal(values, control_refs[control_key])
            check(f'{row.run_id}:identical_control_across_P', values, control_refs[control_key], 0)
        # Check original fixed-lambda histories, rather than relying on a prior report cache.
        fr = fixed.iloc[i]
        fh_path = FIXED_SOURCE / fr.output_dir / 'history.csv'
        record(fh_path)
        fh = pd.read_csv(fh_path)
        assert fh.shape[0] == 10001 and np.array_equal(fh['round'], h['round'])
        for name in ['cooperation_rate', 'average_payoff', 'control_cooperation_rate', 'control_average_payoff']:
            trajectories['fixed_' + name][i] = fh[name].values
        for name, hist_name in [('tail_mean_cooperation_rate', 'cooperation_rate'), ('tail_mean_payoff', 'average_payoff'), ('population_time_mean_payoff', 'population_mean_payoff')]:
            values = fh[hist_name].values
            check(f'fixed:{fr.run_id}:{name}', fr[name], mean(values[1:] if name.startswith('population') else values[9001:]))
        control_equal = np.array_equal(fh['control_cooperation_rate'].values, arr['control_cooperation_rate']) and np.array_equal(fh['control_average_payoff'].values, arr['control_average_payoff'])
        old_control_equal.append(bool(control_equal))
        if row.llm_access_fraction == 0:
            old_p0_equal.append(bool(np.array_equal(fh['cooperation_rate'].values, arr['cooperation_rate'])))
        fixed_records.append({'b': row.b, 'llm_access_fraction': row.llm_access_fraction, 'seed': row.seed,
                              'fixed_tail_cooperation_drift': mean(fh.cooperation_rate.values[9001:]) - mean(fh.cooperation_rate.values[8001:9001]),
                              'fixed_first_all_cooperation_round': float(np.flatnonzero(fh.cooperation_rate.values == 1)[0]) if (fh.cooperation_rate.values == 1).any() else np.nan,
                              'fixed_time_mean_population_advice_rate': float(fh.advice_used.values[1:].mean() / 2500),
                              'fixed_tail_mean_eligible_advice_rate': float(fh.advice_used.values[9001:].mean() / row.eligible_count) if row.eligible_count else np.nan})
        calculated_records.append(calc)
        if (i + 1) % 22 == 0:
            print(f'Audited {i + 1}/154 adaptive histories and matched fixed histories', flush=True)
    assert not errors, f'Validation failures: {errors[:20]}'
    assert all(old_control_equal) and all(old_p0_equal)
    computed = pd.DataFrame(calculated_records)
    extra_columns = [c for c in computed if c not in runs]
    validated = pd.concat([runs, computed[extra_columns]], axis=1)
    validated.to_csv(DATA / 'validated_runs.csv', index=False)
    numeric = [c for c in validated.select_dtypes(include=['number', 'bool']).columns if c not in KEYS]
    parameters = validated.groupby(['b', 'llm_access_fraction'])[numeric].agg(['mean', 'min', 'max'])
    parameters.columns = ['_'.join(c) for c in parameters.columns]
    parameters = parameters.copy()
    parameters['n_runs'] = 2
    parameters.reset_index().to_csv(DATA / 'parameter_summary.csv', index=False)
    matched = validated.rename(columns={c: 'adaptive_' + c for c in validated if c not in KEYS}).merge(
        fixed.rename(columns={c: 'fixed_' + c for c in fixed if c not in KEYS}), on=KEYS, validate='one_to_one')
    matched = matched.merge(pd.DataFrame(fixed_records), on=KEYS, validate='one_to_one')
    matched = matched.copy()
    for field in ['tail_mean_cooperation_rate', 'time_mean_cooperation_rate', 'population_time_mean_payoff', 'tail_mean_payoff', 'final_cooperation_rate']:
        matched['adaptive_minus_fixed_' + field] = matched['adaptive_' + field] - matched['fixed_' + field]
    matched.to_csv(DATA / 'matched_fixed_gpt.csv', index=False)
    np.savez_compressed(DATA / 'trajectory_cache.npz', **trajectories, final_grids=grids)
    metadata = {
        'schema_version': 1, 'source': str(SOURCE.relative_to(ROOT)), 'fixed_source': str(FIXED_SOURCE.relative_to(ROOT)),
        'key_columns': KEYS, 'ordered_keys': [list(k) for k in expected], 'shape': [154, 10001],
        'history_columns': list(h.columns), 'fixed_history_columns': ['cooperation_rate', 'average_payoff', 'control_cooperation_rate', 'control_average_payoff'],
        'cache_semantics': 'Native history fields, run-major float64 arrays; fixed_ fields matched by b/P/seed; final_grids is uint8 1=C,0=D.',
        'windows': {'early': [1, 200], 'tail': [9001, 10000], 'late_previous': [8001, 9000], 'time': [1, 10000]},
        'artifact_sha256': {name: sha256(DATA / name) for name in ['validated_runs.csv', 'parameter_summary.csv', 'matched_fixed_gpt.csv', 'trajectory_cache.npz']}}
    json_write(DATA / 'cache_metadata.json', metadata)
    audit = {
        'status': 'passed', 'runs': 154, 'parameter_sets': 77, 'replicates_per_setting': 2, 'steps_per_run': 10000,
        'history_columns': 51, 'checks': checks, 'largest_abs_numeric_recalculation_error': largest_abs_error,
        'validation_tolerance': 'np.allclose rtol=1e-10, atol=1e-8 (cumulative histories atol=1e-7)',
        'final_grid_cooperation_matches': 154, 'summary_recomputed_runs': 154, 'old_fixed_histories_recomputed': 154,
        'unique_control_trajectories_b_seed': len(control_refs), 'controls_equal_across_P': True,
        'new_vs_old_control_trajectories_identical': all(old_control_equal), 'new_vs_old_P0_trajectories_identical': all(old_p0_equal),
        'advice_batch_id': advice_cfg['batch_id'], 'advice_model_logged': advice_cfg['model'], 'advice_n': 2,
        'advice_case_count': 512, 'advice_source_hashes_match_both_scan_configs': True,
        'advice_local_directory': str(ADVICE_SOURCE.relative_to(ROOT)), 'system_prompt_sha256': prompt_hash,
        'windows': metadata['windows'], 'initial_grid_and_eligibility_mask_saved': False,
        'comparison_caveats': ['Cross-batch matched descriptive comparison, not a dedicated rerun of both modes under a pinned identical source revision.',
                               'Advice table hashes, nominal setup and complete Fermi control trajectories agree.',
                               'Initial grids and eligibility masks are not archived; equality checks are documented in the scan metadata but cannot be independently replayed from saved masks.',
                               'Adaptive update can consume random draws differently; seed matching does not imply identical decision-level random events.',
                               'Two independent simulation seeds per parameter set; no confidence intervals or significance claims.',
                               'Trust histograms are aggregate saved fractions; no individual trust grids were saved.'],
        'source_sha256': hashes,
    }
    json_write(DATA / 'audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['status', 'runs', 'checks', 'largest_abs_numeric_recalculation_error', 'new_vs_old_control_trajectories_identical', 'advice_source_hashes_match_both_scan_configs']}, indent=2))
    return validated, parameters.reset_index(), matched, metadata, audit


def load_scan(rebuild=False):
    if rebuild or not (DATA / 'cache_metadata.json').exists():
        build()
    metadata = json.loads((DATA / 'cache_metadata.json').read_text())
    for name, digest in metadata['artifact_sha256'].items():
        assert sha256(DATA / name) == digest, f'Changed cached artifact: {name}'
    return (pd.read_csv(DATA / 'validated_runs.csv'), pd.read_csv(DATA / 'parameter_summary.csv'),
            pd.read_csv(DATA / 'matched_fixed_gpt.csv'), np.load(DATA / 'trajectory_cache.npz'),
            metadata, json.loads((DATA / 'audit.json').read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    if args.rebuild or not (DATA / 'audit.json').exists():
        build()
    else:
        runs, _, _, _, _, audit = load_scan()
        print(f"Validated cache: {len(runs)} runs; audit={audit['status']}")

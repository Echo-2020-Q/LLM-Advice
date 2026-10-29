"""Read-only raw audits and alignment of the October 27 GPT and Qwen scans."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from io import BytesIO
from itertools import product
import json
from pathlib import Path, PureWindowsPath
import numpy as np
import pandas as pd
import qwen_scan_data as qdata

ROOT = Path(__file__).resolve().parents[1]
WRITE_ROOT = Path(str(ROOT).lower())
if not (WRITE_ROOT.is_dir() and WRITE_ROOT.samefile(ROOT)):
    WRITE_ROOT = ROOT
DEFAULT_REPORT = WRITE_ROOT / 'reports/gpt_20261027_repeats2'
GPT_BATCH = ROOT / 'main/SPD/results/scan_20261027_121246_2次gpt微扫描'
QWEN_BATCH = ROOT / 'main/SPD/results/scan_20261027_122530_2次Qwen微扫描'
QWEN_REPORT = WRITE_ROOT / 'reports/qwen_20261027_repeats2'
KEYS = qdata.KEY_COLUMNS


def _load_gpt_advisor(config, hashes, checks):
    sources = config['advisor_sources']
    checks.require(bool(sources), 'advisor_sources_present')
    source = sources[0]
    fingerprint = {n: v['sha256'] for n, v in source['source_files'].items()}
    checks.require(all({n: v['sha256'] for n, v in s['source_files'].items()} == fingerprint for s in sources), 'single_advisor_source_fingerprint')
    checks.require(all(s['model'] == source['model'] and s['batch_id'] == source['batch_id'] for s in sources), 'single_advisor_model_and_batch')
    directory = ROOT / 'logs/GPT_Multistepsa' / PureWindowsPath(config['advisor_batch']).name
    raw = {}
    for name, expected in fingerprint.items():
        raw[name] = qdata._read(directory / name, hashes)
        checks.require(qdata._sha(raw[name]) == expected, 'advisor_recorded_sha256', name)
    cfg = json.loads(raw['config.json'])
    summary = json.loads(qdata._read(directory / 'summary.json', hashes))
    advisor = pd.read_csv(BytesIO(raw['results.csv']))
    checks.require(qdata._sha(cfg['system_prompt'].encode()) == source['system_prompt_sha256'], 'advisor_prompt_sha256')
    checks.require(cfg['model'] == source['model'], 'advisor_model_config_matches_scan')
    if 'model' in summary:
        checks.require(summary['model'] == cfg['model'], 'advisor_model_summary_matches_config')
    checks.require(cfg['batch_id'] == summary['batch_id'] == source['batch_id'], 'advisor_batch_id_metadata')
    checks.require(summary['status'] == 'completed' and summary['successful_requests'] == 1024 and summary['failed_requests'] == 0, 'advisor_batch_completed')
    checks.require(len(advisor) == 1024 and (advisor.status == 'success').all(), 'advisor_1024_successful_rows')
    checks.require(set(advisor.grid_key) == set(map(''.join, product('CD', repeat=9))), 'advisor_all_512_states')
    checks.require((advisor.groupby('grid_key').size() == 2).all(), 'advisor_two_answers_per_state')
    checks.require(set(advisor.recommendation) <= {'C', 'D'}, 'advisor_valid_actions')
    advisor['center'] = advisor.grid_key.str[4]
    advisor['cardinal_c'] = advisor.grid_key.map(lambda k: sum(k[i] == 'C' for i in (1, 3, 5, 7)))
    advisor['is_c'] = advisor.recommendation.eq('C')
    checks.require((advisor.center == advisor.current_action).all() and (advisor.cardinal_c == advisor.cooperating_neighbors).all(), 'advisor_grid_metadata')
    checks.require(cfg['b'] is None and cfg['b_constraint'] == '1 < b < 2', 'advisor_unknown_b_shared_policy')
    counts = advisor.groupby('grid_key').is_c.sum()
    metadata = {
        'directory': qdata._source_name(directory), 'model': source['model'], 'batch_id': source['batch_id'],
        'model_identity_basis': 'local config and recorded scan metadata; not independent service-model verification',
        'source_summary_has_model': 'model' in summary,
        'metadata_adapter_sha256': qdata._sha(Path(__file__).read_bytes()),
        'states': 512, 'answers_per_state': 2, 'successful_answers': 1024,
        'cooperate_answers': int(advisor.is_c.sum()), 'uniform_state_cooperation_probability': float(advisor.is_c.mean()),
        'agreement_states': int(counts.isin([0, 2]).sum()), 'agreement_fraction': float(counts.isin([0, 2]).mean()),
        'answer_pairs': {'CC': int((counts == 2).sum()), 'CD': int((counts == 1).sum()), 'DD': int((counts == 0).sum())},
        'all_C_answers': advisor.loc[advisor.grid_key == 'C' * 9, 'recommendation'].tolist(),
        'all_D_answers': advisor.loc[advisor.grid_key == 'D' * 9, 'recommendation'].tolist(),
        'conditioning': 'same table for every b; b unknown except 1 < b < 2; numerical K/P/lambda absent',
        'sampling': source['sampling'],
        'probability_caveat': 'Uniformly weighted enumerated states, not visited-state weighting in simulation.',
    }
    return advisor, metadata


def _load_gpt(report, use_cache):
    audit_file = report / '数据核验.json'
    if use_cache and audit_file.is_file():
        previous = json.loads(audit_file.read_text(encoding='utf-8'))
        if previous.get('advisor', {}).get('metadata_adapter_sha256') != qdata._sha(Path(__file__).read_bytes()):
            print('GPT adapter changed; performing full audit.', flush=True)
            use_cache = False
    original = qdata._load_advisor
    try:
        qdata._load_advisor = _load_gpt_advisor
        return qdata.load_scan(GPT_BATCH, report, use_cache=use_cache)
    finally:
        qdata._load_advisor = original


def _advice_generation_comparison(gc, qc):
    """Read only whitelisted request fields; never copy connection settings."""
    configs, records, settings, hashes = {}, {}, {}, {}
    direct_fields = ['temperature', 'top_p', 'seed', 'max_tokens', 'max_output_tokens',
                     'reasoning_effort', 'reasoning']
    extra_fields = ['reasoning_effort', 'reasoning_format', 'cache_prompt']
    config_fields = direct_fields + ['qwen_reasoning_effort', 'n', 'observation_format',
                                    'b_mode', 'b_constraint', 'api_interface']
    for label, scan in [('gpt', gc), ('qwen', qc)]:
        directory = ROOT / 'logs/GPT_Multistepsa' / PureWindowsPath(scan['advisor_batch']).name
        cfg = configs[label] = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        table = pd.read_csv(directory / 'results.csv')
        records[label], options = {}, {}
        for row in table.to_dict('records'):
            path = (directory / row['record_file']).resolve()
            qdata._require(path.is_relative_to(directory), 'advice request path within source directory')
            record = json.loads(qdata._read(path, hashes))
            req = record['request']
            qdata._require(record['status'] == 'success' and record['action'] == row['recommendation'], 'source request response agrees with advice table')
            qdata._require(req['model'] == cfg['model'], 'all source requests match recorded model')
            key = (record['observation']['grid_key'], int(record['repetition']))
            qdata._require(key == (row['grid_key'], int(row['repetition'])), 'source request grid and repeat agree with table')
            messages = req.get('messages', req.get('input'))
            qdata._require(isinstance(messages, list) and len(messages) == 2 and
                           [m['role'] for m in messages] == ['system', 'user'], 'two-role system/user request format')
            normalized = [(m['role'], m['content']) for m in messages]
            qdata._require(normalized[0][1] == cfg['system_prompt'], 'all request system prompts match config')
            qdata._require(key not in records[label], 'unique advice request grid/repeat')
            records[label][key] = normalized
            selected = {f: {'recorded': f in req, 'value': req.get(f)} for f in direct_fields}
            extra = req.get('extra_body', {})
            selected['extra_body'] = {f: {'recorded': f in extra, 'value': extra.get(f)} for f in extra_fields}
            token = json.dumps(selected, ensure_ascii=False, sort_keys=True)
            options[token] = options.get(token, 0) + 1
        settings[label] = [{'settings': json.loads(k), 'requests': v} for k, v in options.items()]
    qdata._require(records['gpt'].keys() == records['qwen'].keys(), 'matching advice input grid/repeat keys')
    same = sum(records['gpt'][k] == records['qwen'][k] for k in records['gpt'])
    qdata._require(same == 1024, 'all paired GPT/Qwen system and user messages are exactly equal')
    visible = {label: {f: {'recorded': f in cfg, 'value': cfg.get(f)} for f in config_fields}
               for label, cfg in configs.items()}
    return {'visible_config_settings': visible, 'visible_request_settings': settings,
            'identical_system_prompt': configs['gpt']['system_prompt'] == configs['qwen']['system_prompt'],
            'identical_enumerated_cases': configs['gpt']['cases'] == configs['qwen']['cases'],
            'identical_normalized_system_user_message_pairs': same,
            'normalization': 'GPT request.input and Qwen request.messages compared as role/content pairs for each grid and repetition.',
            'request_records_checked': len(hashes), 'request_source_sha256': dict(sorted(hashes.items())),
            'generation_settings_limit': 'Temperature, top_p and generation seed are absent from both recorded requests; effective backend defaults cannot be assumed equal. Qwen explicitly requests reasoning_effort=medium; GPT does not record a reasoning setting. max_output_tokens/max_tokens are both 10240 but API meanings may differ.'}


def _compare(gpt, qwen, report):
    gc, gd, gt, gg, ga, gu = gpt
    qc, qd, qt, qg, qa, qu = qwen
    checks = qdata.Checks()
    config_fields = ['scan_values', 'repeats', 'parameter_sets', 'seed_values', 'seed_policy', 'fixed_parameters', 'steps',
                     'tail_average_steps', 'early_average_steps', 'paired_fermi_control', 'control_lambda_prob',
                     'statistics_semantics', 'summary_semantics', 'output_dir_semantics']
    for field in config_fields:
        checks.require(gc[field] == qc[field], 'matching_scan_design_and_semantics', field)
    source_fields = ['b', 'n', 'sampling', 'observation_format', 'case_count', 'matching_fields', 'grid_layout',
                     'interaction_neighborhood', 'system_prompt_sha256']
    for field in source_fields:
        checks.require(gc['advisor_sources'][0][field] == qc['advisor_sources'][0][field], 'matching_advice_protocol', field)
    gi, qi = (d.set_index(KEYS).sort_index() for d in (gd, qd))
    checks.require(gi.index.equals(qi.index), 'exact_parameter_seed_alignment')
    controls = sorted(c for c in gi.columns if c.startswith('control_'))
    checks.require(controls == sorted(c for c in qi.columns if c.startswith('control_')), 'same_control_summary_columns')
    for col in controls:
        same = gi[col].eq(qi[col]) | (gi[col].isna() & qi[col].isna())
        checks.require(same.all(), 'exact_control_summary_equality', col)
    checks.require(gu['control_population_hashes'] == qu['control_population_hashes'], 'exact_control_population_history_sha256')
    for key in gt:
        checks.require(gt[key][0] == qt[key][0], 'same_initial_cooperation_rate', str(key))
        if key[1] == 0 or key[2] == 0:
            checks.require(np.array_equal(gt[key], qt[key]), 'exact_no_advice_cooperation_trajectory', str(key))
            checks.require(np.array_equal(gg[key], qg[key]), 'exact_no_advice_final_grid', str(key))
    joined = gi.add_prefix('gpt_').join(qi.add_prefix('qwen_'), validate='one_to_one').copy()
    metrics = ['tail_mean_cooperation_rate', 'final_cooperation_rate', 'time_mean_cooperation_rate',
               'cooperation_4001_5000_mean', 'cooperation_tail_drift', 'population_time_mean_payoff',
               'eligible_time_mean_payoff', 'ineligible_time_mean_payoff', 'population_tail_mean_payoff',
               'eligible_tail_mean_payoff', 'ineligible_tail_mean_payoff']
    joined = pd.concat([joined, pd.DataFrame({'gpt_minus_qwen_' + m: gi[m] - qi[m] for m in metrics})], axis=1).reset_index()
    state = pd.concat([ga.groupby('grid_key').is_c.mean().rename('gpt_probability_c'),
                       qa.groupby('grid_key').is_c.mean().rename('qwen_probability_c')], axis=1).reset_index()
    state['center'] = state.grid_key.str[4]
    state['cardinal_c'] = state.grid_key.map(lambda k: sum(k[i] == 'C' for i in (1, 3, 5, 7)))
    state['gpt_minus_qwen_probability_c'] = state.gpt_probability_c - state.qwen_probability_c
    cond = state.groupby(['center', 'cardinal_c'])[['gpt_probability_c', 'qwen_probability_c']].mean().reset_index()
    cond['states'] = state.groupby(['center', 'cardinal_c']).size().to_numpy()
    cond['answers_per_model'] = cond.states * 2
    cond['gpt_minus_qwen_probability_c'] = cond.gpt_probability_c - cond.qwen_probability_c
    events = {}
    for label, frame in [('gpt', gd), ('qwen', qd)]:
        active = frame[(frame.lambda_prob > 0) & (frame.llm_access_fraction > 0)]
        events[label] = {'active_runs': len(active), 'active_final_all_D': int(active.extinct_at_end.sum()),
                         'active_extinction_observed': int(active.extinction_observed.sum()),
                         'active_revivals': int(active.cooperation_revivals.sum()),
                         'all_runs_revivals': int(frame.cooperation_revivals.sum()),
                         'active_final_all_C': int(active.final_cooperation_rate.eq(1).sum()),
                         'active_tail_mean_all_C': int(active.tail_mean_cooperation_rate.eq(1).sum())}
    delta = state.gpt_minus_qwen_probability_c
    audit = {
        'validation': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(), 'comparisons': checks.data,
        'matched_runs': len(joined), 'matched_parameter_sets': int(joined[KEYS[:-1]].drop_duplicates().shape[0]),
        'control_summary_columns': controls, 'identical_control_summary_values': len(joined) * len(controls),
        'control_population_history_columns': ['control_cooperation_rate', 'control_average_payoff', 'control_population_mean_payoff', 'control_population_cumulative_mean_payoff'],
        'unique_control_b_seed_pairs': len(gu['control_population_hashes']),
        'identical_config_fields': config_fields, 'identical_advice_protocol_fields': source_fields,
        'system_prompt_sha256': gc['advisor_sources'][0]['system_prompt_sha256'],
        'model_identity': {'gpt': gu['advisor']['model'], 'qwen': qu['advisor']['model']},
        'model_identity_basis': 'Recorded local metadata, not independent verification of a hosted model or file contents.',
        'advisor_state_probabilities': {'states': len(state), 'gpt_higher': int(delta.gt(0).sum()), 'equal': int(delta.eq(0).sum()), 'qwen_higher': int(delta.lt(0).sum())},
        'advisor': {'gpt': gu['advisor'], 'qwen': qu['advisor']}, 'outcome_events': events,
        'advice_generation_comparison': _advice_generation_comparison(gc, qc),
        'comparison_scope': 'Two fixed empirical advice tables under the same saved design, seeds and pure-Fermi controls; not a population-level model ranking.',
        'uncertainty': 'Two simulation seeds per setting; two LLM answers per state form one fixed table, not independent table replications. No significance claims or confidence intervals.',
        'saved_initial_state_limit': 'Initial cellwise grids and eligibility masks were not saved. Initial cooperation rates, group counts and pure-Fermi controls agree; exact cellwise initial equality is not independently reconstructed.',
        'randomness_limit': 'Same seeds and controls do not imply identical random events after interventions; random-number consumption can diverge.',
        'source_audits': {'gpt': 'data_audit_gpt/数据核验.json', 'qwen': '../qwen_20261027_repeats2/数据核验.json'},
        'source_audit_sha256': {'gpt': qdata._sha((report / 'data_audit_gpt/数据核验.json').read_bytes()), 'qwen': qdata._sha((QWEN_REPORT / '数据核验.json').read_bytes())},
        'comparison_module_sha256': qdata._sha(Path(__file__).read_bytes()),
    }
    data = report / 'data'
    data.mkdir(parents=True, exist_ok=True)
    joined.to_csv(data / 'matched_runs_gpt_qwen.csv', index=False, encoding='utf-8-sig')
    state.to_csv(data / 'advisor_state_comparison.csv', index=False, encoding='utf-8-sig')
    cond.to_csv(data / 'advisor_conditioned_comparison.csv', index=False, encoding='utf-8-sig')
    qdata._json_write(data / 'comparison_audit.json', audit)
    return audit


def load_comparison(use_cache=True, report=DEFAULT_REPORT):
    """Return {'gpt': load_scan six-tuple, 'qwen': six-tuple, 'comparison': audit}.

    Trajectory/grid keys are (b, lambda_prob, P, seed). Cache reuse checks every
    source SHA256, cache artifacts, the original loader, and the GPT adapter.
    """
    report = Path(report).absolute()
    if report.is_relative_to(ROOT):
        report = WRITE_ROOT / report.relative_to(ROOT)
    report.mkdir(parents=True, exist_ok=True)
    gpt = _load_gpt(report / 'data_audit_gpt', use_cache)
    qwen = qdata.load_scan(QWEN_BATCH, QWEN_REPORT, use_cache=True)
    comparison = _compare(gpt, qwen, report)
    print(f"Comparison passed: {comparison['matched_runs']} matched runs, {comparison['identical_control_summary_values']} identical control summary values.", flush=True)
    return {'gpt': gpt, 'qwen': qwen, 'comparison': comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--use-cache', action='store_true')
    args = parser.parse_args()
    result = load_comparison(use_cache=args.use_cache, report=args.report)
    print(json.dumps({k: result['comparison'][k] for k in ['model_identity', 'advisor_state_probabilities', 'outcome_events']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

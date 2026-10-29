"""Audit and align three empirical 3x3 advice tables; never invoke an API.

Only final successful records referenced by results.csv enter the comparison.
Archived failed/retried attempts are counted from the summary, not pooled.
All exported metadata use a whitelist: no endpoints, credentials, response IDs,
encrypted response fields, or reasoning text are copied to the report.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_ALIAS = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
ROOT = _ALIAS if _ALIAS.is_dir() and _ALIAS.samefile(Path(__file__).resolve().parents[1]) else Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / 'reports/advice_deepseek_gpt_qwen_20261029'
SOURCES = {
    'DeepSeek': ROOT / 'logs/GPT_Multistepsa/20261029_121229_Deepseek1029',
    'GPT': ROOT / 'logs/GPT_Multistepsa/20261016_173605_GPT5.5',
    'Qwen': ROOT / 'logs/GPT_Multistepsa/20261024_122849_Qwen1024',
}
EXPECTED_N = {'DeepSeek': 1, 'GPT': 2, 'Qwen': 2}
PREVIOUS_HASHES = {
    'GPT': {'config.json': '84247203ef5ce302656164cdf83ff49c8a4ccbe010c8ff2ca87940f414e2241f',
            'results.csv': '72b84472777ca278d98c1d73137a71b8502fb32b9c632523a3fe54b7a52ba900'},
    'Qwen': {'config.json': '88cc8021ef8de651b085d7eadb2e5472053211721ec1744206981b8b9b6f3a2e',
             'results.csv': '5766a86bde589daedeec2c27ce744ef0bc12a92b83eff15d7a935b3b7a32ed16'},
}
KEYS = [''.join(v) for v in itertools.product('CD', repeat=9)]
SAFE_SETTINGS = ('temperature', 'top_p', 'top_k', 'seed', 'max_tokens',
                 'max_output_tokens', 'max_completion_tokens', 'frequency_penalty',
                 'presence_penalty', 'reasoning_effort', 'stream', 'store')
SAFE_REASONING = ('effort', 'context', 'mode', 'summary', 'generate_summary')
SAFE_EXTRA = ('cache_prompt', 'reasoning_format', 'reasoning_effort', 'enable_thinking')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def state_metadata(key):
    return {'b': None, 'current_action': key[4],
            'cooperating_neighbors': sum(key[i] == 'C' for i in (1, 3, 5, 7)),
            'defecting_neighbors': sum(key[i] == 'D' for i in (1, 3, 5, 7)),
            'grid_key': key, 'grid_rows': [key[:3], key[3:6], key[6:]]}


def expected_user_prompt(key):
    """Rebuild the observed user text independently from its row-major state."""
    k = sum(key[i] == 'C' for i in (1, 3, 5, 7))
    payoff = str(k) if key[4] == 'C' or k == 0 else f'{k}*b'
    grid = '\n'.join(' '.join(key[i:i + 3]) for i in (0, 3, 6))
    return f'''Game state:

Current local 3x3 grid (C = Cooperate, D = Defect):
{grid}

I am the center player at row 2, column 2.

My current payoff is {payoff}.

What action would you recommend for the center player in the next round to maximizes my expected cumulative payoff over the entire spatial repeated game.?

Return only one action: C or D.'''


def safe_settings(container):
    out = {k: container[k] for k in SAFE_SETTINGS if k in container}
    if isinstance(container.get('reasoning'), dict):
        out['reasoning'] = {k: container['reasoning'][k] for k in SAFE_REASONING if k in container['reasoning']}
    if isinstance(container.get('extra_body'), dict):
        out['extra_body'] = {k: container['extra_body'][k] for k in SAFE_EXTRA if k in container['extra_body']}
    return out


def response_text(response):
    if 'choices' in response:
        require(len(response['choices']) == 1, 'Expected one final chat choice')
        return response['choices'][0]['message']['content']
    text = []
    for item in response.get('output', []):
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            for part in item.get('content', []):
                if part.get('type') == 'output_text':
                    text.append(part['text'])
    return ''.join(text)


def counter_export(values):
    counts = Counter(json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values)
    return [{'settings': json.loads(k), 'count': v} for k, v in sorted(counts.items())]


def load_advice(report=DEFAULT_REPORT):
    """Return frames, aligned states, safe metadata, and complete source audit.

    frames[label] has one row per successful answer, with center/cardinal_c/
    diagonal_c/is_c. states has exactly 512 rows: label columns are empirical
    C fractions, not calibrated action probabilities. label_n/C/D give counts.
    """
    report = Path(report)
    if not report.is_absolute():
        report = ROOT / report
    data_dir = report / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    frames, source_audits, source_hashes, prompts = {}, {}, {}, {}
    total_records = 0
    metadata = []
    states = pd.DataFrame({'grid_key': KEYS})
    states['center'] = states.grid_key.str[4]
    states['cardinal_c'] = states.grid_key.map(lambda k: sum(k[i] == 'C' for i in (1, 3, 5, 7)))
    states['diagonal_c'] = states.grid_key.map(lambda k: sum(k[i] == 'C' for i in (0, 2, 6, 8)))
    states['total_c'] = states.grid_key.str.count('C')
    user_hashes = {k: sha(expected_user_prompt(k).encode()) for k in KEYS}
    for label, path in SOURCES.items():
        raw = {name: (path / name).read_bytes() for name in ('config.json', 'summary.json', 'results.csv')}
        config, summary = json.loads(raw['config.json']), json.loads(raw['summary.json'])
        df = pd.read_csv(io.BytesIO(raw['results.csv']), encoding='utf-8-sig')
        hashes = {name: sha(value) for name, value in raw.items()}
        for name, value in hashes.items():
            source_hashes[str((path / name).relative_to(ROOT))] = value
        n = EXPECTED_N[label]
        expected_count = 512 * n
        require(config['n'] == summary['n'] == n, f'{label}: n mismatch')
        require(len(df) == expected_count, f'{label}: wrong row count')
        require(config['planned_requests'] == summary['planned_requests'] == expected_count, f'{label}: planned requests')
        require(summary['status'] == 'completed', f'{label}: incomplete batch')
        require(summary['successful_requests'] == expected_count, f'{label}: success count')
        require(all(summary.get(k, 0) == 0 for k in ['failed_requests', 'cancelled_requests', 'not_run_requests', 'pending_requests']), f'{label}: outstanding/failing rows')
        require(config['b'] is None and summary['b'] is None, f'{label}: numerical b unexpectedly specified')
        require(config['b_constraint'] == summary['b_constraint'] == '1 < b < 2', f'{label}: b interval')
        require(config['observation_format'] == summary['observation_format'] == 'grid_3x3', f'{label}: wrong format')
        require(config['case_count'] == summary['case_count'] == 512, f'{label}: case count')
        require(config['cases'] == [state_metadata(k) for k in KEYS], f'{label}: config cases metadata/order')
        require(len(config['requests']) == 512, f'{label}: request templates')
        require(df['grid_key'].nunique() == 512 and set(df.grid_key) == set(KEYS), f'{label}: state coverage')
        require(df.groupby('grid_key').size().eq(n).all(), f'{label}: state repeat count')
        require(set(df.repetition) == set(range(1, n + 1)), f'{label}: repetition indices')
        require(not df.duplicated(['grid_key', 'repetition']).any(), f'{label}: duplicate state/repetition')
        require(df.request_index.tolist() == list(range(1, expected_count + 1)), f'{label}: request ordering')
        require(df.record_file.nunique() == expected_count, f'{label}: duplicate record paths')
        require(df.status.eq('success').all() and df.recommendation.isin(['C', 'D']).all(), f'{label}: non-success/invalid action')
        require(df.output_text.eq(df.recommendation).all(), f'{label}: output/action mismatch')
        require(df.error.isna().all(), f'{label}: unexpected errors')
        for column, value in [('b', 'unknown'), ('b_mode', 'unknown_interval'), ('b_constraint', '1 < b < 2'), ('observation_format', 'grid_3x3')]:
            require(df[column].eq(value).all(), f'{label}: {column} mismatch')
        for key, group in df.groupby('grid_key'):
            m = state_metadata(key)
            for col in ['current_action', 'cooperating_neighbors', 'defecting_neighbors']:
                require(group[col].eq(m[col]).all(), f'{label}: {key}: {col}')
            require(group.grid_rows.eq('/'.join(m['grid_rows'])).all(), f'{label}: grid rows')
        prompts[label] = config['system_prompt']

        def audit_record(row):
            relative = Path(row['record_file'])
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'records', f'{label}: invalid record path')
            record_path = path / relative
            b = record_path.read_bytes()
            r = json.loads(b)
            key = row['grid_key']
            require(r['observation'] == state_metadata(key), f'{label}: record observation mismatch')
            for col in ['request_index', 'repetition', 'combination_index']:
                require(r[col] == row[col], f'{label}: record {col} mismatch')
            require(row['combination_index'] == KEYS.index(key) + 1, f'{label}: combination index')
            require(row['request_index'] == (row['repetition'] - 1) * 512 + row['combination_index'], f'{label}: request index')
            require(r['batch_id'] == config['batch_id'] == summary['batch_id'], f'{label}: batch ID mismatch')
            require(r['status'] == 'success' and r['error'] is None and r['api_call_started'] is True, f'{label}: record failed')
            require(r['action'] == r['output_text'] == row['recommendation'], f'{label}: record action mismatch')
            require(r['b_mode'] == 'unknown_interval' and r['b_constraint'] == '1 < b < 2', f'{label}: record b interval')
            q, response = r['request'], r['response']
            require(q == config['requests'][row['combination_index'] - 1], f'{label}: request/template mismatch')
            require(q['model'] == config['model'], f'{label}: request model mismatch')
            if 'model' in row:
                require(row['model'] == q['model'], f'{label}: row model mismatch')
            # A service may return a provider alias/version different from the
            # requested model string. Preserve that evidence, without asserting
            # that aliases identify the same underlying model.
            require(isinstance(response.get('model'), str) and bool(response['model']), f'{label}: missing response model metadata')
            require(response_text(response) == row['output_text'], f'{label}: response/action mismatch')
            require(response['id'] == row['response_id'], f'{label}: response ID mismatch')
            messages = q.get('messages', q.get('input'))
            require(len(messages) == 2 and [m['role'] for m in messages] == ['system', 'user'], f'{label}: unexpected history')
            require(messages[0]['content'] == config['system_prompt'], f'{label}: system prompt mismatch')
            require(messages[1]['content'] == expected_user_prompt(key), f'{label}: user state/payoff text mismatch')
            return {'path': str(record_path.relative_to(ROOT)), 'sha256': sha(b),
                    'response_model': response['model'],
                    'request_settings': safe_settings(q), 'response_settings': safe_settings(response)}

        with ThreadPoolExecutor(max_workers=6) as pool:
            audited = list(pool.map(audit_record, df.to_dict('records')))
        total_records += len(audited)
        for a in audited:
            source_hashes[a['path']] = a['sha256']
        # Compare summary case histograms to the actual final successful actions.
        require(len(summary['cases']) == 512, f'{label}: summary cases')
        for index, item in enumerate(summary['cases']):
            key = KEYS[index]
            require(all(item[k] == v for k, v in state_metadata(key).items()), f'{label}: summary state metadata')
            subset = df[df.grid_key.eq(key)]
            require(item['C'] == int(subset.recommendation.eq('C').sum()), f'{label}: summary C count')
            require(item['D'] == int(subset.recommendation.eq('D').sum()), f'{label}: summary D count')
            require(item.get('errors', 0) == item.get('cancelled', 0) == 0, f'{label}: summary case failure')
        prior = {name: {'expected': fingerprint, 'actual': hashes[name], 'matches': hashes[name] == fingerprint}
                 for name, fingerprint in PREVIOUS_HASHES.get(label, {}).items()}
        require(all(v['matches'] for v in prior.values()), f'{label}: historical source changed')
        # Keep only analytical columns, excluding infrastructure/provider details.
        clean_cols = ['request_index', 'repetition', 'combination_index', 'grid_key', 'grid_rows',
                      'current_action', 'cooperating_neighbors', 'defecting_neighbors',
                      'recommendation', 'output_text', 'duration_seconds', 'input_tokens',
                      'output_tokens', 'total_tokens', 'record_file']
        df = df[clean_cols].copy()
        df['model_label'] = label
        df['request_model'] = config['model']
        df['response_model'] = [a['response_model'] for a in audited]
        df['center'] = df.grid_key.str[4]
        df['cardinal_c'] = df.cooperating_neighbors
        df['diagonal_c'] = df.grid_key.map(lambda k: sum(k[i] == 'C' for i in (0, 2, 6, 8)))
        df['total_c'] = df.grid_key.str.count('C')
        df['is_c'] = df.recommendation.eq('C').astype(int)
        frames[label] = df
        grouped = df.groupby('grid_key').is_c.agg(['mean', 'count', 'sum']).reindex(KEYS)
        states[label] = grouped['mean'].to_numpy()
        states[f'{label}_n'] = grouped['count'].to_numpy()
        states[f'{label}_C'] = grouped['sum'].to_numpy()
        states[f'{label}_D'] = (grouped['count'] - grouped['sum']).to_numpy()
        same_count = int(grouped['mean'].isin([0, 1]).sum()) if n >= 2 else None
        safe_config = {k: config[k] for k in ['model', 'n', 'b', 'b_mode', 'b_constraint', 'planned_requests',
                                            'observation_format', 'case_count', 'max_output_tokens',
                                            'qwen_reasoning_effort', 'order'] if k in config}
        safe_config['api_interface'] = config.get('api_interface', 'responses' if 'input' in config['requests'][0] else 'chat.completions')
        entry = {'source': str(path.relative_to(ROOT)), 'safe_config': safe_config,
                 'successful_answers': len(df), 'unique_states': 512, 'answers_per_state': n,
                 'cooperation_answers': int(df.is_c.sum()), 'cooperation_fraction': float(df.is_c.mean()),
                 'two_answer_agreement_states': same_count,
                 'two_answer_agreement_fraction': same_count / 512 if same_count is not None else None,
                 'agreement_note': 'Not estimable from one answer per state' if n == 1 else 'Descriptive agreement of the two recorded answers per state',
                 'archived_attempts_excluded': summary.get('archived_attempts', 0),
                 'api_calls_started_total': summary.get('api_calls_started_total'),
                 'started_at': summary.get('started_at'), 'finished_at': summary.get('finished_at'),
                 'request_settings_variants': counter_export([a['request_settings'] for a in audited]),
                 'response_settings_variants': counter_export([a['response_settings'] for a in audited]),
                 'response_model_counts': dict(Counter(a['response_model'] for a in audited)),
                 'response_model_differs_from_request_count': sum(a['response_model'] != config['model'] for a in audited),
                 'system_prompt_sha256': sha(config['system_prompt'].encode()),
                 'config_results_summary_sha256': hashes, 'previous_report_fingerprint_comparison': prior}
        source_audits[label] = entry
        metadata.append({'model_label': label, 'recorded_model': config['model'], 'answers_per_state': n,
                         'successful_answers': len(df), 'C_answers': int(df.is_c.sum()),
                         'C_fraction': float(df.is_c.mean()), 'two_answer_agreement_states': same_count,
                         'two_answer_agreement_fraction': same_count / 512 if same_count is not None else None,
                         'max_output_tokens': config['max_output_tokens'], 'archived_attempts_excluded': summary.get('archived_attempts', 0)})
        print(f'{label}: validated {len(audited)} records and all 512 state prompts', flush=True)

    require(len(set(prompts.values())) == 1, 'System prompts differ across models')
    require(states.shape[0] == 512 and not states.isna().any().any(), 'Aligned states incomplete')
    audit = {'status': 'passed', 'scope': 'Empirical advice output tables only; no SPD or API calls',
             'source_count': 3, 'records_validated': total_records, 'source_files_hashed': len(source_hashes),
             'all_system_prompts_identical': True, 'all_paired_user_prompts_identical': True,
             'all_requests_independent_two_message_context': True,
             'checks': ['Full 512-state Cartesian coverage without rotation/reflection merging',
                        'One or two final successful answers per state according to source configuration',
                        'Row-major grid, center action, orthogonal neighbor counts and payoff prompt',
                        'Config request templates, record observations, actions and final response content',
                        'Summary action histograms, record paths, indices, and model metadata',
                        'Identical system prompt and per-state user prompt across all three sources',
                        'SHA256 of all config, summary, results, and referenced records',
                        'Historical GPT and Qwen fingerprints when available'],
             'limitations': ['DeepSeek has one answer per state: within-state repeat agreement/variance cannot be estimated.',
                             'GPT/Qwen have only two answers per state; empirical fractions are not calibrated probabilities.',
                             'Uniform weighting of 512 states does not represent state visitation during an SPD run.',
                             'Sampling/reasoning settings and generation dates differ; this is not a controlled model-capability ranking.',
                             'Model identity is the local request/response metadata, not independently verified.',
                             'GPT response metadata contain multiple provider aliases/version strings despite a single requested model; their underlying equivalence is unverified.',
                             'Missing request or response settings mean unrecorded, not zero or identical defaults.',
                             'Archived attempts are excluded; only the final successful table is compared.'],
             'sources': source_audits, 'user_prompt_sha256_by_state': user_hashes,
             'source_sha256': dict(sorted(source_hashes.items())),
             'loader_sha256': sha(Path(__file__).read_bytes())}
    for label, df in frames.items():
        df.to_csv(data_dir / f'{label}_clean_answers.csv', index=False, encoding='utf-8-sig')
    states.to_csv(data_dir / 'aligned_512_states.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metadata).to_csv(data_dir / 'source_overview.csv', index=False, encoding='utf-8-sig')
    (data_dir / 'common_system_prompt.txt').write_text(next(iter(prompts.values())), encoding='utf-8')
    (data_dir / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'frames': frames, 'states': states, 'audit': audit, 'metadata': pd.DataFrame(metadata)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = load_advice(args.report_dir)
    print(result['metadata'].to_string(index=False))
    print(f"Audit passed: {result['audit']['records_validated']} records, {result['audit']['source_files_hashed']} source files")

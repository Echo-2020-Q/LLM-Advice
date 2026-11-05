"""Audit six local batches and preserve five final answers per state/model.

No API requests or simulations are made. The derived per-model config/results
folders can be read by main/SPD/llm_advisor.py. Original logs stay unchanged.
Only whitelisted metadata are exported; raw reasoning and service details are
not copied. Requires Python's standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ALIAS = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
ROOT = ALIAS if ALIAS.exists() else Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/1101_3LLM_Analysis'
SOURCES = {
    'DeepSeek': [('old', '20261029_121229_Deepseek1029_1次', 1),
                 ('new', '20261029_140923_Deepseek1030_4次', 4)],
    'GPT': [('old', '20261016_173605_GPT5.5_2次', 2),
            ('new', '20261029_141751_GPT5.5_3次', 3)],
    'Qwen': [('old', '20261024_122849_Qwen1024_2次', 2),
             ('new', '20261029_141235_Qwen1031_3次', 3)],
}
KEYS = [''.join(v) for v in itertools.product('CD', repeat=9)]
SAFE = ('temperature', 'top_p', 'top_k', 'seed', 'max_tokens',
        'max_output_tokens', 'max_completion_tokens', 'frequency_penalty',
        'presence_penalty', 'reasoning_effort', 'stream', 'store')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state(key):
    k = sum(key[i] == 'C' for i in (1, 3, 5, 7))
    return dict(b=None, current_action=key[4], cooperating_neighbors=k,
                defecting_neighbors=4-k, grid_key=key,
                grid_rows=[key[:3], key[3:6], key[6:]])


def user_prompt(key):
    k = state(key)['cooperating_neighbors']
    payoff = str(k) if key[4] == 'C' or k == 0 else f'{k}*b'
    grid = '\n'.join(' '.join(key[i:i+3]) for i in (0, 3, 6))
    return f'''Game state:

Current local 3x3 grid (C = Cooperate, D = Defect):
{grid}

I am the center player at row 2, column 2.

My current payoff is {payoff}.

What action would you recommend for the center player in the next round to maximizes my expected cumulative payoff over the entire spatial repeated game.?

Return only one action: C or D.'''


def safe_settings(container):
    result = {k: container[k] for k in SAFE if k in container}
    for outer, keys in [('reasoning', ('effort', 'summary', 'generate_summary')),
                        ('extra_body', ('cache_prompt', 'reasoning_format',
                                        'reasoning_effort', 'enable_thinking'))]:
        if isinstance(container.get(outer), dict):
            result[outer] = {k: container[outer][k] for k in keys
                             if k in container[outer]
                             and isinstance(container[outer][k], (str, int, float, bool, type(None)))}
    return result


def variants(values):
    counts = Counter(json.dumps(x, sort_keys=True) for x in values)
    return [dict(settings=json.loads(k), count=v) for k, v in sorted(counts.items())]


def response_text(response):
    if 'choices' in response:
        require(len(response['choices']) == 1, 'Expected exactly one final chat choice')
        return response['choices'][0]['message']['content']
    return ''.join(part['text'] for item in response.get('output', [])
                   if item.get('type') == 'message' and item.get('role') == 'assistant'
                   for part in item.get('content', []) if part.get('type') == 'output_text')


def audit_batch(model, label, folder, n, offset):
    path = ROOT / 'logs/GPT_Multistepsa' / folder
    blobs = {name: (path/name).read_bytes() for name in ('config.json', 'summary.json', 'results.csv')}
    config, summary = (json.loads(blobs[name]) for name in ('config.json', 'summary.json'))
    rows = list(csv.DictReader(io.StringIO(blobs['results.csv'].decode('utf-8-sig'))))
    prefix = f'{model}/{label}'
    require(config['n'] == summary['n'] == n, f'{prefix}: n mismatch')
    require(config['planned_requests'] == summary['planned_requests'] == len(rows) == 512*n, f'{prefix}: row count')
    require(summary['status'] == 'completed' and summary['successful_requests'] == len(rows), f'{prefix}: incomplete')
    require(all(summary.get(k, 0) == 0 for k in ('failed_requests', 'cancelled_requests', 'not_run_requests', 'pending_requests')), f'{prefix}: unsuccessful rows')
    require(config['b'] is None and summary['b'] is None, f'{prefix}: numeric b')
    require(config['b_constraint'] == summary['b_constraint'] == '1 < b < 2', f'{prefix}: b interval')
    require(config['observation_format'] == summary['observation_format'] == 'grid_3x3', f'{prefix}: format')
    require(config['case_count'] == summary['case_count'] == 512, f'{prefix}: case count')
    require(config['cases'] == [state(k) for k in KEYS], f'{prefix}: states metadata/order')
    require(len(config['requests']) == 512, f'{prefix}: request templates')
    require([int(r['request_index']) for r in rows] == list(range(1, len(rows)+1)), f'{prefix}: request order')
    require(Counter(r['grid_key'] for r in rows) == Counter({k:n for k in KEYS}), f'{prefix}: coverage')
    require(len({(r['grid_key'], r['repetition']) for r in rows}) == len(rows), f'{prefix}: repeated key')
    require(len({r['record_file'] for r in rows}) == len(rows), f'{prefix}: repeated file')

    def audit_record(row):
        key = row['grid_key']; meta = state(key)
        relative = Path(row['record_file'])
        require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'records', f'{prefix}: unsafe record path')
        blob = (path/relative).read_bytes(); record = json.loads(blob)
        idx, rep, case = [int(row[k]) for k in ('request_index', 'repetition', 'combination_index')]
        require(1 <= rep <= n and case == KEYS.index(key)+1 and idx == (rep-1)*512+case, f'{prefix}: indices')
        require(record['observation'] == meta, f'{prefix}: observation')
        require(all(record[k] == int(row[k]) for k in ('request_index', 'repetition', 'combination_index')), f'{prefix}: record indices')
        require(record['batch_id'] == config['batch_id'] == summary['batch_id'], f'{prefix}: batch ID')
        require(row['status'] == record['status'] == 'success' and row['recommendation'] in 'CD' and len(row['recommendation']) == 1, f'{prefix}: status/action')
        require(record['error'] is None and not row['error'] and record['api_call_started'] is True, f'{prefix}: failed record')
        require(record['action'] == record['output_text'] == row['output_text'] == row['recommendation'], f'{prefix}: action/output')
        for col in ('b_mode', 'b_constraint'):
            require(row[col] == record[col] == config[col], f'{prefix}: {col}')
        require(row['b'] == 'unknown' and row['observation_format'] == 'grid_3x3', f'{prefix}: row format')
        require(row['current_action'] == meta['current_action'] and row['grid_rows'] == '/'.join(meta['grid_rows']), f'{prefix}: grid')
        require(all(int(row[k]) == meta[k] for k in ('cooperating_neighbors', 'defecting_neighbors')), f'{prefix}: neighbor counts')
        request, response = record['request'], record['response']
        require(request == config['requests'][case-1] and request['model'] == config['model'], f'{prefix}: template/model')
        require(not row.get('model') or row['model'] == config['model'], f'{prefix}: row model')
        require(isinstance(response.get('model'), str) and response['model'], f'{prefix}: response model')
        require(response_text(response) == row['recommendation'] and response['id'] == row['response_id'], f'{prefix}: response text/id')
        usage = response.get('usage') or {}
        token_names = {'input_tokens': ('input_tokens', 'prompt_tokens'),
                       'output_tokens': ('output_tokens', 'completion_tokens'),
                       'total_tokens': ('total_tokens',)}
        for column, names in token_names.items():
            value = next((usage[name] for name in names if name in usage), None)
            require((not row[column] and value is None) or (row[column] and int(row[column]) == value), f'{prefix}: usage {column}')
        require(float(row['duration_seconds']) == record['duration_seconds'], f'{prefix}: duration')
        finish_status = (response['choices'][0].get('finish_reason', 'unrecorded')
                         if 'choices' in response else response.get('status', 'unrecorded'))
        messages = request.get('messages', request.get('input'))
        require(len(messages) == 2 and [m['role'] for m in messages] == ['system', 'user'], f'{prefix}: history')
        require(messages[0]['content'] == config['system_prompt'] and messages[1]['content'] == user_prompt(key), f'{prefix}: prompt')
        rs, ss = safe_settings(request), safe_settings(response)
        clean = dict(model_label=model, batch_label=label, source_folder=str(path.relative_to(ROOT)),
                     source_batch_id=config['batch_id'], original_request_index=idx,
                     original_repetition=rep, request_index=(offset+rep-1)*512+case,
                     repetition=offset+rep, combination_index=case,
                     b='unknown', b_mode='unknown_interval', b_constraint='1 < b < 2',
                     observation_format='grid_3x3', grid_key=key,
                     grid_rows='/'.join(meta['grid_rows']), current_action=key[4],
                     cooperating_neighbors=meta['cooperating_neighbors'],
                     defecting_neighbors=meta['defecting_neighbors'], center=key[4],
                     cardinal_c=meta['cooperating_neighbors'],
                     diagonal_c=sum(key[i] == 'C' for i in (0, 2, 6, 8)), total_c=key.count('C'),
                     status='success', recommendation=row['recommendation'], output_text=row['output_text'],
                     is_c=int(row['recommendation'] == 'C'), request_model=request['model'], response_model=response['model'],
                     reported_finish_status=finish_status,
                     duration_seconds=row['duration_seconds'], input_tokens=row['input_tokens'],
                     output_tokens=row['output_tokens'], total_tokens=row['total_tokens'],
                     source_record_file=str((path/relative).relative_to(ROOT)), source_record_sha256=sha(blob),
                     request_settings_sha256=sha(json.dumps(rs, sort_keys=True).encode()),
                     response_settings_sha256=sha(json.dumps(ss, sort_keys=True).encode()))
        return clean, rs, ss, sha(response['id'].encode())

    with ThreadPoolExecutor(max_workers=8) as pool:
        audited = list(pool.map(audit_record, rows))
    clean = [x[0] for x in audited]
    require(len({x[3] for x in audited}) == len(clean), f'{prefix}: repeated final response IDs')
    counts = Counter((r['grid_key'], r['recommendation']) for r in clean)
    require(len(summary['cases']) == 512, f'{prefix}: summary coverage')
    for key, item in zip(KEYS, summary['cases']):
        require(all(item[k] == v for k,v in state(key).items()), f'{prefix}: summary state')
        require(item['C'] == counts[key,'C'] and item['D'] == counts[key,'D'], f'{prefix}: summary histogram')
        require(item.get('errors', 0) == item.get('cancelled', 0) == 0, f'{prefix}: summary errors')
    hashes = {str((path/name).relative_to(ROOT)):sha(blob) for name,blob in blobs.items()}
    hashes.update({r['source_record_file']:r['source_record_sha256'] for r in clean})
    attempts = list((path/'attempts').rglob('*.json')) if (path/'attempts').exists() else []
    batch = dict(model_label=model, batch_label=label, source_folder=str(path.relative_to(ROOT)),
                 source_batch_id=config['batch_id'], request_model=config['model'],
                 n=n, successful_answers=len(clean), C_answers=sum(r['is_c'] for r in clean),
                 C_fraction=sum(r['is_c'] for r in clean)/len(clean),
                 merged_repetition_start=offset+1, merged_repetition_end=offset+n,
                 started_at=summary['started_at'], finished_at=summary['finished_at'],
                 archived_attempts_excluded=summary.get('archived_attempts', 0),
                 archived_attempt_files_present=len(attempts),
                 request_settings_variants=variants([x[1] for x in audited]),
                 response_settings_variants=variants([x[2] for x in audited]),
                 response_model_counts=dict(Counter(r['response_model'] for r in clean)),
                 reported_finish_status_counts=dict(Counter(r['reported_finish_status'] for r in clean)),
                 final_response_ids_unique_within_batch=True,
                 response_model_differs_from_request_count=sum(r['response_model'] != config['model'] for r in clean),
                 system_prompt_sha256=sha(config['system_prompt'].encode()),
                 config_results_summary_sha256={name:sha(b) for name,b in blobs.items()},
                 max_output_tokens=config['max_output_tokens'])
    print(f'{prefix}: audited {len(clean)} final records; C={batch["C_answers"]}', flush=True)
    return clean, batch, hashes, config['system_prompt']


def merge(report=REPORT):
    report = Path(report)
    (report/'merged').mkdir(parents=True, exist_ok=True)
    (report/'data').mkdir(exist_ok=True)
    combined, batches, hashes, prompts = [], [], {}, []
    frames = {}
    for model, sources in SOURCES.items():
        offset = 0; frames[model] = []
        for label, folder, n in sources:
            rows, batch, source_hashes, prompt = audit_batch(model, label, folder, n, offset)
            frames[model].extend(rows); batches.append(batch); hashes.update(source_hashes); prompts.append(prompt)
            offset += n
        require(offset == 5 and len(frames[model]) == 2560, f'{model}: merged count')
        require(Counter((r['grid_key'], r['repetition']) for r in frames[model]) == Counter({(k,i):1 for k in KEYS for i in range(1,6)}), f'{model}: merge alignment')
        combined.extend(frames[model])
    require(len(set(prompts)) == 1, 'System prompts differ across batches')
    write_csv(report/'merged/combined_answers.csv', combined)
    states = []
    for key in KEYS:
        item = dict(grid_key=key, center=key[4], cardinal_c=state(key)['cooperating_neighbors'],
                    diagonal_c=sum(key[i] == 'C' for i in (0,2,6,8)), total_c=key.count('C'))
        for model, rows in frames.items():
            selected = [r for r in rows if r['grid_key'] == key]
            c = sum(r['is_c'] for r in selected)
            item.update({f'{model}_n':5, f'{model}_C':c, f'{model}_D':5-c, f'{model}_p_C':c/5})
            for label in ('old', 'new'):
                subset = [r for r in selected if r['batch_label'] == label]; csub = sum(r['is_c'] for r in subset)
                item.update({f'{model}_{label}_n':len(subset), f'{model}_{label}_C':csub,
                             f'{model}_{label}_p_C':csub/len(subset)})
        states.append(item)
    write_csv(report/'data/aligned_states.csv', states)
    metadata = [{k:v for k,v in b.items() if not isinstance(v, (dict,list))} for b in batches]
    write_csv(report/'data/metadata.csv', metadata)
    (report/'data/common_system_prompt.txt').write_text(prompts[0]+'\n', encoding='utf-8')
    (report/'data/example_user_prompt.txt').write_text(user_prompt('CCCCDCCCC')+'\n', encoding='utf-8')
    compatibility = {}
    spec = importlib.util.spec_from_file_location('_five_advice_loader', ROOT/'main/SPD/llm_advisor.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    for model, rows in frames.items():
        target = report/'merged'/model; target.mkdir(exist_ok=True)
        source_entries = [b for b in batches if b['model_label'] == model]
        config = dict(batch_id=f'derived_1101_{model}_five_answers', model=rows[0]['request_model'],
                      b=None, b_mode='unknown_interval', b_constraint='1 < b < 2', n=5,
                      planned_requests=2560, observation_format='grid_3x3', case_count=512,
                      system_prompt=prompts[0], cases=[state(k) for k in KEYS],
                      derived_dataset=True, source_batches=source_entries,
                      merge_order='Old batch first, then new batch; repetitions retain chronological batch order.',
                      source_record_paths_relative_to='repository root',
                      note='Offline merger of final successful answers; no new requests. Generation settings differ by source batch.')
        dump(target/'config.json', config); write_csv(target/'results.csv', rows)
        dump(target/'summary.json', dict(status='completed', derived_dataset=True, n=5, case_count=512,
                 successful_answers=2560, C_answers=sum(r['is_c'] for r in rows),
                 sources=source_entries))
        advisor = module.EmpiricalAdvisor(target)
        loaded = {r['grid_key']:r for r in advisor.table_rows()}
        require(all(loaded[s['grid_key']]['C'] == s[f'{model}_C'] and loaded[s['grid_key']]['samples'] == 5 for s in states), f'{model}: loader altered histogram')
        compatibility[model] = dict(status='passed', loaded_states=len(loaded), samples_per_state=5,
                                   every_C_D_histogram_matches_merged_data=True)
    audit = dict(status='passed', source_batch_count=6, models=list(SOURCES), records_validated=len(combined),
                 source_files_hashed=len(hashes), states_per_model=512, answers_per_state_per_model=5,
                 source_logs_modified=False, all_system_prompts_identical=True,
                 all_paired_user_prompts_identical=True, all_requests_independent_two_message_context=True,
                 archived_attempts_excluded=sum(b['archived_attempts_excluded'] for b in batches),
                 empirical_advisor_compatibility=compatibility, sources=batches,
                 checks=['All 7,680 final rows verified against original JSON records and raw final response text.',
                         'Every state has exactly five distinct original request identities per model.',
                         'State observations, row-major keys, cardinal-neighbor counts, payoff prompts, request templates and summary histograms agree.',
                         'Final response IDs are unique within each batch; exported token counts and duration agree with original records.',
                         'Each merged state/repetition identity appears once; no majority-vote replacement or symmetry merging.',
                         'Identical system text and independently reconstructed per-state user prompts across all six batches.',
                         'All source config, summary, results and referenced final records SHA256-hashed.',
                         'Existing EmpiricalAdvisor loaded each derived folder and reproduced every state histogram; no simulation executed.'],
                 limitations=['Five answers per state are a small empirical sample, not calibrated action probabilities.',
                              'Merged answers come from two batches per model; sampling dates/settings can differ.',
                              'GPT and Qwen output-token limits changed from 10,240 to 51,200 between batches; DeepSeek used 51,200 in both.',
                              'Retries are excluded; results characterize the final successful recorded advice tables.',
                              'Uniform weighting of all 512 states is not their visitation frequency in an SPD simulation.',
                              'Returned model aliases and settings are log metadata; provider identity and undocumented defaults were not independently verified.'],
                 user_prompt_sha256_by_state={k:sha(user_prompt(k).encode()) for k in KEYS},
                 merge_script_sha256=sha(Path(__file__).read_bytes()))
    dump(report/'data/audit.json', audit)
    dump(report/'data/source_manifest.json', dict(hash_algorithm='SHA256', paths_relative_to='repository root',
                 scope='6 x (config.json, summary.json, results.csv) plus all 7680 final referenced records; retry archives excluded.',
                 source_sha256=dict(sorted(hashes.items()))))
    (report/'merged/README.txt').write_text('MERGED FIVE-ANSWER ADVICE TABLES\n\n'
        'Each model folder contains a sanitized config.json and results.csv accepted by main/SPD/llm_advisor.py.\n'
        'There are 512 oriented 3x3 states and five original final successful answers per state (2,560 rows/model).\n'
        'Old batch answers precede new batch answers. Merged repetition is 1..5; original repetition/request IDs and\n'
        'source record paths/hashes remain in every row. Paths are relative to the repository root.\n'
        'Original logs are unchanged. No retries, reasoning text, service URLs or credentials are copied.\n'
        'combined_answers.csv stacks all 7,680 answers; it is for analysis, not a single-model advisor input.\n'
        'C and D counts are preserved. Selecting a majority answer would change the empirical lookup policy.\n'
        'Generation settings are retained per source in config.json and ../data/audit.json, not asserted identical.\n'
        'GPT/Qwen old/new output-token caps are 10,240/51,200; DeepSeek used 51,200 in both. None of the retained answers reached its cap; all recorded output counts are below 10,240. This does not prevent pooling the five replies.\n', encoding='utf-8')
    print(f'PASS: {len(combined)} final answers, {len(hashes)} hashed source files, three compatible merged folders.', flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=REPORT)
    merge(parser.parse_args().report_dir)

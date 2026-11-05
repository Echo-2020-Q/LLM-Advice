#!/usr/bin/env python3
"""Recompute pooled, equally weighted five-reply descriptions from merged rows.

No splitting or weighting by collection history. All reported rates are
descriptive frequencies on the complete 512-state grid, not significance tests.
Uses only Python's standard library; leaves the input CSV untouched.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
REPORT = ROOT / 'reports/1101_3LLM_Analysis/Chinese_Pooled5'
SOURCE = ROOT / 'reports/1101_3LLM_Analysis/merged/combined_answers.csv'
MODELS = ('GPT', 'DeepSeek', 'Qwen')
STATES = tuple(''.join(v) for v in itertools.product('CD', repeat=9))
CARDINAL = (1, 3, 5, 7)
DIAGONAL = (0, 2, 6, 8)


def features(state):
    return {'center': state[4], 'cardinal_c': sum(state[i] == 'C' for i in CARDINAL),
            'diagonal_c': sum(state[i] == 'C' for i in DIAGONAL)}


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    out = REPORT / 'data'
    review = REPORT / 'review'
    out.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == 7680
    groups = defaultdict(list)
    clean = []
    unique = set()
    for row in raw:
        model, state = row['model_label'], row['grid_key']
        answer, rep = row['recommendation'], int(row['repetition'])
        assert model in MODELS and state in STATES
        assert row['status'] == 'success' and answer in ('C', 'D')
        assert rep in range(1, 6)
        assert (model, state, rep) not in unique
        unique.add((model, state, rep))
        f = features(state)
        assert row['center'] == row['current_action'] == f['center']
        assert int(row['cardinal_c']) == f['cardinal_c']
        assert int(row['diagonal_c']) == f['diagonal_c']
        assert int(row['is_c']) == int(answer == 'C')
        record = {'model': model, 'grid_key': state, 'center': f['center'],
                  'repetition': rep, 'recommendation': answer,
                  'is_c': int(answer == 'C'), 'cardinal_c': f['cardinal_c'],
                  'diagonal_c': f['diagonal_c']}
        clean.append(record)
        groups[(model, state)].append(record)
    assert set(groups) == {(m, s) for m in MODELS for s in STATES}
    assert all(len(v) == 5 and {r['repetition'] for r in v} == set(range(1, 6))
               for v in groups.values())
    clean.sort(key=lambda r: (MODELS.index(r['model']), r['grid_key'], r['repetition']))
    counts = {(m, s): sum(r['is_c'] for r in groups[(m, s)])
              for m in MODELS for s in STATES}
    state_rows = []
    model_rows = []
    neighbor_rows = []
    context_rows = []
    vote_rows = []
    for model in MODELS:
        values = [counts[(model, s)] for s in STATES]
        entropy = []
        agreeing_pairs = []
        for state, k in zip(STATES, values):
            p = k / 5
            entropy.append(-p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0)
            n_agree = math.comb(k, 2) + math.comb(5 - k, 2)
            agreeing_pairs.append(n_agree)
            state_rows.append({'model': model, 'grid_key': state,
                               'grid_rows': '/'.join(state[i:i + 3] for i in (0, 3, 6)),
                               **features(state), 'C_answers': k, 'D_answers': 5 - k,
                               'p_C': p, 'majority_advice': 'C' if k >= 3 else 'D',
                               'unanimous': int(k in (0, 5)),
                               'within_state_pair_agreement': n_agree / 10})
        center_c = sum(counts[(model, s)] for s in STATES if s[4] == 'C')
        center_d = sum(counts[(model, s)] for s in STATES if s[4] == 'D')
        unanimous = sum(k in (0, 5) for k in values)
        model_rows.append({'model': model, 'n_states': 512, 'n_answers': 2560,
                           'C_answers': sum(values), 'p_C': sum(values) / 2560,
                           'p_C_given_current_C': center_c / 1280,
                           'p_C_given_current_D': center_d / 1280,
                           'current_action_retention': (center_c + 1280 - center_d) / 2560,
                           'unanimous_states': unanimous, 'unanimity_fraction': unanimous / 512,
                           'mixed_states': 512 - unanimous,
                           'within_state_pair_agreement': sum(agreeing_pairs) / 5120,
                           'mean_binary_entropy_bits': sum(entropy) / 512,
                           'majority_C_states': sum(k >= 3 for k in values)})
        votes = Counter(values)
        for k in range(6):
            vote_rows.append({'model': model, 'C_answers_out_of_5': k,
                              'observed_p_C': k / 5, 'n_states': votes[k],
                              'state_fraction': votes[k] / 512})
        for center in ('C', 'D'):
            for n in range(5):
                subset = [s for s in STATES if s[4] == center and features(s)['cardinal_c'] == n]
                c = sum(counts[(model, s)] for s in subset)
                neighbor_rows.append({'model': model, 'center': center, 'cardinal_c': n,
                                      'n_states': len(subset), 'n_answers': 5 * len(subset),
                                      'C_answers': c, 'p_C': c / (5 * len(subset)),
                                      'min_state_p_C': min(counts[(model, s)] for s in subset) / 5,
                                      'max_state_p_C': max(counts[(model, s)] for s in subset) / 5})
                for d in range(5):
                    sub = [s for s in subset if features(s)['diagonal_c'] == d]
                    k = sum(counts[(model, s)] for s in sub)
                    context_rows.append({'model': model, 'center': center, 'cardinal_c': n,
                                         'diagonal_c': d, 'n_states': len(sub),
                                         'n_answers': 5 * len(sub), 'C_answers': k,
                                         'p_C': k / (5 * len(sub))})
    model_pairs = []
    for first, second in itertools.combinations(MODELS, 2):
        delta = [counts[(first, s)] - counts[(second, s)] for s in STATES]
        disagreement = sum((counts[(first, s)] >= 3) != (counts[(second, s)] >= 3) for s in STATES)
        model_pairs.append({'first_model': first, 'second_model': second, 'n_states': 512,
                            'mean_delta_p_C_first_minus_second': sum(delta) / 2560,
                            'mean_absolute_p_C_difference': sum(abs(d) for d in delta) / 2560,
                            'first_higher_states': sum(d > 0 for d in delta),
                            'same_p_C_states': sum(d == 0 for d in delta),
                            'second_higher_states': sum(d < 0 for d in delta),
                            'majority_disagreement_states': disagreement,
                            'majority_disagreement_fraction': disagreement / 512,
                            'opposite_unanimous_states': sum(abs(d) == 5 for d in delta)})
    center_pairs = []
    center_details = []
    for model in MODELS:
        deltas = []
        for c_state in STATES:
            if c_state[4] != 'C':
                continue
            d_state = c_state[:4] + 'D' + c_state[5:]
            delta = counts[(model, c_state)] - counts[(model, d_state)]
            deltas.append(delta)
            center_details.append({'model': model, 'neighbor_key': c_state[:4] + c_state[5:],
                                   'grid_key_center_C': c_state, 'grid_key_center_D': d_state,
                                   'p_C_center_C': counts[(model, c_state)] / 5,
                                   'p_C_center_D': counts[(model, d_state)] / 5,
                                   'center_C_minus_D_difference': delta / 5})
        center_pairs.append({'model': model, 'n_paired_neighbor_configurations': len(deltas),
                             'mean_center_C_minus_D_difference': sum(deltas) / (5 * len(deltas)),
                             'positive_pairs': sum(d > 0 for d in deltas),
                             'zero_pairs': sum(d == 0 for d in deltas),
                             'negative_pairs': sum(d < 0 for d in deltas),
                             'minimum_difference': min(deltas) / 5,
                             'maximum_difference': max(deltas) / 5})
    tables = {'replies': clean, 'state_summary': state_rows, 'model_summary': model_rows,
              'neighbor_response': neighbor_rows, 'context_response': context_rows,
              'vote_histogram': vote_rows, 'model_pair_summary': model_pairs,
              'center_pair_summary': center_pairs, 'center_pair_details': center_details}
    for name, rows in tables.items():
        dump_csv(out / f'{name}.csv', rows)

    summaries = {r['model']: r for r in model_rows}
    gpt, ds, qwen = (summaries[m] for m in MODELS)
    examples = []
    example_labels = {
        'CCCCDCCCC': '中心当前为 D，周围八个位置均为 C',
        'CDCDCDCDC': '中心当前为 C，四个直接邻居为 D，四个对角位置为 C',
        'CCCCCCCCC': '九个位置均为 C',
        'DDDDDDDDD': '九个位置均为 D',
    }
    for state, label in example_labels.items():
        examples.append({'grid_key': state, 'grid_rows': '/'.join(state[i:i + 3] for i in (0, 3, 6)),
                         'description_zh': label, **features(state),
                         'models': {m: {'C_answers': counts[(m, state)],
                                       'D_answers': 5 - counts[(m, state)],
                                       'p_C': counts[(m, state)] / 5,
                                       'majority_advice': 'C' if counts[(m, state)] >= 3 else 'D'}
                                    for m in MODELS}})
    all_direct_c_d = [r for r in neighbor_rows if r['center'] == 'D' and r['cardinal_c'] == 4]
    main_points = [
        f"总体建议 C 的频率：Qwen {qwen['p_C']:.2%}、DeepSeek {ds['p_C']:.2%}、GPT {gpt['p_C']:.2%}。该频率按 512 个局面等权汇总，每个局面各有 5 次回答。",
        f"当前为 D 时，Qwen 建议改为 C 的频率为 {qwen['p_C_given_current_D']:.2%}，高于 GPT 的 {gpt['p_C_given_current_D']:.2%} 和 DeepSeek 的 {ds['p_C_given_current_D']:.2%}，这是三者差异最清楚的方向之一。",
        f"DeepSeek 保持当前动作的频率最高，为 {ds['current_action_retention']:.2%}；GPT 为 {gpt['current_action_retention']:.2%}，Qwen 为 {qwen['current_action_retention']:.2%}。DeepSeek 的较高总体 C 频率主要来自对当前 C 的保持，而非频繁推动 D 转为 C。",
        f"当前为 C 时，建议继续 C 的频率为 DeepSeek {ds['p_C_given_current_C']:.2%}、Qwen {qwen['p_C_given_current_C']:.2%}、GPT {gpt['p_C_given_current_C']:.2%}。因此，GPT 的低总体 C 频率还表现为较多 C 转为 D 的建议。",
        f"GPT 的五次回答最一致：{gpt['unanimous_states']}/512 个局面五次全一致（{gpt['unanimity_fraction']:.2%}），DeepSeek 为 {ds['unanimous_states']}/512（{ds['unanimity_fraction']:.2%}），Qwen 为 {qwen['unanimous_states']}/512（{qwen['unanimity_fraction']:.2%}）。一致性反映重复回答的稳定程度，不代表策略更优。",
        f"每个局面 5 次回答共有 10 对；这些回答对的一致率依次为 GPT {gpt['within_state_pair_agreement']:.2%}、DeepSeek {ds['within_state_pair_agreement']:.2%}、Qwen {qwen['within_state_pair_agreement']:.2%}。",
        '四个直接邻居均为 C、中心当前为 D 的 16 个局面共有 80 次回答：' + '；'.join(f"{r['model']} 有 {r['C_answers']}/80 次建议 C（{r['p_C']:.2%}）" for r in all_direct_c_d) + '。',
    ]
    for pair in model_pairs:
        main_points.append(f"{pair['first_model']} 与 {pair['second_model']} 在 {pair['majority_disagreement_states']}/512 个局面的多数建议不同（{pair['majority_disagreement_fraction']:.2%}）；多数建议按 5 次中至少 3 次选择同一动作定义。")
    output = {
        'title_zh': '三个大模型的五次重复回答合并分析',
        'analysis_convention_zh': '每个模型、每个局面的五次回答均视为同一采样设置下的五次等权重复；分析只使用模型、局面和回答，不按采集来源分层。',
        'sample': {'models': list(MODELS), 'states_per_model': 512, 'replies_per_state': 5,
                   'replies_per_model': 2560, 'total_replies': 7680,
                   'current_C_states_per_model': 256, 'current_D_states_per_model': 256},
        'definitions_zh': {
            'C_D': 'C 为合作，D 为背叛；中心格表示当前被询问个体的动作。',
            'p_C': '建议 C 的回答数 ÷ 回答总数。单个局面只可能取 0%、20%、40%、60%、80%、100%。',
            'current_action_retention': '建议动作与中心当前动作相同的回答数 ÷ 2,560；同时包含 C→C 和 D→D。',
            'unanimity_fraction': '五次回答全部相同的局面数 ÷ 512。',
            'within_state_pair_agreement': '每个局面的五次回答形成 10 个不重复回答对，同答对数为 C(k,2)+C(5−k,2)，k 为该局面的 C 回答数；将 512 个局面的同答对数相加后除以 5,120。',
            'majority_advice': '五次中至少三次建议 C，则多数建议为 C；否则为 D。完整回答仍保留，只有需要比较多数建议时才汇总为一个动作。',
            'cardinal_c': '上、下、左、右四个直接邻居中当前为 C 的数量，范围 0–4。',
            'diagonal_c': '左上、右上、左下、右下四个对角位置中当前为 C 的数量，范围 0–4。',
            'center_pair_difference': '固定周围八个位置，只改变中心当前动作，计算 p(C建议|中心C)−p(C建议|中心D)；每个模型有 256 个配对。该差异是描述性比较。',
            'mean_binary_entropy_bits': '对单个局面的 C 频率 p 计算二元熵 −p·log₂(p)−(1−p)·log₂(1−p)，再对 512 个局面取均值。数值越大表示回答更混合；p 为 0 或 1 时记为 0。',
        },
        'model_summary': model_rows, 'model_pair_summary': model_pairs,
        'center_pair_summary': center_pairs, 'main_points_zh': main_points,
        'examples': examples,
        'interpretation_limits_zh': [
            '这是一份五次重复的描述性分析；五次提供的是经验频率，尚不能精确估计单个局面的长期回答概率。',
            '总体结果对所有 512 个二元局面等权，不等于空间博弈实际演化过程中这些局面的出现权重。',
            '建议 C 的频率和回答一致性均不等同于更高收益或更高最终合作水平，后者需要仿真实验验证。',
            '中心动作配对也会改变提示中对应的收益信息，因此不能仅凭配对差异认定因果性惯性或心理锚定。',
        ],
        'source_sha256': source_hash,
    }
    dump_json(out / 'results_zh.json', output)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    # Independent totals from the clean reply rows verify every aggregate path.
    for model in MODELS:
        rows = [r for r in clean if r['model'] == model]
        summary = summaries[model]
        assert len(rows) == 2560 and sum(r['is_c'] for r in rows) == summary['C_answers']
        assert sum(r['recommendation'] == r['center'] for r in rows) / len(rows) == summary['current_action_retention']
        state_subset = [r for r in state_rows if r['model'] == model]
        assert sum(r['C_answers'] for r in state_subset) == summary['C_answers']
        assert sum(r['n_answers'] for r in neighbor_rows if r['model'] == model) == 2560
        assert sum(r['n_answers'] for r in context_rows if r['model'] == model) == 2560
        assert sum(r['n_states'] for r in vote_rows if r['model'] == model) == 512
    for state_row in state_rows:
        replies = groups[(state_row['model'], state_row['grid_key'])]
        pair_matches = sum(a['recommendation'] == b['recommendation']
                           for a, b in itertools.combinations(replies, 2))
        assert pair_matches / 10 == state_row['within_state_pair_agreement']
    validation = {
        'status': 'passed', 'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': source_hash,
        'source_unchanged': True, 'total_rows': len(clean), 'model_state_groups': len(groups),
        'models': list(MODELS), 'rows_per_model': {m: sum(r['model'] == m for r in clean) for m in MODELS},
        'states_per_model': {m: len({r['grid_key'] for r in clean if r['model'] == m}) for m in MODELS},
        'all_states_have_five_unique_repetitions': True, 'complete_binary_state_space': True,
        'features_recomputed_from_grid': True, 'answers_and_numeric_labels_agree': True,
        'all_rows_successful': True, 'pooled_equal_reply_weights': True,
        'aggregation_totals_cross_checked': True,
        'all_15360_answer_pairs_enumerated_against_formula': True,
        'csv_rows': {name + '.csv': len(rows) for name, rows in tables.items()},
        'files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.glob('*')) if path.is_file()},
    }
    dump_json(review / 'stats_validation.json', validation)
    print(json.dumps({'status': 'passed', 'output': str(out), 'model_summary': model_rows,
                      'model_pairs': model_pairs, 'examples': examples}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()

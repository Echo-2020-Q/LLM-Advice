#!/usr/bin/env python3
"""Chinese figures treating all five retained answers as equal repeated calls.

No figure conditions on the collection batch. All figures use the complete
512-state grid with five retained answers per state and model.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle
from matplotlib.text import Text

ROOT = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
if not ROOT.is_dir():
    ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/1101_3LLM_Analysis'
DATA = SOURCE / 'data'
OUT = SOURCE / 'Chinese_Pooled5'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
FONT_CANDIDATES = [Path('/mnt/c/Windows/Fonts/simhei.ttf'), Path('/mnt/c/Windows/Fonts/msyh.ttc')]
FONT = next((p for p in FONT_CANDIDATES if p.is_file()), None)
if FONT is None:
    raise FileNotFoundError('A Chinese font is required: install SimHei or Microsoft YaHei.')
fontManager.addfont(str(FONT))
FONT_NAME = FontProperties(fname=str(FONT)).get_name()
MODELS = ['GPT', 'DeepSeek', 'Qwen']
COLORS = {'GPT': '#4776af', 'DeepSeek': '#cb8138', 'Qwen': '#138b80'}
NAVY = '#17394f'
GRAY = '#566e7e'
plt.rcParams.update({
    'font.family': FONT_NAME, 'font.size': 13, 'axes.labelsize': 13,
    'axes.titlesize': 16, 'axes.titleweight': 'normal', 'text.color': NAVY,
    'axes.labelcolor': NAVY, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#b8c7d1', 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'grid.color': '#dde6eb', 'savefig.facecolor': 'white', 'pdf.fonttype': 42,
    'svg.fonttype': 'path', 'axes.unicode_minus': False,
})
S = pd.read_csv(DATA / 'model_summary.csv').set_index('model').loc[MODELS]
A = pd.read_csv(DATA / 'aligned_states.csv')
R = pd.read_csv(SOURCE / 'merged/combined_answers.csv')
assert len(R) == 7680 and len(A) == 512
assert R.groupby(['model_label', 'grid_key']).size().eq(5).all()
AUDITS = []


def read(name):
    return pd.read_csv(DATA / f'{name}.csv')


def setup(title, subtitle, rows=1, cols=3, size=(13.2, 6.1), top=.75, bottom=.24):
    f, a = plt.subplots(rows, cols, figsize=size, squeeze=False)
    f.suptitle(title, x=.065, y=.975, ha='left', fontsize=23)
    f.text(.065, .90, subtitle, fontsize=12.5, color=GRAY, va='top')
    f.subplots_adjust(left=.08, right=.97, top=top, bottom=bottom, wspace=.36, hspace=.55)
    return f, a


def save(f, name, note):
    f.text(.065, .035, note, fontsize=11, color=GRAY, va='bottom', linespacing=1.5)
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    tick_labels = {
        t for ax in f.axes for axis in (ax.xaxis, ax.yaxis)
        for tick in axis.get_major_ticks() for t in (tick.label1, tick.label2)
    }
    bad = []
    for t in f.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or t in tick_labels:
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > f.bbox.width + .5 or b.y1 > f.bbox.height + .5:
            bad.append(t.get_text())
    assert not bad, (name, bad)
    artifacts = {}
    for ext in ['png', 'pdf', 'svg']:
        dest = FIG / f'{name}.{ext}'
        f.savefig(dest, dpi=230)
        artifacts[ext] = {'bytes': dest.stat().st_size, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    AUDITS.append({'figure': name, 'text_within_canvas': True, 'outputs': artifacts})
    plt.close(f)


def overview():
    f, a = setup('五次回答合并后：三个模型的建议倾向与一致性',
                 '每个模型覆盖相同的 512 个局面；每个局面 5 次等权重复，共 2,560 条回答。')
    specs = [
        ('p_C', '(a) 建议合作的比例', '建议 C 的回答占比（%）'),
        ('current_action_retention', '(b) 保持当前动作的比例', '建议动作与当前动作相同（%）'),
        ('within_state_pair_agreement', '(c) 同一局面的回答一致性', '回答对选择相同动作的比例（%）'),
    ]
    for ax, (col, title, ylabel) in zip(a[0], specs):
        vals = S[col].to_numpy() * 100
        ax.bar(MODELS, vals, color=[COLORS[m] for m in MODELS], width=.65)
        for i, v in enumerate(vals):
            ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontsize=13)
        ax.set(title=title, ylabel=ylabel, ylim=(0, 106), yticks=[0, 25, 50, 75, 100])
        ax.grid(axis='y', alpha=.55)
        ax.set_axisbelow(True)
    save(f, '01_overview',
         '横轴：模型；C 表示合作，D 表示背叛。(a)(b) 的分母均为每个模型的 2,560 条回答。\n'
         '(c) 每个局面的 5 次回答组成 10 个不同回答对，计算动作相同的比例，再对 512 个局面等权平均。\n'
         '一致性反映重复回答的稳定程度；本图不评价建议是否正确，也不表示实际博弈收益。')


def transitions():
    f, a = setup('维持合作与转向合作：模型差异主要体现在哪里？',
                 '按中心玩家当前动作分组；每一行表示该组的建议动作分布。', size=(13.2, 5.9))
    for ax, model in zip(a[0], MODELS):
        pc = S.loc[model, 'p_C_given_current_C']
        pd_ = S.loc[model, 'p_C_given_current_D']
        vals = np.array([[pc, 1 - pc], [pd_, 1 - pd_]])
        ax.imshow(vals, vmin=0, vmax=1, cmap='YlGnBu')
        ax.set(xticks=[0, 1], xticklabels=['建议 C', '建议 D'], yticks=[0, 1],
               yticklabels=['当前 C', '当前 D'], title=model, xlabel='建议的下一步动作', ylabel='中心当前动作')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{vals[i, j] * 100:.2f}%\n{round(vals[i, j] * 1280):,} / 1,280',
                        ha='center', va='center', fontsize=13, color='white' if vals[i, j] > .62 else NAVY)
    save(f, '02_transitions',
         '纵轴：中心玩家当前为 C 或 D；横轴：模型建议 C 或 D。每行的百分比之和为 100%。\n'
         '每个模型每行的分母为 256 个局面 × 5 次回答 = 1,280；颜色深浅与建议比例对应。\n'
         '这里统计的是模型给出的建议，不能直接等同于模拟运行中的实际动作转移。')


def neighbors():
    n = read('neighbor_response')
    f, a = setup('直接邻居的合作数量如何影响建议？',
                 '直接邻居指上、下、左、右四个位置；每组汇总所有角点配置和位置排列。', cols=2)
    for ax, center in zip(a[0], ['C', 'D']):
        for model in MODELS:
            d = n[(n.model == model) & (n.center == center)].sort_values('cardinal_c')
            ax.plot(d.cardinal_c, 100 * d.p_C, marker='o', lw=2.2, color=COLORS[model], label=model)
        ax.set(title=f'中心当前为 {center}', xlabel='直接邻居中的合作人数', ylabel='建议 C 的回答占比（%）',
               xticks=range(5), ylim=(-3, 104), yticks=[0, 25, 50, 75, 100])
        ax.grid(alpha=.5)
    a[0, 0].legend(fontsize=12)
    save(f, '03_neighbors',
         '横轴：4 个直接邻居中选择 C 的人数（0—4）；纵轴：该组建议 C 的回答数 / 该组全部回答数。\n'
         '对每个模型、每种中心动作，横轴 0、1、2、3、4 的分母依次为 80、320、480、320、80 条回答。\n'
         '连线连接各组的经验比例，不表示连续函数或模拟结果；所有回答均来自每个局面的 5 次等权重复。')


def context():
    c = read('context_response')
    f, a = setup('同时观察直接邻居和对角邻居：合作建议的局面分布',
                 '上排中心为 C，下排中心为 D；每格汇总对应人数下的全部位置排列。',
                 rows=2, cols=3, size=(13.2, 8.8), top=.81, bottom=.19)
    f.subplots_adjust(right=.90, wspace=.37, hspace=.56)
    for i, center in enumerate(['C', 'D']):
        for j, model in enumerate(MODELS):
            ax = a[i, j]
            d = c[(c.model == model) & (c.center == center)]
            v = d.pivot(index='diagonal_c', columns='cardinal_c', values='p_C').reindex(
                index=range(5), columns=range(5)).to_numpy() * 100
            im = ax.imshow(v, origin='lower', vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
            ax.set(title=f'{model} | 中心 {center}', xticks=range(5), yticks=range(5),
                   xlabel='直接邻居中的合作人数', ylabel='对角邻居中的合作人数')
            for r in range(5):
                for col in range(5):
                    ax.text(col, r, f'{v[r, col]:.0f}', ha='center', va='center', fontsize=12,
                            color='white' if v[r, col] > 62 else NAVY)
    cax = f.add_axes([.925, .26, .015, .48])
    f.colorbar(im, cax=cax, label='建议 C 的回答占比（%）')
    save(f, '04_context',
         '横轴：上、下、左、右的 C 人数；纵轴：四个角点的 C 人数。颜色和数字为组内建议 C 的百分比，数字取整。\n'
         '每个模型、每种中心动作下，格子 (k, d) 的分母 = 5 × C(4,k) × C(4,d)，范围为 5—180 条回答。\n'
         'C(4,k) 为组合数；对角邻居包含在提示词局面中，但不与中心玩家直接计算收益。')


def consistency():
    v = read('vote_histogram')
    f, a = setup('五次重复能看到什么：完全一致与混合回答',
                 '对每个完全相同的局面，统计 5 次回答中有多少次建议 C。', size=(13.2, 6.4))
    for ax, model in zip(a[0], MODELS):
        d = v[v.model.eq(model)].sort_values('C_answers_out_of_5')
        vals = d.fraction_of_512_states.to_numpy() * 100
        ax.bar(range(6), vals, color=COLORS[model], alpha=.9)
        for k, (value, count) in enumerate(zip(vals, d.n_states)):
            ax.text(k, value + 1, f'{count}', ha='center', fontsize=12)
        ax.set(title=f'{model}\n完全一致：{int(S.loc[model, "unanimous_states"])}/512（{S.loc[model, "unanimity_fraction"] * 100:.1f}%）',
               xlabel='5 次回答中建议 C 的次数 k', ylabel='占全部 512 个局面的比例（%）',
               xticks=range(6), yticks=[0, 20, 40, 60], ylim=(0, 66))
        ax.grid(axis='y', alpha=.5)
        ax.set_axisbelow(True)
    save(f, '05_consistency',
         '横轴 k=0 表示 5 次全为 D，k=5 表示 5 次全为 C；k=1—4 表示同一局面下出现不同回答。\n'
         '纵轴分母为 512 个局面，柱顶数字为局面数量；“完全一致”合并 k=0 与 k=5 两类。\n'
         '一次局面的 5 次观测仍然有限；观察到全 C 或全 D，并不意味着真实回答概率必然等于 1 或 0。')


def model_agreement():
    f, a = setup('同一个局面，不同模型的合作建议有多接近？',
                 '将模型两两配对，在相同局面上比较各自 5 次回答中的 C 比例。', size=(13.2, 6.5), bottom=.25)
    pairs = [('GPT', 'DeepSeek'), ('GPT', 'Qwen'), ('DeepSeek', 'Qwen')]
    matrices = []
    for x, y in pairs:
        z = np.zeros((6, 6), dtype=int)
        for k, ell in zip(A[f'{x}_C'], A[f'{y}_C']):
            z[int(ell), int(k)] += 1
        assert z.sum() == 512
        matrices.append(z)
    vmax = max(z.max() for z in matrices)
    f.subplots_adjust(right=.90, wspace=.43)
    for ax, (x, y), z in zip(a[0], pairs, matrices):
        im = ax.imshow(z, origin='lower', vmin=0, vmax=vmax, cmap='Blues', aspect='equal')
        ax.plot([-.5, 5.5], [-.5, 5.5], color='#8e9ba4', ls='--', lw=.8)
        dis = ((A[f'{x}_C'] >= 3) != (A[f'{y}_C'] >= 3)).sum()
        ax.set(title=f'{x} / {y}\n多数建议不同：{dis}/512', xticks=range(6),
               xticklabels=[0, 20, 40, 60, 80, 100], yticks=range(6),
               yticklabels=[0, 20, 40, 60, 80, 100],
               xlabel=f'{x}：建议 C 的比例（%）', ylabel=f'{y}：建议 C 的比例（%）')
        for r in range(6):
            for col in range(6):
                if z[r, col]:
                    ax.text(col, r, str(z[r, col]), ha='center', va='center', fontsize=12,
                            color='white' if z[r, col] > vmax * .55 else NAVY)
    cax = f.add_axes([.92, .30, .014, .40])
    f.colorbar(im, cax=cax, label='局面数量')
    save(f, '06_model_agreement',
         '横纵轴：两个模型对同一局面的经验 C 比例，取值为 0%、20%、40%、60%、80%、100%。\n'
         '数字和颜色表示局面数量；每张子图合计 512 个局面，空白格为 0；虚线表示两个模型的 C 比例相等。\n'
         '“多数建议”以 5 次中至少 3 次为准；多数建议不同与回答比例不同是两个统计量。')


def examples():
    cases = [('CCCCCCCCC', '全体合作'), ('DDDDDDDDD', '全体背叛'),
             ('CCCCDCCCC', '中心 D，周围全部为 C'), ('CDCDCDCDC', '中心 C，直接邻居全部为 D')]
    f = plt.figure(figsize=(13.2, 8.5))
    f.suptitle('四个具体局面：五次回答的合并结果', x=.065, y=.975, ha='left', fontsize=23)
    f.text(.065, .905, '显示完整的 3 × 3 局面；橙色边框标出中心玩家，右侧统计 5 次回答的 C / D 数量。',
           fontsize=12.5, color=GRAY)
    grid = f.add_gridspec(2, 2, left=.07, right=.97, top=.84, bottom=.18, wspace=.19, hspace=.36)
    for z, (key, title) in enumerate(cases):
        ax = f.add_subplot(grid[z // 2, z % 2])
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5.2)
        ax.set_title(title, loc='left', fontsize=17)
        for i, ch in enumerate(key):
            row, col = divmod(i, 3)
            x, y = .1 + col * .83, 3.9 - row * .83
            ax.add_patch(Rectangle((x, y), .79, .79, facecolor='#15978b' if ch == 'C' else '#e7edf1', edgecolor='white'))
            ax.text(x + .395, y + .395, ch, ha='center', va='center', color='white' if ch == 'C' else NAVY, fontsize=16)
        ax.add_patch(Rectangle((.93, 3.07), .79, .79, fill=False, lw=2.5, edgecolor='#e5a342'))
        ax.text(3.3, 4.65, '模型', fontsize=12, color=GRAY)
        ax.text(5.9, 4.65, '5 次回答统计', fontsize=12, color=GRAY)
        ax.text(9.8, 4.65, 'C 比例', fontsize=12, color=GRAY, ha='right')
        for i, model in enumerate(MODELS):
            d = R[(R.model_label == model) & (R.grid_key == key)]
            assert len(d) == 5
            n = int(d.is_c.sum())
            y = 3.76 - i * .93
            ax.text(3.25, y, model, fontsize=14, color=COLORS[model])
            ax.text(5.9, y, f'C: {n}   D: {5 - n}', fontsize=14)
            ax.text(9.8, y, f'{n / 5:.0%}', ha='right', fontsize=14)
        ax.text(.1, .40, ' / '.join(key[i:i + 3] for i in [0, 3, 6]), fontsize=11, color=GRAY)
    save(f, '07_examples',
         '左侧每一格表示一个玩家的当前动作；右侧 C、D 数量之和均为 5，C 比例 = C 回答数 / 5。\n'
         '四个局面用于说明典型一致与分歧，并非全部 512 个局面的代表性抽样。\n'
         '每个模型的 5 次回答按同一设置下的等权重复汇总；C / D 表示动作，不表示更优或更差的结果。')


def center_effect():
    e = read('center_pair_effects')
    q = read('center_pair_summary').set_index('model')
    f, a = setup('周围八格不变时，中心当前动作与建议的关系',
                 '将中心为 C 与中心为 D 的局面逐一配对，共得到 256 对；每个局面均使用 5 次回答。',
                 size=(13.2, 6.5), bottom=.25)
    xs = np.arange(-5, 6) * 20
    for ax, model in zip(a[0], MODELS):
        v = np.rint(e.loc[e.model == model, 'center_C_minus_D_difference'] * 5).astype(int).value_counts()
        counts = np.array([v.get(k, 0) for k in range(-5, 6)])
        assert counts.sum() == 256
        ax.bar(xs, counts / 256 * 100, width=16, color=COLORS[model])
        ax.axvline(0, color=GRAY, lw=1, ls='--')
        ax.set(title=f'{model}\n平均差值：{q.loc[model, "mean_center_C_minus_D_difference"] * 100:+.2f} 个百分点',
               xlabel='建议 C 的比例之差（百分点）', ylabel='占 256 对局面的比例（%）',
               xticks=[-100, -50, 0, 50, 100], xlim=(-113, 113), ylim=(0, 74), yticks=[0, 20, 40, 60])
        ax.grid(axis='y', alpha=.5)
        ax.set_axisbelow(True)
    save(f, '08_center_effect',
         '横轴 = 100 × [中心为 C 时的 C 回答数 / 5 - 中心为 D 时的 C 回答数 / 5]，单位为百分点。\n'
         '纵轴 = 具有该差值的局面对数 / 256；正值表示中心原本合作时，模型更常建议合作。\n'
         '改变中心动作也可能改变提示词中的中心收益；因此，该差值不能单独解释为当前动作的因果效应。')


if __name__ == '__main__':
    overview()
    transitions()
    neighbors()
    context()
    consistency()
    model_agreement()
    examples()
    center_effect()
    (OUT / 'review').mkdir(exist_ok=True)
    record = {
        'method': 'All five replies are pooled and equally weighted; no batch-stratified analysis.',
        'language': 'Chinese', 'font_file': str(FONT), 'font_family': FONT_NAME,
        'models': MODELS, 'states_per_model': 512, 'answers_per_state': 5,
        'answers_per_model': 2560, 'total_answers': 7680,
        'source_csv_sha256': hashlib.sha256((SOURCE / 'merged/combined_answers.csv').read_bytes()).hexdigest(),
        'figures': AUDITS,
    }
    (OUT / 'review/figure_validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Generated 8 Chinese figures (PNG / PDF / SVG) in {FIG}')

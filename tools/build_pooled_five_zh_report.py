"""Build the Chinese report for five equally weighted replies per model/state."""
from pathlib import Path
from xml.sax.saxutils import escape
import csv
import hashlib
import json

from PIL import Image as PILImage, ImageDraw
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/1101_3LLM_Analysis/Chinese_Pooled5'
DATA, FIG = OUT / 'data', OUT / 'figures'
MODELS = ['GPT', 'DeepSeek', 'Qwen']
WIDTH = A4[0] - 84
NAVY, MUTED, TEAL = '#17394f', '#526b7c', '#087f82'


def read(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pct(x):
    return f'{float(x) * 100:.2f}%'


def block(kind, text='', **kwargs):
    return dict(kind=kind, text=text, **kwargs)


def content():
    summary = {r['model']: r for r in read('model_summary.csv')}
    pair = read('model_pair_summary.csv')
    for m in MODELS:
        assert int(summary[m]['n_states']) == 512 and int(summary[m]['n_answers']) == 2560
    overview = [['统计指标', *MODELS]]
    for label, key in [
        ('总体合作建议率', 'p_C'),
        ('当前 C 时建议 C', 'p_C_given_current_C'),
        ('当前 D 时建议 C', 'p_C_given_current_D'),
        ('建议保持当前行动', 'current_action_retention'),
    ]:
        overview.append([label, *[pct(summary[m][key]) for m in MODELS]])
    consistency = [['重复回答的一致性', *MODELS]]
    for label, key in [('五次完全一致的局面比例', 'unanimity_fraction'), ('同一局面内两两一致率', 'within_state_pair_agreement')]:
        consistency.append([label, *[pct(summary[m][key]) for m in MODELS]])
    consistency.append(['五次回答中同时出现 C 和 D 的局面数', *[summary[m]['mixed_states'] for m in MODELS]])
    pair_table = [['模型对', '多数建议不同的局面', '平均绝对频率差']]
    for r in pair:
        pair_table.append([
            r['first_model'] + ' / ' + r['second_model'],
            f"{r['majority_disagreement_states']}/512（{pct(r['majority_disagreement_fraction'])}）",
            f"{100 * float(r['mean_absolute_p_C_difference']):.2f} 个百分点",
        ])
    B = block
    return [
        [
            B('kicker', '三模型建议比较 · 五次重复统一分析'),
            B('title', 'GPT、DeepSeek 与 Qwen\n五次回答的中文分析'),
            B('meta', '2026 年 11 月 2 日｜512 个局面 × 5 次回答 × 3 个模型'),
            B('lead', '本报告将每个模型对同一局面的五次回答，视为同一设置下的五次等权重复采样。对三模型使用相同的 512 个局面进行比较，每个模型共 2,560 条回答，合计 7,680 条。'),
            B('figure', file='01_overview.png'),
            B('caption', '图 1｜横轴为模型，三个面板的纵轴依次为合作建议率、保持当前行动的比例、同一局面内的两两一致率。前两项以每模型 2,560 条回答为分母；第三项先计算每个局面五次回答中 10 对回答的一致比例，再对 512 个局面取平均。'),
            B('table', rows=overview, widths=[.43, .19, .19, .19]),
            B('subhead', '最值得汇报的三个差异'),
            B('body', 'Qwen 更常建议当前背叛者转向合作；DeepSeek 更常建议保持当前行动，尤其很少建议背叛者改变；GPT 的五次回答最一致，但总体上更少建议合作。这三点描述的是不同维度，不能仅靠一个总体合作比例概括。'),
            B('note', 'C 表示合作，D 表示背叛。这里统计的是模型给出的行动建议；图中比例不等于空间博弈实际达到的合作水平，也不直接代表建议质量或最终收益。'),
        ],
        [
            B('kicker', '01｜合作建议来自哪里'),
            B('heading', '区分维持合作与建议转向合作'),
            B('figure', file='02_transitions.png'),
            B('caption', '图 2｜纵轴为中心玩家当前行动，横轴为建议行动。每行包含 256 个局面 × 5 次＝1,280 条回答，行内两个百分比之和为 100%。C→C 表示建议已有合作者继续合作；D→C 表示建议当前背叛者转向合作。'),
            B('figure', file='03_neighbors.png'),
            B('caption', '图 3｜横轴为上、下、左、右四个直接邻居中合作者的数量；纵轴为该组回答中的合作建议比例。固定中心行动后，横轴 0、1、2、3、4 对应的分母分别为 80、320、480、320、80 条回答。折线连接分组观察值。'),
            B('body', 'DeepSeek 的总体合作建议率高于 GPT，主要体现为更常维持已有合作：C→C 为 76.17%，而 GPT 为 48.20%。但对于当前 D，DeepSeek 的 D→C 只有 2.27%，低于 GPT 的 7.03%，更远低于 Qwen 的 48.05%。'),
            B('body', '当中心当前为 D、四个直接邻居均为 C 时，Qwen 有 75/80 条回答建议合作（93.75%），GPT 为 14/80（17.50%），DeepSeek 为 3/80（3.75%）。这是三模型差异最容易在组会上解释的情景之一。'),
        ],
        [
            B('kicker', '02｜周围空间结构'),
            B('heading', '邻居数量相同，角落信息也有区别'),
            B('figure', file='04_context.png'),
            B('caption', '图 4｜每个面板固定模型与中心行动。横轴为四个直接邻居中的合作者数量，纵轴为四个对角邻居中的合作者数量；颜色与数字表示合作建议率（%）。数字取整，精确值见数据表。'),
            B('subhead', '先看每格的分母'),
            B('body', '一格中的回答数为 5 × 组合数(4,直接合作数) × 组合数(4,对角合作数)。例如直接与对角均有 2 名合作者时，共 36 个局面、180 条回答；两种数量都取极端值 0 或 4 时，仅有 1 个局面、5 条回答。因此不同格子的样本量不同。'),
            B('subhead', 'Qwen 对合作环境的建议变化更突出'),
            B('body', '固定中心为 D、直接合作邻居数为 2，Qwen 在没有合作对角邻居时的合作建议率为 6.67%，而四个对角邻居均合作时为 73.33%（两端各 30 条回答）。这说明其建议与可见的局部合作环境有关。'),
            B('note', '四个对角邻居不直接与中心玩家计算收益，但会出现在输入信息中。这些分组比较展示建议与空间结构的关联，不能单独据此确定模型采用了哪种推理机制。'),
        ],
        [
            B('kicker', '03｜五次回答的一致性'),
            B('heading', 'GPT 最一致，Qwen 更常出现混合回答'),
            B('figure', file='05_consistency.png'),
            B('caption', '图 5｜横轴 k 是同一局面五次回答中建议 C 的次数，范围为 0 到 5；纵轴为对应局面数占全部 512 个局面的比例。柱顶数字是局面数。k＝0 或 5 表示五次完全一致，k＝1～4 表示同一局面中同时出现 C 和 D。'),
            B('table', rows=consistency, widths=[.46, .18, .18, .18]),
            B('body', 'GPT 有 382/512 个局面五次完全一致，DeepSeek 为 355/512，Qwen 为 328/512。混合回答局面分别为 130、157、184 个。按这一观察指标，GPT 的回答最稳定，Qwen 对同一局面更常给出不同建议。'),
            B('subhead', '两种一致性指标含义不同'),
            B('body', '“五次完全一致”只判断是否全部相同。“两两一致率”则使用五次回答组成的 10 个不同无序对：5∶0 的分布为 100%，4∶1 为 60%，3∶2 为 40%，再对局面取平均。它保留了混合程度的信息。'),
            B('note', '一致性高不等于建议更好。五次重复能揭示回答波动；单个局面的合作频率只能取 0%、20%、40%、60%、80%、100%，因此不宜对相邻档位作过细解释。'),
        ],
        [
            B('kicker', '04｜相同局面下的模型差异'),
            B('heading', 'GPT 与 Qwen 的多数建议分歧最多'),
            B('figure', file='06_model_agreement.png'),
            B('caption', '图 6｜每个面板比较一对模型。横纵轴分别为两个模型在同一局面中五次回答的合作频率；格内数字和颜色为落入该组合的局面数，每个面板合计 512 个局面。对角线上的两个模型具有相同的观察合作频率。'),
            B('table', rows=pair_table, widths=[.29, .40, .31]),
            B('body', '把五次中至少三次建议 C 定义为“多数建议 C”，GPT 与 Qwen 在 183 个局面中多数建议不同，占 35.74%；DeepSeek 与 Qwen 为 167 个，占 32.62%；DeepSeek 与 GPT 为 104 个，占 20.31%。'),
            B('body', '多数建议与频率差衡量不同问题。DeepSeek 与 Qwen 的平均绝对合作频率差为 33.05 个百分点，略高于 GPT 与 Qwen 的 32.46 个百分点，但它们的多数建议分歧数更少。不能将两种排序视为矛盾。'),
            B('note', '“平均绝对频率差”先对同一局面的两个合作频率求绝对差，再对 512 个局面取平均；它不是两个总体合作建议率之差。多数建议只用于比较，原始五次回答全部保留。'),
        ],
        [
            B('kicker', '05｜可直接展示的局面案例'),
            B('heading', '同样的输入，可以得到相反的建议'),
            B('figure', file='07_examples.png'),
            B('caption', '图 7｜每个 3×3 方格给出一个完整局面，边框标出中心玩家；右侧列出三个模型五次回答的结果。C 为合作，D 为背叛。局面按从左到右、从上到下的顺序编码。'),
            B('body', '全 C 局面中，三个模型均为 5 次 C；全 D 局面中，三个模型均为 5 次 D。分歧主要出现在合作与背叛混合的局部结构中。'),
            B('body', '局面 CCC / CDC / CCC：中心为 D，周围八个玩家全为 C。GPT 有 2/5 次建议 C，DeepSeek 为 0/5，Qwen 为 5/5。这个案例直接体现 Qwen 更常建议被合作环境包围的背叛者转向合作。'),
            B('body', '局面 CDC / DCD / CDC：中心为 C，四个直接邻居均为 D，四个角落均为 C。GPT 与 Qwen 都为 0/5 次建议 C，而 DeepSeek 为 5/5。这说明模型差异不能简单概括成“谁在所有局面中都更合作”。'),
            B('note', '以上案例用于展示具体行为差异；总体结论仍以全部 512 个局面的统计为依据。交互报告可查看其他任意局面的五次回答。'),
        ],
        [
            B('kicker', '06｜固定周围八格的比较'),
            B('heading', 'DeepSeek 的建议与当前行动关联最强'),
            B('figure', file='08_center_effect.png'),
            B('caption', '图 8｜将周围八格完全相同、仅中心行动为 C 或 D 的两个局面配为一对，共 256 对。横轴为“中心为 C 时的合作频率－中心为 D 时的合作频率”，单位是百分点；纵轴为相应配对数占 256 对的比例。'),
            B('table', rows=[['配对统计', *MODELS], ['平均合作频率差', '+41.17 个百分点', '+73.91 个百分点', '+18.44 个百分点'], ['正值 / 零 / 负值的配对数', '157 / 94 / 5', '248 / 8 / 0', '125 / 125 / 6']], widths=[.37, .21, .21, .21]),
            B('body', '在相同的周围环境下，中心当前为 C 时，DeepSeek 比中心为 D 时平均多给出 73.91 个百分点的合作建议；GPT 为 41.17，Qwen 为 18.44。这个结果与 DeepSeek 较高的当前行动保持率一致。'),
            B('body', '对 DeepSeek，256 对中有 248 对在中心为 C 时给出更多合作建议，8 对相同，没有负差值。Qwen 有 125 对正差值、125 对相同、6 对负差值；它的建议相对较少随中心当前行动改变。'),
            B('note', '输入中，改变中心行动也会改变相应的当前收益描述。因此这里是控制周围八格后的描述性比较，不能把差异全部归因于某一种心理偏好或模型内部机制。'),
        ],
        [
            B('kicker', '07｜结论与后续实验'),
            B('heading', '从建议差异走向空间博弈检验'),
            B('subhead', '1. Qwen：更常建议背叛者转向合作'),
            B('body', '总体合作建议率 57.27%，当前 D 时建议 C 的比例为 48.05%。其最突出的特点，是在直接邻居较合作时也会向当前背叛者建议合作。后续可观察这类建议是否推动合作区域向外扩张。'),
            B('subhead', '2. DeepSeek：更常维持当前行动'),
            B('body', '保持当前行动的比例为 86.95%，当前 C 时继续建议 C 的比例为 76.17%，当前 D 时建议 C 仅为 2.27%。后续可观察它是否主要维持已有合作结构，以及初始合作水平是否明显影响结果。'),
            B('subhead', '3. GPT：回答较稳定，总体合作建议较少'),
            B('body', '总体合作建议率为 27.62%，五次完全一致的局面占 74.61%。当前中心为 C、直接合作邻居从 1 个增加到 2 个和 3 个时，合作建议率依次为 1.25%、45.21%、98.75%，表现出较明显的邻居数量分组差异。'),
            B('subhead', '建议优先验证的三个问题'),
            B('body', '在相同仿真参数和配对随机种子下，比较三模型的长期合作水平与收敛过程；分别统计实际被访问的局面中 C→C 和 D→C 建议及采纳情况；结合个人信任，检查不同建议是否被采纳、是否带来更高收益。以上是待检验的问题，而不是本报告已经得到的仿真结论。'),
            B('subhead', '统一的统计口径'),
            B('body', '每模型、每局面保留五条等权回答。单局面合作频率＝C 次数/5；分组合作建议率＝组内 C 总次数/组内总回答数；总体为 512 局面等权。每局面多数建议为五次中至少三次所选行动。五次内的 10 对回答共享原始回答，不应当作 10 次独立采样。'),
            B('subhead', '本报告能回答什么'),
            B('body', '这里可以比较建议偏向、对局面特征的响应和五次回答的一致性。512 个局面是完整枚举，实际仿真访问它们的频率可能不同。因此建议频率不能直接预测最终合作率；若要估计单个局面的响应概率，提高重复次数比仅依赖 5 次更合适。'),
            B('note', '配套交互 HTML 可筛选 512 个局面并查看全部五次回答；data/ 提供统一口径重新计算的数据表；figures/ 提供 8 张中文图的 PNG、PDF、SVG。此报告没有运行新的模型询问或空间博弈仿真。'),
        ],
    ]


def register_fonts():
    paths = ['/mnt/c/Windows/Fonts/msyh.ttc', '/mnt/c/Windows/Fonts/msyhbd.ttc']
    for name, path in zip(['ReportZH', 'ReportZH-Bold'], paths):
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily('ReportZH', normal='ReportZH', bold='ReportZH-Bold')


def styles():
    specs = {'body': (10, 16, 9), 'lead': (10.5, 17, 12), 'note': (8.7, 13.8, 8),
             'caption': (8.7, 13.5, 11), 'meta': (9, 14, 12), 'table': (8.8, 14, 0),
             'kicker': (9, 14, 9), 'title': (25, 34, 13), 'heading': (20, 29, 12), 'subhead': (11.5, 18, 7)}
    result = {}
    for name, (size, leading, after) in specs.items():
        bold = name in ('kicker', 'title', 'heading', 'subhead')
        result[name] = ParagraphStyle(name, fontName='ReportZH-Bold' if bold else 'ReportZH',
            fontSize=size, leading=leading, spaceAfter=after, wordWrap='CJK',
            textColor=colors.HexColor(TEAL if name == 'kicker' else MUTED if name in ('note', 'caption', 'meta') else NAVY),
            keepWithNext=bold)
    return result


def rich(text):
    return escape(str(text)).replace('\n', '<br/>')


def build(pages):
    register_fonts()
    style = styles()
    story, heights = [], []
    for number, page in enumerate(pages):
        if number:
            story.append(PageBreak())
        height = 0
        for b in page:
            if b['kind'] == 'figure':
                path = FIG / b['file']
                with PILImage.open(path) as im:
                    w, h = im.size
                objects = [Image(str(path), width=WIDTH, height=WIDTH*h/w), Spacer(1, 3)]
            elif b['kind'] == 'table':
                rows = [[Paragraph(('<b>'+rich(x)+'</b>') if i == 0 else rich(x), style['table']) for x in row] for i, row in enumerate(b['rows'])]
                table = Table(rows, colWidths=[WIDTH*f for f in b['widths']], hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eaf3f5')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('LINEBELOW', (0, 0), (-1, -1), .35, colors.HexColor('#d8e3e9'))]))
                objects = [table, Spacer(1, 11)]
            else:
                objects = [Paragraph(rich(b['text']), style[b['kind']])]
            for obj in objects:
                _, h = obj.wrap(WIDTH, 10000)
                height += h + obj.getSpaceBefore() + obj.getSpaceAfter()
            story.extend(objects)
        heights.append(round(height, 2))
    def chrome(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d3e0e7'))
        for y in [A4[1]-31, 30]:
            canvas.line(42, y, A4[0]-42, y)
        canvas.setFont('ReportZH', 7.6)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(42, A4[1]-23, '三模型建议比较｜每个局面五次等权回答')
        canvas.drawRightString(A4[0]-42, A4[1]-23, '2026.11.02')
        canvas.drawString(42, 18, '512 个局面 · 7,680 条回答 · 建议行为的描述性分析')
        canvas.drawRightString(A4[0]-42, 18, f'{doc.page} / {len(pages)}')
        canvas.restoreState()
    target = OUT / '三模型五次回答_中文分析.pdf'
    doc = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=48, bottomMargin=43, title='三模型五次回答：中文分析', author='LLM Advice Research')
    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    pdf = PdfReader(target)
    assert len(pdf.pages) == len(pages), (len(pdf.pages), heights)
    for i, (p, blocks) in enumerate(zip(pdf.pages, pages)):
        title = next(b['text'] for b in blocks if b['kind'] in ('title', 'heading'))
        text = p.extract_text()
        assert ''.join(title.split()) in ''.join(text.split()), (i, title)
        assert '\ufffd' not in text and '\x00' not in text
    return target, heights


def render(path):
    import pypdfium2 as pdfium
    review = OUT / 'review'
    review.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(path))
    sheet = PILImage.new('RGB', (1000, 1420), '#e4ecf0')
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=1.6)
        im = bitmap.to_pil()
        im.save(review / f'中文报告_{i+1:02d}.png')
        thumb = im.copy()
        thumb.thumbnail((240, 340))
        x, y = (i%4)*250+5, (i//4)*710+20
        sheet.paste(thumb, (x, y))
        ImageDraw.Draw(sheet).text((x, y-15), f'Page {i+1}', fill=NAVY)
        bitmap.close()
        page.close()
    doc.close()
    sheet.save(review / '中文报告_总览.png')


def main():
    pages = content()
    OUT.mkdir(parents=True, exist_ok=True)
    text = ['三模型五次回答：中文分析', '2026 年 11 月 2 日', '']
    for i, p in enumerate(pages, 1):
        text.extend([f'第 {i} 页', ''])
        for b in p:
            if b['kind'] == 'figure':
                text.append('配图：figures/'+b['file'])
            elif b['kind'] == 'table':
                text.extend(' | '.join(row) for row in b['rows'])
            else:
                text.append(b['text'])
            text.append('')
    (OUT/'中文分析.txt').write_text('\n'.join(text), encoding='utf-8')
    (OUT/'report_content_zh.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    target, heights = build(pages)
    render(target)
    metrics = dict(pages=len(pages), figures=8, content_heights_pt=heights,
                   frame_height_pt=round(A4[1]-48-43-12, 2),
                   text_extraction_verified=True, rendered_pages=8,
                   pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (OUT/'review/report_validation.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

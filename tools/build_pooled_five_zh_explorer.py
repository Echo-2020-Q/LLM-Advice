"""Build the offline Chinese report treating five replies as equally weighted repeats."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT=Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
SOURCE=ROOT/'reports/1101_3LLM_Analysis'
REPORT=SOURCE/'Chinese_Pooled5'
MODELS=['GPT','DeepSeek','Qwen']
FIGURES=[
('01_overview','整体建议倾向与当前动作保持',
 '三个子图的横轴均为模型，纵轴依次是总体 C 建议百分比、保持当前动作百分比、同局面回答对一致百分比。前两项分母为每模型 2,560 条回答；第三项先计算同局面十对回答的一致率，再对 512 个局面等权平均。',
 'Qwen 给出 C 的比例最高；DeepSeek 更常保持输入中的当前动作；GPT 的重复回答最一致。这是三种不同维度，不能将其合并为单一的“最好模型”。'),
('02_transitions','当前是 C 或 D 时，模型分别建议什么',
 '每个模型一张热图，横轴为建议动作 C/D，纵轴为中心当前动作 C/D。格内颜色与数字表示该当前动作下的建议百分比及回答数；每一行含 256 个局面、1,280 条回答，同一行的两个百分比相加为 100%。',
 'DeepSeek 的 C→C 比例为 76.17%，而 D→C 仅为 2.27%；Qwen 的 D→C 达到 48.05%。因此 DeepSeek 总体 C 比例高于 GPT，并不意味着它更愿意将 D 转为 C。'),
('03_neighbors','直接邻居中的合作者数量如何对应建议',
 '横轴是上下左右四个直接邻居中 C 的数量（0–4）；纵轴是该组回答中 C 的百分比；按当前中心动作分别比较。每个模型、每种中心动作下，各横坐标含 16、64、96、64、16 个局面，每局面贡献五条回答。',
 '关注中心当前为 D 的曲线，可直接比较模型是否会在周围更合作时建议 D→C。四个直接邻居全为 C 时，Qwen 有 75/80 次建议 C（93.75%），GPT 为 14/80（17.50%），DeepSeek 为 3/80（3.75%）。'),
('04_context','直接邻居与对角邻居的联合环境',
 '热图横轴为四个直接邻居中的 C 数量，纵轴为四个对角邻居中的 C 数量，均为 0–4；不同子图对应模型及中心 C/D。颜色和数字表示组内 C 建议百分比，同一格内所有完整局面及其五次回答等权汇总。',
 '对角位置也出现在提示中，但不直接贡献中心当前收益。这里展示完整局部环境与建议的关系；不能仅由热图认定某个位置对模型具有因果影响。'),
('05_consistency','五次重复回答有多稳定',
 '横轴为单个局面五次回答中 C 的次数 k，取值 0–5；纵轴为对应局面占全部 512 个局面的百分比，柱顶数字为局面数，全部柱高相加为 100%。k=0 或 5 表示五次完全一致；k=1–4 表示出现混合回答。',
 'GPT 有 382/512 个局面五次完全一致（74.61%），DeepSeek 为 355（69.34%），Qwen 为 328（64.06%）。一致性只描述重复稳定程度，不代表建议质量或长期收益。'),
('06_model_agreement','不同模型在相同局面上的差异',
 '每个子图比较两个模型：横纵轴分别为它们在同一局面的 C 建议比例（0%、20%、…、100%），颜色和数字表示局面数量，每图合计 512。虚线是两模型 C 比例相等的位置；标题中的“多数建议不同”以各模型至少三票的多数动作判断。',
 '模型总体 C 比例接近，仍可能在具体局面上给出相反建议；反过来，多数动作相同也不代表五次回答分布相同。因此需要同时查看多数一致率和逐局面的 C 比例差。'),
('07_examples','用具体局面理解模型差别',
 '每个示例展示完整 3×3 局面，中心为被询问个体；C 是合作，D 是背叛。表格列出每个模型五次回答中的 C 次数、D 次数和 C 比例；例如 2/5 C 即 40%。示例是说明性选择，不代表整体局面分布。',
 '在 CCC/CDC/CCC（中心 D、周围八格 C）中，GPT 给出 2 次 C，DeepSeek 0 次，Qwen 5 次。查看完整布局有助于说明不同模型对同一输入的具体反应。'),
('08_center_effect','保持周围八格相同，改变中心当前动作',
 '将周围八格完全相同、只有中心 C/D 不同的两个局面配成一对，共 256 对。横轴为每对的 C 建议比例差：P(C 建议 | 中心 C) − P(C 建议 | 中心 D)，单位为百分点；纵轴为具有该差值的局面对数占 256 对的百分比。正值表示中心当前为 C 时更常建议 C。',
 '平均差为 GPT 41.17、DeepSeek 73.91、Qwen 18.44 个百分点。DeepSeek 的建议与中心当前动作关联最强。中心动作变化也伴随提示中的收益描述变化，因此该配对不是单独识别“心理锚定”的因果实验。')]

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def build(allow_partial=False):
    rows=read_csv(SOURCE/'merged/combined_answers.csv')
    groups={}
    for r in rows:groups.setdefault(r['grid_key'],{}).setdefault(r['model_label'],[]).append(r)
    states=[]
    for key,group in sorted(groups.items()):
        example=next(iter(group.values()))[0]
        s=dict(key=key,center=key[4],direct=int(example['cardinal_c']),diagonal=int(example['diagonal_c']))
        for m in MODELS:
            replies=sorted(group[m],key=lambda r:int(r['repetition']))
            assert len(replies)==5 and {int(r['repetition']) for r in replies}==set(range(1,6))
            votes=''.join(r['recommendation'] for r in replies)
            assert set(votes)<=set('CD')
            s[m]=dict(c=votes.count('C'),votes=votes)
        s['spread']=max(s[m]['c'] for m in MODELS)-min(s[m]['c'] for m in MODELS)
        s['disagree']=len({s[m]['c']>=3 for m in MODELS})>1
        states.append(s)
    assert len(rows)==7680 and len(states)==512
    summaries={}
    for m in MODELS:
        values=[s[m]['c'] for s in states]
        c=sum(values);u=sum(k in (0,5) for k in values)
        cc=sum(s[m]['c'] for s in states if s['center']=='C')
        dc=sum(s[m]['c'] for s in states if s['center']=='D')
        retain=cc+1280-dc
        summaries[m]=dict(c=c,u=u,cc=cc,dc=dc,retain=retain,pair=sum((k*(k-1)+(5-k)*(4-k))/20 for k in values)/512)
    assert [summaries[m]['c'] for m in MODELS]==[707,1004,1466]
    overview=''.join(f'<article class="metric {m.lower()}"><h3>{m}</h3><div class="big">{summaries[m]["c"]/2560*100:.2f}%</div><p>C 建议比例：<strong>{summaries[m]["c"]:,} / 2,560</strong></p><p>五次完全一致：<strong>{summaries[m]["u"]} / 512 个局面</strong>（{summaries[m]["u"]/512*100:.2f}%）</p></article>' for m in MODELS)
    metrics=[('总体 C 建议比例','全部回答',lambda x:x['c']/2560),('当前 C → 建议 C','当前中心为 C 的回答',lambda x:x['cc']/1280),('当前 D → 建议 C','当前中心为 D 的回答',lambda x:x['dc']/1280),('保持当前动作','全部回答',lambda x:x['retain']/2560),('五次完全一致的局面','全部局面',lambda x:x['u']/512),('同局面回答对一致率','每局面十对回答，再按局面平均',lambda x:x['pair'])]
    stat_rows=''.join('<tr><th scope="row">'+name+'</th>'+''.join(f'<td>{fn(summaries[m])*100:.2f}%</td>' for m in MODELS)+'<td>'+den+'</td></tr>' for name,den,fn in metrics)
    gallery=[];figure_hashes={}
    for stem,title,note,interpretation in FIGURES:
        path=REPORT/'figures'/f'{stem}.png'
        if not path.exists():
            assert allow_partial,f'Missing figure: {path}'
            continue
        data=path.read_bytes();encoded=base64.b64encode(data).decode();figure_hashes[path.name]=hashlib.sha256(data).hexdigest()
        gallery.append(f'<figure id="figure-{stem[:2]}"><h3>图 {int(stem[:2])} · {title}</h3><a href="data:image/png;base64,{encoded}" download="{path.name}" aria-label="下载图 {int(stem[:2])}"><img loading="lazy" src="data:image/png;base64,{encoded}" alt="{html.escape(title)}。{html.escape(note)}"></a><figcaption><p><strong>坐标与统计量：</strong>{note}</p><p class="interpretation"><strong>如何解读：</strong>{interpretation}</p></figcaption></figure>')
    page=TEMPLATE.replace('__OVERVIEW__',overview).replace('__STAT_ROWS__',stat_rows).replace('__GALLERY__','\n'.join(gallery)).replace('__FIGURE_COUNT__',str(len(gallery)))
    page=page.replace('__STATE_DATA__',json.dumps(states,separators=(',',':')).replace('</','<\\/'))
    output=REPORT/'三模型五次回答_交互报告.html';output.parent.mkdir(exist_ok=True,parents=True);output.write_text(page,encoding='utf-8')
    (REPORT/'review').mkdir(exist_ok=True)
    validation=dict(status='build_passed',states=512,replies=7680,replies_per_model=2560,replies_per_state=5,all_replies_equally_weighted=True,independent_batch_grouping=False,figures_embedded=len(gallery),external_assets_required=False,html_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),figure_sha256=figure_hashes,summary_counts=summaries)
    (REPORT/'review/html_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Built {output} ({output.stat().st_size:,} bytes; {len(gallery)} figures)')

TEMPLATE=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三模型五次回答 · 中文比较分析</title><style>
:root{--ink:#17394f;--muted:#526a79;--line:#d8e3e9;--paper:#fff;--bg:#f2f6f8;--gpt:#4776af;--deepseek:#b66d28;--qwen:#087f76;--c:#076e61;--cbg:#d9f0e9;--d:#8a471f;--dbg:#f9e8d9}
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Microsoft YaHei","Noto Sans CJK SC",system-ui,-apple-system,sans-serif}a{color:#195d88;text-underline-offset:3px}a:hover{color:#083e61}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #e6a134;outline-offset:3px}
.wrap{max-width:1220px;margin:auto;padding:0 28px}header{background:#17394f;color:white;padding:58px 0 40px}header h1{font-size:clamp(30px,4vw,48px);line-height:1.13;letter-spacing:-.03em;margin:8px 0 18px;max-width:900px}.eyebrow{font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:12px;color:#a9dbd5}.lead{max-width:950px;font-size:19px;line-height:1.6;color:#d8e8ef}header a{color:#d7f5f0}nav{display:flex;gap:12px;flex-wrap:wrap;padding-top:12px}nav a{border:1px solid #5b7b8e;border-radius:5px;padding:7px 12px;text-decoration:none;font-size:14px}main{padding-top:24px!important;padding-bottom:55px!important}section{margin:25px 0 38px;scroll-margin-top:20px}h2{font-size:28px;line-height:1.25;letter-spacing:-.02em;margin:0 0 14px}h3{margin:0 0 8px;font-size:19px}p{margin:0 0 13px}.muted{color:var(--muted)}.small{font-size:13px}.panel{background:white;border:1px solid var(--line);border-radius:9px;padding:26px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:19px 0}.metric{padding:20px 22px;background:white;border:1px solid var(--line);border-top:4px solid;border-radius:7px}.gpt{border-top-color:var(--gpt)}.deepseek{border-top-color:var(--deepseek)}.qwen{border-top-color:var(--qwen)}.big{font-size:36px;line-height:1.2;font-weight:750;margin:8px 0 12px}.metric p{font-size:14px;margin-bottom:5px}.takeaways{display:grid;grid-template-columns:repeat(3,1fr);gap:23px;margin-top:24px}.takeaways p{font-size:15px}.callout{border-left:4px solid #4b8c98;background:#eaf3f5;padding:16px 20px;margin-top:20px}.filters{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:20px 0}label{display:block;font-size:13px;font-weight:700;margin-bottom:5px}select,input{border:1px solid #9db4c1;border-radius:5px;background:white;color:var(--ink);padding:9px 10px;width:100%;font:inherit;font-size:14px;min-height:43px}.query{grid-column:span 2}.actions{display:flex;gap:9px;align-items:end;justify-content:flex-end}.actions button{min-height:43px}button{background:#fff;border:1px solid #9db4c1;color:var(--ink);border-radius:5px;padding:8px 13px;cursor:pointer;font:inherit;font-size:14px}button:hover:not(:disabled){background:#e9f1f4}button:disabled{opacity:.4;cursor:default}.primary{background:#17394f;color:white;border-color:#17394f}.primary:hover:not(:disabled){background:#28556f}.selection-head{display:flex;justify-content:space-between;gap:12px;align-items:center;border-top:1px solid var(--line);padding-top:16px}.selection-head strong{font-size:19px}.selected-cards .big{font-size:29px}.selected-cards{gap:12px}.selected-cards .metric{padding:14px 16px}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:14px}caption{text-align:left;padding:14px 17px;background:#eff5f7;color:var(--muted);font-size:13px}th,td{padding:14px 15px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}thead th{background:#eaf1f5;font-size:13px;position:sticky;top:0;z-index:1}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}tbody tr:hover{background:#f7fafb}.sort-button{border:0;background:transparent;padding:3px 0;font-weight:750;color:#184e73;white-space:nowrap}.sort-button .arrow{margin-left:4px}.key{font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;margin-top:6px}.grid{display:grid;grid-template-columns:repeat(3,23px);gap:2px;width:max-content}.cell{width:23px;height:23px;text-align:center;line-height:23px;font-size:12px;font-weight:600;border:1px solid transparent;border-radius:2px}.cell.center{outline:2px solid #17394f;outline-offset:-2px;font-weight:900}.C{background:var(--cbg);color:var(--c)}.D{background:var(--dbg);color:var(--d)}.context{white-space:nowrap;color:var(--muted);font-size:13px}.fraction{font-size:16px;font-weight:750;white-space:nowrap}.votes{display:flex;gap:2px}.vote{font:700 11px/21px ui-monospace,SFMono-Regular,Consolas,monospace;display:inline-block;width:18px;height:21px;text-align:center;border-radius:2px}.majority{font-size:12px;margin-top:5px;color:var(--muted)}.range{white-space:nowrap;font-size:15px;font-weight:700}.range span{display:block;font-size:11px;font-weight:400;color:var(--muted);white-space:normal}.pagination{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:16px}.pagination .buttons{display:flex;gap:8px}.error{color:#a33c27;font-size:13px;min-height:20px;margin:0 0 10px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:13px;margin:14px 0}.legend .mark{display:inline-block;padding:0 5px;border-radius:3px;margin-right:4px;font-weight:bold}.empty{padding:35px;text-align:center}.method-list{display:grid;grid-template-columns:1fr 1fr;gap:12px 30px}.method-list dt{font-weight:750;margin-bottom:4px}.method-list dd{margin:0;font-size:14px;color:var(--muted)}.method-list div{padding-bottom:14px}.figure-list figure{margin:22px 0;background:#fff;padding:24px 22px;border:1px solid var(--line);border-radius:8px}.figure-list img{width:100%;height:auto;display:block}.figure-list h3{font-size:17px}.figure-list figcaption{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:12px}.footnote{border-top:1px solid var(--line);padding-top:18px;font-size:13px;color:var(--muted)}.pill{display:inline-block;background:#e3edf2;border-radius:20px;padding:3px 10px;font-size:12px;margin:2px 5px 2px 0}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.link-row{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}details summary{cursor:pointer;font-weight:700;padding:8px 0}details p{font-size:14px}#audit-result{display:none}
@media(max-width:900px){.wrap{padding:0 18px}.panel{padding:20px}.filters{grid-template-columns:repeat(2,minmax(0,1fr))}.takeaways{grid-template-columns:1fr;gap:8px}.cards{gap:10px}.metric{padding:16px}.big{font-size:30px}.actions{grid-column:span 2;justify-content:flex-start}.query{grid-column:span 2}#state-table{min-width:890px}.method-list{grid-template-columns:1fr}}
@media(max-width:580px){header{padding:35px 0 26px}.lead{font-size:16px}.cards{grid-template-columns:1fr}.selected-cards{grid-template-columns:1fr}.selected-cards .metric{display:grid;grid-template-columns:105px 1fr;align-items:center;column-gap:12px}.selected-cards h3{margin:0}.selected-cards .big{margin:0}.selected-cards p{grid-column:span 2}.panel{padding:15px}.filters{gap:11px}.selection-head{align-items:flex-start;flex-direction:column}.pagination{flex-wrap:wrap}.figure-list figure{padding:15px 9px}.source-table{min-width:630px}h2{font-size:24px}}
@media print{body{background:white}.wrap{max-width:none;padding:0}header{color:var(--ink);background:white;padding:10px 0}header .eyebrow,header .lead{color:var(--muted)}nav,.filters,.pagination,#explorer,.link-row{display:none}.panel,.metric{break-inside:avoid}.figure-list figure{break-inside:avoid;page-break-before:auto}section{margin-bottom:25px}}

.votes{gap:4px;margin-top:9px}.vote{width:23px;height:25px;line-height:25px;font-size:13px}.reading{max-width:950px}.interpretation{background:#f3f7f8;padding:17px 20px;border-radius:5px;margin:12px 0 0}.interpretation strong{color:#17394f}.figure-list figcaption p:last-child{margin-bottom:0}.metric p{line-height:1.7}h1,h2,h3{letter-spacing:0!important}body{line-height:1.75}header h1{line-height:1.3}#stat-table td,#stat-table th{white-space:nowrap}
</style></head><body>
<header><div class="wrap"><div class="eyebrow">初步结果分析 · 2026 年 11 月 2 日</div><h1>同一局面，五次回答：<br>GPT、DeepSeek 与 Qwen 的建议差异</h1><p class="lead">每个模型均按同一设置下的五次等权重复回答分析：512 个局面 × 5 次 = 2,560 条回答，三个模型共 7,680 条。以下比较建议倾向、局部环境响应、重复一致性和模型间分歧。</p><nav aria-label="报告导航"><a href="#findings">主要发现</a><a href="#explorer">查看 512 个局面</a><a href="#figures">__FIGURE_COUNT__ 幅解释图</a><a href="#methods">指标与方法</a><a href="#files">下载文件</a></nav></div></header>
<main class="wrap"><section id="findings"><h2>三个模型表现出不同的建议风格</h2><p class="muted reading">每个局面在汇总中权重相同，每条回答权重相同。C 表示合作，D 表示背叛；“当前动作”指输入局面中心个体的动作，“建议动作”指模型给出的 C/D 回答。</p><div class="cards">__OVERVIEW__</div><div class="panel"><div class="takeaways"><div><h3>Qwen：最常建议从 D 转为 C</h3><p>中心当前为 D 时，Qwen 有 <strong>48.05%</strong> 的回答建议 C；GPT 为 <strong>7.03%</strong>，DeepSeek 为 <strong>2.27%</strong>。Qwen 的总体 C 建议比例也是三者中最高的。</p></div><div><h3>DeepSeek：最常保持当前动作</h3><p><strong>86.95%</strong> 的回答与中心当前动作相同。它既较多维持 C，也很少将 D 改为 C；因此应结合两个方向理解其行为，而不是只看总体 C 比例。</p></div><div><h3>GPT：五次重复最一致</h3><p><strong>382 / 512 个局面（74.61%）</strong>得到五次相同回答。它的总体 C 建议比例最低；回答稳定性和合作建议倾向是两个不同维度。</p></div></div><div class="callout"><strong>这里测量的是“模型建议”，不是演化后的合作率。</strong>真实仿真访问各局面的频率会随动态过程改变。每个局面只有五次回答，观察到的 C 比例以 20% 为间隔；它用于描述这五次结果，不等于已经精确确定模型的真实回答概率。</div></div>
<div class="table-scroll" style="margin-top:20px"><table id="stat-table"><caption>统一使用五次回答后的关键指标。比例均来自原始 C/D 回答，未先将五次压缩为一票。</caption><thead><tr><th>指标</th><th>GPT</th><th>DeepSeek</th><th>Qwen</th><th>分母 / 计算对象</th></tr></thead><tbody>__STAT_ROWS__</tbody></table></div>
<p class="muted reading" style="margin-top:14px">进一步看局部环境：当中心当前为 D、四个直接邻居全为 C 时，Qwen 有 <strong>75/80 次（93.75%）</strong>建议 C，而 GPT 为 <strong>14/80 次（17.50%）</strong>、DeepSeek 为 <strong>3/80 次（3.75%）</strong>。这个分组包含对角四格的全部 16 种排列，每种贡献五次回答。</p></section>

<section id="explorer" class="panel"><h2>逐局面查看：同一行保留全部五次回答</h2><p>每行是一个完整的 3×3 局面。旋转或镜像后的布局仍视为不同局面；描边中心是被询问个体，上下左右是四个直接邻居，四个角是对角邻居。</p><form id="filters" class="filters" onsubmit="return false">
<div><label for="center-filter">中心当前动作</label><select id="center-filter"><option value="all">C 或 D（全部）</option><option value="C">C · 合作</option><option value="D">D · 背叛</option></select></div>
<div><label for="direct-filter">直接邻居中 C 的数量</label><select id="direct-filter"><option value="all">全部（0–4）</option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
<div><label for="diagonal-filter">对角邻居中 C 的数量</label><select id="diagonal-filter"><option value="all">全部（0–4）</option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
<div><label for="disagreement-filter">三个模型的多数建议</label><select id="disagreement-filter"><option value="all">全部</option><option value="yes">至少两个模型不同</option><option value="no">三个模型都相同</option></select></div>
<div class="query"><label for="query-filter">局面编码或前缀（? 可代表 C 或 D）</label><input id="query-filter" type="text" placeholder="例如 CCC / CDC / CCC" autocomplete="off" spellcheck="false" aria-describedby="query-help query-error"><div id="query-help" class="small muted">按从左到右、从上到下读取。最多九个 C、D 或 ?；空格及 / 自动忽略。</div></div>
<div><label for="sort-filter">排序方式</label><select id="sort-filter"><option value="grid-asc">局面编码：C 在 D 前</option><option value="spread-desc">模型间 C 比例差：大 → 小</option><option value="spread-asc">模型间 C 比例差：小 → 大</option><option value="GPT-desc">GPT 的 C 票数：多 → 少</option><option value="GPT-asc">GPT 的 C 票数：少 → 多</option><option value="DeepSeek-desc">DeepSeek 的 C 票数：多 → 少</option><option value="DeepSeek-asc">DeepSeek 的 C 票数：少 → 多</option><option value="Qwen-desc">Qwen 的 C 票数：多 → 少</option><option value="Qwen-asc">Qwen 的 C 票数：少 → 多</option></select></div>
<div class="actions"><button id="example-button" type="button">示例：C 包围 D</button><button id="reset-button" class="primary" type="button">重置</button></div></form><p id="query-error" class="error" role="status"></p>
<div class="selection-head"><strong id="selected-count" aria-live="polite">已选 512 个局面 / 512</strong><span id="selected-denominator" class="muted small">每个模型 2,560 条回答</span></div><div class="cards selected-cards" id="selected-cards"></div>
<p class="small muted"><strong>筛选后的 C 建议比例</strong> = 已选局面中的 C 回答总数 ÷（已选局面数 × 5）。<strong>五次完全一致率</strong> = 已选局面中五次回答均相同的局面数 ÷ 已选局面数。这些数字会随筛选实时更新；下方八幅图始终使用完整数据。</p>
<div class="legend"><span><span class="mark C">C</span>建议合作</span><span><span class="mark D">D</span>建议背叛</span><span><strong>五票等权：</strong>左至右为重复编号 1–5，无时间演化含义</span></div>
<p class="small muted">窄屏可横向滚动表格，比较三个模型的回答。</p><div class="table-scroll" tabindex="0" role="region" aria-label="可横向滚动的局面比较表"><table id="state-table"><caption>每个模型展示全部五票；至少三票 C，则多数建议为 C，否则为 D。点击列名可排序，当前筛选保持不变。</caption><thead><tr><th scope="col">完整局面</th><th scope="col">局部环境</th><th scope="col" id="head-GPT" aria-sort="none"><button class="sort-button" data-sort="GPT" type="button">GPT <span class="arrow" aria-hidden="true">↕</span><span class="sr-only">：按 C 票数排序</span></button></th><th scope="col" id="head-DeepSeek" aria-sort="none"><button class="sort-button" data-sort="DeepSeek" type="button">DeepSeek <span class="arrow" aria-hidden="true">↕</span><span class="sr-only">：按 C 票数排序</span></button></th><th scope="col" id="head-Qwen" aria-sort="none"><button class="sort-button" data-sort="Qwen" type="button">Qwen <span class="arrow" aria-hidden="true">↕</span><span class="sr-only">：按 C 票数排序</span></button></th><th scope="col" id="head-spread" aria-sort="none"><button class="sort-button" data-sort="spread" type="button">模型间差值 <span class="arrow" aria-hidden="true">↕</span><span class="sr-only">：按 C 比例差排序</span></button></th></tr></thead><tbody id="state-rows"></tbody></table></div>
<div class="pagination"><span id="page-status" class="small muted" aria-live="polite"></span><div class="buttons"><button id="previous" type="button">← 上一页</button><button id="next" type="button">下一页 →</button></div></div><p class="small muted" style="margin-top:12px"><strong>模型间差值</strong> = 同一局面三个模型 C 建议比例中的最大值 − 最小值，单位为百分点。例如 100% − 40% = 60 个百分点；这是模型间差异，不是置信区间。</p><noscript><p>局面筛选和表格需要 JavaScript，静态正文与图仍可阅读。完整逐局面数据见 <a href="data/state_summary.csv">局面汇总 CSV</a>。</p></noscript></section>

<section id="figures"><h2>八幅图：从总体差异到具体局面</h2><p class="muted">各图均使用完整的五次重复数据。图内标注坐标与数值，图下说明分母、统计量和解读；点击图可另存 PNG。</p><div class="figure-list">__GALLERY__</div></section>

<section id="methods" class="panel"><h2>分析方法与统计量定义</h2><dl class="method-list">
<div><dt>统一的五次重复分析单位</dt><dd>同一模型、同一局面下的五次回答作为同一设置的等权重复样本；每次回答均保留。512 = 2⁹ 个二元 3×3 局面全部纳入，因此每个模型为 512 × 5 = 2,560 条。</dd></div>
<div><dt>C 建议比例</dt><dd>单个局面为 k/5，其中 k 是五次回答中 C 的次数。分组时，C 回答总数除以该组全部回答数；由于每个局面均为五次，这也等于组内各局面 C 比例的简单平均。</dd></div>
<div><dt>五次完全一致与多数建议</dt><dd>五次完全一致指 k=0 或 k=5。多数建议在 k≥3 时为 C，否则为 D。多数标签只用于模型间比较；主要比例统计和数据表始终保留全部五票。</dd></div>
<div><dt>回答对一致率</dt><dd>同一局面的五次回答可组成十对不同回答，一致率为 [k(k−1)+(5−k)(4−k)]/20，再对 512 个局面平均。这十对共享回答，不能作为十个独立样本。</dd></div>
<div><dt>保持当前动作</dt><dd>模型建议与中心输入动作相同即计为保持。分子为 C→C 和 D→D 的回答数之和，分母为 2,560；中心 C 与 D 的输入数量各占一半。</dd></div>
<div><dt>直接邻居与对角邻居</dt><dd>上下左右四个直接邻居参与中心当前收益；对角四格可在输入中看到，但不直接参与该中心收益。不同邻居数量对应的完整局面数不同，分组比例都使用本组回答数作分母。</dd></div>
<div><dt>配对中心差值</dt><dd>固定周围八格，配对中心为 C 与 D 的两个局面，共 256 对。计算两者 C 建议比例之差；中心改变时相应收益文字也改变，因此该差值描述输入变化的整体关联。</dd></div>
<div><dt>解释范围</dt><dd>这是 C/D 建议记录的描述性分析。提示给出 1&lt;b&lt;2，未指定数值 b；本分析没有运行新的演化仿真，也不据此判定模型的长期合作率、收益、最优性或实际推理过程。</dd></div></dl>
<div class="callout"><strong>下一步可以用这三套五次回答表进行同条件仿真。</strong>在相同 b、询问概率 p、信任相关参数及随机种子配对条件下，比较长期合作率、收益和收敛过程，才能检验上述建议差异如何转化为系统行为。</div></section>

<section id="files" class="panel"><h2>报告与数据</h2><div class="link-row"><a href="三模型五次回答_中文分析.pdf">中文图文 PDF</a><a href="data/replies.csv" download>7,680 条回答 CSV</a><a href="data/state_summary.csv" download>逐局面统计 CSV</a></div><p class="small muted">本 HTML 的图、文字及交互数据均已内嵌，离线可用。下载配套 PDF / CSV 时，请保持它们与 HTML 的相对位置。此页无需联网，不会发送模型请求。</p></section>
<footer class="footnote">本轮统一按每个模型五次等权重复回答进行分析。原始数据保留于上级报告目录；本页的重复编号仅用于逐条查看，统计汇总不区分采集来源。模型名称沿用本地记录标签。</footer></main>
<script id="state-data" type="application/json">__STATE_DATA__</script>
<script>
'use strict';
const STATES=JSON.parse(document.getElementById('state-data').textContent),MODELS=['GPT','DeepSeek','Qwen'],PAGE_SIZE=24;
const el=id=>document.getElementById(id);let page=0,selected=[];
function options(){return {center:el('center-filter').value,direct:el('direct-filter').value,diagonal:el('diagonal-filter').value,disagree:el('disagreement-filter').value,query:el('query-filter').value,sort:el('sort-filter').value};}
function queryPattern(text){const clean=text.toUpperCase().replace(/[\s/]/g,'');if(clean.length>9||/[^CD?]/.test(clean))return {error:'请输入最多九个 C、D 或 ?；可包含空格和 /。'};return {pattern:new RegExp('^'+clean.replace(/\?/g,'[CD]'))};}
function filterStates(o){const q=queryPattern(o.query||'');if(q.error)return [];const items=STATES.filter(s=>(o.center==='all'||s.center===o.center)&&(o.direct==='all'||s.direct===Number(o.direct))&&(o.diagonal==='all'||s.diagonal===Number(o.diagonal))&&(o.disagree==='all'||s.disagree===(o.disagree==='yes'))&&q.pattern.test(s.key));const bits=o.sort.split('-'),key=bits[0],direction=bits[1]==='asc'?1:-1;return items.sort((a,b)=>{if(key==='grid')return a.key.localeCompare(b.key);const av=key==='spread'?a.spread:a[key].c,bv=key==='spread'?b.spread:b[key].c;return direction*(av-bv)||a.key.localeCompare(b.key);});}
function fraction(num,den){return den?(100*num/den).toFixed(2)+'%':'—';}
function grid(s){return '<div class="grid" role="img" aria-label="局面 '+s.key.slice(0,3)+' / '+s.key.slice(3,6)+' / '+s.key.slice(6)+'；中心 '+s.center+'">'+[...s.key].map((c,i)=>'<span aria-hidden="true" class="cell '+c+(i===4?' center':'')+'">'+c+'</span>').join('')+'</div><div class="key">'+s.key+'</div>';}
function modelCell(v){return '<div class="fraction">'+v.c+' / 5 C <span class="small muted">('+v.c*20+'%)</span></div><div class="votes" aria-label="五次回答 '+v.votes.split('').join('、')+'">'+[...v.votes].map((c,i)=>'<span class="vote '+c+'" title="第 '+(i+1)+' 次：'+c+'" aria-hidden="true">'+c+'</span>').join('')+'</div><div class="majority">多数建议：'+(v.c>=3?'C':'D')+'</div>';}
function render(resetPage=true){if(resetPage)page=0;const o=options();selected=filterStates(o);const total=selected.length;page=Math.max(0,Math.min(page,Math.max(0,Math.ceil(total/PAGE_SIZE)-1)));el('query-error').textContent=queryPattern(o.query).error||'';el('query-filter').setAttribute('aria-invalid',queryPattern(o.query).error?'true':'false');el('selected-count').textContent='已选 '+total+' 个局面 / 512';el('selected-denominator').textContent='每个模型 '+(total*5).toLocaleString('en-US')+' 条回答';el('selected-cards').innerHTML=MODELS.map(m=>{const c=selected.reduce((a,s)=>a+s[m].c,0),u=selected.filter(s=>s[m].c===0||s[m].c===5).length;return '<article class="metric '+m.toLowerCase()+'"><h3>'+m+'</h3><div class="big">'+fraction(c,total*5)+'</div><p>建议 C：<strong>'+c.toLocaleString('en-US')+' / '+(total*5).toLocaleString('en-US')+'</strong> 条回答</p><p>五次完全一致：<strong>'+u+' / '+total+'</strong> 个局面（'+fraction(u,total)+'）</p></article>';}).join('');const visible=selected.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE);el('state-rows').innerHTML=visible.length?visible.map(s=>'<tr><th scope="row">'+grid(s)+'</th><td class="context">直接 C：<strong>'+s.direct+'</strong> / 4<br>对角 C：<strong>'+s.diagonal+'</strong> / 4<br>中心：<strong>'+s.center+'</strong></td>'+MODELS.map(m=>'<td>'+modelCell(s[m])+'</td>').join('')+'<td class="range">'+s.spread*20+' 个百分点<span>'+(s.disagree?'多数建议不同':'多数建议相同')+'</span></td></tr>').join(''):'<tr><td colspan="6" class="empty">没有符合条件的局面，请调整筛选或点击重置。</td></tr>';el('page-status').textContent=total?'第 '+(page*PAGE_SIZE+1)+'–'+Math.min((page+1)*PAGE_SIZE,total)+' 个 / 共 '+total+' 个局面 · 第 '+(page+1)+' / '+Math.ceil(total/PAGE_SIZE)+' 页':'0 个局面';el('previous').disabled=page===0;el('next').disabled=(page+1)*PAGE_SIZE>=total;[...MODELS,'spread'].forEach(key=>{const active=o.sort.startsWith(key+'-'),asc=o.sort.endsWith('-asc');el('head-'+key).setAttribute('aria-sort',active?(asc?'ascending':'descending'):'none');el('head-'+key).querySelector('.arrow').textContent=active?(asc?'↑':'↓'):'↕';});}
function reset(){['center','direct','diagonal','disagreement'].forEach(k=>el(k+'-filter').value='all');el('query-filter').value='';el('sort-filter').value='grid-asc';render();}
['center','direct','diagonal','disagreement','sort'].forEach(k=>el(k+'-filter').addEventListener('change',()=>render()));el('query-filter').addEventListener('input',()=>render());el('reset-button').addEventListener('click',reset);el('example-button').addEventListener('click',()=>{reset();el('query-filter').value='CCC / CDC / CCC';render();});el('previous').addEventListener('click',()=>{page--;render(false);});el('next').addEventListener('click',()=>{page++;render(false);});document.querySelectorAll('[data-sort]').forEach(b=>b.addEventListener('click',()=>{const key=b.dataset.sort;el('sort-filter').value=el('sort-filter').value===key+'-desc'?key+'-asc':key+'-desc';render();}));
render();
function selfTest(){const tests=[],check=(label,ok)=>{tests.push({label,passed:!!ok});if(!ok)throw Error(label);};try{check('512 个唯一局面',STATES.length===512&&new Set(STATES.map(s=>s.key)).size===512);check('全部五票保留且一致计数',STATES.every(s=>MODELS.every(m=>s[m].votes.length===5&&[...s[m].votes].filter(v=>v==='C').length===s[m].c)));check('没有拆分回答组',STATES.every(s=>MODELS.every(m=>Object.keys(s[m]).join(',')==='c,votes')));const base={center:'all',direct:'all',diagonal:'all',disagree:'all',query:'',sort:'grid-asc'};check('中心 C 为 256 个局面',filterStates({...base,center:'C'}).length===256);check('四邻 C 的中心 D 为 16 个局面',filterStates({...base,center:'D',direct:'4'}).length===16);check('邻居数量分组分母',JSON.stringify([0,1,2,3,4].map(k=>filterStates({...base,center:'D',direct:String(k)}).length))===JSON.stringify([16,64,96,64,16]));const one=filterStates({...base,query:'CCC / CDC / CCC'});check('具体示例票数 2/0/5',one.length===1&&one[0].GPT.c===2&&one[0].DeepSeek.c===0&&one[0].Qwen.c===5);check('非法查询不返回局面',filterStates({...base,query:'BAD'}).length===0);check('通配符前缀',filterStates({...base,query:'????C'}).length===256);check('多数分歧筛选',filterStates({...base,disagree:'yes'}).every(s=>s.disagree));const sorted=filterStates({...base,sort:'Qwen-desc'});check('C 票数降序',sorted.every((s,i)=>i===0||sorted[i-1].Qwen.c>=s.Qwen.c));check('初始页 24 行',el('state-rows').rows.length===24);check('页面无横向溢出',document.documentElement.scrollWidth<=window.innerWidth);el('center-filter').value='D';render();check('筛选后分母更新',el('selected-denominator').textContent==='每个模型 1,280 条回答');check('筛选后 C 计数更新',['90 / 1,280','29 / 1,280','615 / 1,280'].every(v=>el('selected-cards').textContent.includes(v)));el('next').click();check('翻页保持筛选',options().center==='D'&&page===1&&selected.length===256);el('head-Qwen').querySelector('button').click();check('列排序保持筛选',options().center==='D'&&options().sort==='Qwen-desc'&&page===0);el('query-filter').value='CCCCCCCCC';render();check('零结果不出现 NaN',selected.length===0&&!el('selected-cards').textContent.includes('NaN'));reset();el('example-button').click();check('示例按钮且五票并排',selected.length===1&&el('state-rows').querySelectorAll('.votes').length===3&&el('state-rows').querySelectorAll('.vote').length===15);reset();check('重置恢复全部',selected.length===512&&page===0);document.documentElement.dataset.audit='passed';}catch(e){tests.push({error:e.message});document.documentElement.dataset.audit='failed';}const node=document.createElement('pre');node.id='audit-result';node.textContent=JSON.stringify({tests,viewport:{innerWidth:window.innerWidth,clientWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth}});document.body.append(node);window.EXPLORER_TESTS=tests;}
if(new URLSearchParams(location.search).get('selftest')==='1')selfTest();
</script></body></html>'''

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true',help='Permit preview before all figures exist.')
    build(parser.parse_args().allow_partial)

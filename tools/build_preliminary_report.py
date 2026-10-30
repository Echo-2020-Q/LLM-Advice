"""Build the illustrated English research brief as PDF and editable DOCX.

Run plot_report_macro.py and plot_report_advice.py first. Reads audited results;
never calls an LLM API or runs an SPD simulation.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
LOWER = Path(str(ROOT).lower())
if LOWER.exists() and LOWER.samefile(ROOT):
    ROOT = LOWER
OUT = ROOT / 'reports/Preliminary_1029'
FIG = OUT / 'figures'
NAVY, TEAL, MUTED, PALE = '17394F', '008F85', '556B79', 'EEF4F6'
WIDTH = A4[0] - 84


def color(value):
    return colors.HexColor('#' + value.lstrip('#'))


def b(kind, text='', **kw):
    return dict(kind=kind, text=text, **kw)


PAGES = [
    [
        b('kicker', 'RESEARCH BRIEF  /  PRELIMINARY RESULTS'),
        b('title', 'LLM Advice and\nSpatial Cooperation'),
        b('subtitle', 'Simulation evidence and local decisions from GPT, Qwen and DeepSeek'),
        b('meta', '29 October 2026  |  Spatial Prisoner’s Dilemma  |  Five-figure report'),
        b('lead', 'Cooperation depends on what advice recommends in a particular local state, as well as who receives it and how often it is adopted. Qwen sustains more cooperation than GPT across much of the current scan. DeepSeek’s newly collected advice reveals a different pattern: it often recommends retaining cooperation, but rarely recommends that current defectors cooperate.'),
        b('cards', rows=[
            ['130 / 210', '89.72% → 1.67%', '73.83% vs 1.95%'],
            ['Active settings with higher mean cooperation for Qwen; 80 ties.', 'GPT cooperation when coverage rises from 0.6 to 0.7 at b = 1.02, λ = 1.', 'DeepSeek recommends C when currently C versus when currently D.'],
        ]),
        b('subhead', 'Study at a glance'),
        b('table', widths=[.25, .75], rows=[
            ['Component', 'Design and measurement'],
            ['Population model', '50 × 50 periodic lattice; four direct neighbors; synchronous updates; initial cooperation probability 0.5; Fermi K = 0.1.'],
            ['Simulation scan', 'GPT and Qwen only. 10,000 steps; 308 parameter settings × 2 seeds (42, 43) per model.'],
            ['Parameters', 'b: 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2; P: 0, 0.1, …, 1; λ: 0, 0.1, 0.5, 1.'],
            ['Advice intervention', 'P: fraction of positions assigned advice eligibility at initialization. λ: probability of adopting advice at each eligible update.'],
            ['Primary outcome', 'Mean cooperation over steps 9001–10000. Compare identical parameters and seeds; saved Fermi control trajectories match.'],
            ['Local advice data', 'All 512 binary 3 × 3 configurations; center is the focal player. DeepSeek: 1 reply per state. GPT and Qwen: 2 each.'],
        ]),
        b('note', 'Evidence scope. The simulation comparison concerns two fixed advice tables and two seeds per setting. DeepSeek is compared at the advice-table level only. Final-window averages are not confirmed equilibria.'),
    ],
    [
        b('kicker', '01  /  SIMULATION RESULTS'),
        b('heading', 'Cooperation differs across the parameter scan'),
        b('lead', 'Across all 210 active-advice settings (P > 0 and λ > 0), Qwen has higher two-seed mean final-window cooperation in 130 settings and ties GPT in 80. The maps show where the observed differences are largest.'),
        b('figure', file='macro_overview.png', number=1),
        b('caption', 'Figure 1. Qwen minus GPT in final-window cooperation, measured in percentage points. Each cell averages two matched simulation seeds. Panels show λ = 0.5 and 1; the 130/80 count also includes λ = 0.1, which is not displayed. Darker cells indicate larger differences, not greater statistical confidence.'),
        b('subhead', 'Large gaps can coexist with ties'),
        b('body', 'At b = 1.02, λ = 1 and P = 0.7, final-window cooperation is 99.16% for Qwen and 1.67% for GPT. At b = 1.2, λ = 0.5 and P = 0.9, both GPT runs reach 0% final-window cooperation, while both Qwen runs remain at 100% throughout that window. A zero difference elsewhere can represent either shared failure or shared high cooperation.'),
        b('note', 'Interpretation. These are descriptive comparisons on the chosen parameter grid, not a general model ranking. At P = λ = 1, the advice-only dynamics reuse the same table across b; repeated b values are not independent evidence. The advice prompts do not specify a numerical b.'),
    ],
    [
        b('kicker', '02  /  COVERAGE AND DYNAMICS'),
        b('heading', 'Increasing advice coverage can reduce cooperation'),
        b('lead', 'The GPT table exhibits a sharp observed change between P = 0.6 and 0.7 at b = 1.02 and λ = 1. Both seeds show the same ordering. This motivates a denser scan and more replication around that interval.'),
        b('figure', file='macro_detail.png', number=2),
        b('caption', 'Figure 2. Coverage response and trajectories at a diagnostic parameter slice. Lines show two-seed means; shading shows the observed seed range, not a confidence interval. Panel (a) includes both seed values. Panel (b) is linear in time through step 10 and logarithmic afterward; the no-advice Fermi control is shown for context.'),
        b('table', widths=[.34, .22, .22, .22], rows=[
            ['Setting at b = 1.02, λ = 1', 'Seed 42', 'Seed 43', 'Mean'],
            ['GPT, P = 0.6', '93.29%', '86.15%', '89.72%'],
            ['GPT, P = 0.7', '2.10%', '1.23%', '1.67%'],
            ['Qwen, P = 0.7', '99.13%', '99.20%', '99.16%'],
        ]),
        b('body', 'The 88.05 percentage-point GPT decline shows that broader access to advice need not improve collective cooperation. However, two seeds and a coverage increment of 0.1 cannot locate a critical threshold. The high-cooperation GPT runs at P = 0.6 were still increasing near the end of the observation period, making longer trajectories particularly useful there.'),
    ],
    [
        b('kicker', '03  /  LOCAL ADVICE PATTERNS'),
        b('heading', 'Preserving cooperation and converting defectors differ'),
        b('body', 'All three tables cover the same 512 configurations with identical corresponding prompts. C denotes cooperation and D defection. Overall C recommendations are 37.89% for DeepSeek, 27.34% for GPT and 56.93% for Qwen, but conditioning on the current action reveals a more informative contrast.'),
        b('figure', file='advice_transitions.png', number=3),
        b('caption', 'Figure 3. Recorded recommendations conditional on the focal player’s current action. Each row weights its 256 configurations equally. DeepSeek has one reply per configuration, while GPT and Qwen have two; the cell counts retain those different denominators. These are advice frequencies, not realized simulation transition rates.'),
        b('figure', file='advice_neighbors.png', number=4),
        b('caption', 'Figure 4. Advice to cooperate by current action and number of cooperating direct neighbors, pooling corner arrangements. For a current defector with four cooperating direct neighbors, C recommendations are 0/16 for DeepSeek, 5/32 for GPT and 29/32 for Qwen. Small per-state sample sizes limit inference about stable model behavior.'),
        b('note', 'Main contrast. DeepSeek frequently retains C but rarely recommends D → C. Qwen recommends D → C much more often in cooperative neighborhoods. This distinction suggests a mechanism to test, rather than a demonstrated cause of the simulation gaps.'),
    ],
    [
        b('kicker', '04  /  INTERPRETATION AND NEXT EXPERIMENTS'),
        b('heading', 'Test the mechanism with balanced sampling'),
        b('body', 'A single configuration illustrates the different recorded decisions. These replies suggest that persistence of existing cooperation and expansion into currently defecting positions should be measured separately.'),
        b('figure', file='advice_example.png', number=5),
        b('caption', 'Figure 5. Exact configuration CCC / CDC / CCC, including all four corners. The replies shown are the saved observations. GPT’s C/D pair here should not be confused with the 15.6% group average in Figure 4, which includes every corner arrangement.'),
        b('subhead', 'What the present evidence can establish'),
        b('table', widths=[.27, .73], rows=[
            ['Limitation', 'Consequence for interpretation'],
            ['Few repeats', 'Two simulation seeds and one advice table per model limit generalization. DeepSeek’s one reply per state cannot estimate repeat agreement.'],
            ['Different sampling layers', 'More simulation seeds do not resolve uncertainty in the generated advice table. Empirical advice frequencies weight states equally, unlike visited states in a simulation.'],
            ['Generation settings', 'Settings were not fully standardized. GPT responses record medium reasoning effort, temperature 1.0 and top_p 0.98; equivalent metadata are absent for the other two. Model labels come from logs.'],
            ['Scope and mechanism', 'DeepSeek has no SPD results here. Its middle overall C rate does not predict an intermediate long-term outcome. Testing the mechanism requires state-visitation measurements and controlled interventions.'],
        ]),
        b('subhead', 'A focused next round'),
        b('body', '<b>1. Balance advice sampling.</b> Start with the same number of replies per state (for example, 10), and independently rebuild tables where feasible. This is a practical starting point, not a precision guarantee.'),
        b('body', '<b>2. Resolve the coverage change.</b> Add simulation seeds and finer P values around 0.6–0.7; extend trajectories that are still evolving, then compare equal-length final windows.'),
        b('body', '<b>3. Test conversion directly.</b> Record visited states and advice-induced C → D / D → C events. Replacing only the current-D entries between tables would test whether their different conversion behavior drives the population outcomes.'),
        b('source', 'Sources: audited GPT/Qwen scans dated 27 October 2026; GPT advice dated 16 October, Qwen 24 October and DeepSeek 29 October. Source mappings, plotted values and checksums accompany this report. No new model calls or simulations were made for this brief.'),
    ],
]


def register_fonts():
    folder = Path('/mnt/c/Windows/Fonts')
    if (folder/'arial.ttf').exists():
        files = ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf']
    else:
        folder = Path('/usr/share/fonts/truetype/dejavu')
        files = ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans-Oblique.ttf', 'DejaVuSans-BoldOblique.ttf']
    for name, file in zip(['Brief', 'Brief-Bold', 'Brief-Italic', 'Brief-BoldItalic'], files):
        pdfmetrics.registerFont(TTFont(name, str(folder/file)))
    pdfmetrics.registerFontFamily('Brief',normal='Brief',bold='Brief-Bold',italic='Brief-Italic',boldItalic='Brief-BoldItalic')


def styles():
    base = dict(fontName='Brief',fontSize=10,leading=14,textColor=color(NAVY),spaceAfter=7)
    s = {k:ParagraphStyle(k,**base) for k in ['body','lead','note','source','caption','meta','subtitle','table']}
    s['lead'].fontSize=10.6;s['lead'].leading=15;s['lead'].spaceAfter=13
    for k in ['note','source','caption','meta']:
        s[k].fontSize=8.4;s[k].leading=11.4;s[k].textColor=color(MUTED)
    s['source'].fontSize=7.8;s['source'].leading=10.2
    s['caption'].spaceAfter=11
    s['meta'].spaceAfter=17
    s['subtitle'].fontSize=11.5;s['subtitle'].leading=15;s['subtitle'].textColor=color(MUTED)
    s['table'].fontSize=9;s['table'].leading=12;s['table'].spaceAfter=0
    for k,size,leading,after,col in [('kicker',9,12,10,TEAL),('title',32,35,9,NAVY),('heading',22,26,13,NAVY),('subhead',12,16,7,NAVY)]:
        s[k]=ParagraphStyle(k,fontName='Brief-Bold',fontSize=size,leading=leading,spaceAfter=after,textColor=color(col),keepWithNext=True)
    return s


def rich(txt):
    # Content is authored locally; only bold spans and line breaks are supported.
    return escape(txt).replace('&lt;b&gt;','<b>').replace('&lt;/b&gt;','</b>').replace('\n','<br/>')


def pdf_table(block,s):
    rows=[]
    for i,row in enumerate(block['rows']):
        rows.append([Paragraph(('<b>'+rich(t)+'</b>') if i==0 else rich(t),s['table']) for t in row])
    if block['kind']=='cards':
        card=ParagraphStyle('card',parent=s['table'],fontName='Brief-Bold',fontSize=19,leading=24,textColor=color(TEAL))
        rows[0]=[Paragraph(rich(t),card) for t in block['rows'][0]]
    widths=[WIDTH*w for w in block.get('widths',[1/len(rows[0])]*len(rows[0]))]
    t=Table(rows,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),color(PALE)),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.6,color('BDCED6')),
        ('LINEBELOW',(0,1),(-1,-1),.35,color('DCE5EA')),
    ]))
    if block['kind']=='cards':
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),color(PALE)),('LINEBELOW',(0,0),(-1,-1),0,colors.white),('BOTTOMPADDING',(0,0),(-1,0),0),('TOPPADDING',(0,1),(-1,1),4)]))
    return [t,Spacer(1,13)]


def build_pdf():
    register_fonts();s=styles();story=[]
    for i,page in enumerate(PAGES):
        if i:story.append(PageBreak())
        for block in page:
            k=block['kind']
            if k in ('table','cards'):story.extend(pdf_table(block,s))
            elif k=='figure':
                p=FIG/block['file']
                with PILImage.open(p) as im:w,h=im.size
                story.extend([Image(str(p),width=WIDTH,height=WIDTH*h/w),Spacer(1,5)])
            else:story.append(Paragraph(rich(block['text']),s[k]))
    destination=OUT/'LLM_Advice_Preliminary_Report.pdf'
    doc=SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=49,bottomMargin=43,
        title='LLM Advice and Spatial Cooperation',author='LLM Advice Research',subject='Preliminary simulation and advice-table results')
    def chrome(c,d):
        c.saveState();c.setStrokeColor(color('D5E1E6'));c.setLineWidth(.6)
        c.line(42,A4[1]-32,A4[0]-42,A4[1]-32);c.line(42,31,A4[0]-42,31)
        c.setFont('Brief',7.5);c.setFillColor(color(MUTED))
        c.drawString(42,A4[1]-24,'LLM ADVICE  /  SPATIAL COOPERATION')
        c.drawRightString(A4[0]-42,A4[1]-24,'RESEARCH BRIEF · 29 OCT 2026')
        c.drawString(42,19,'Preliminary evidence · Fixed advice tables · No DeepSeek simulation results')
        c.drawRightString(A4[0]-42,19,f'{d.page} / 5');c.restoreState()
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
    reader=PdfReader(destination)
    if len(reader.pages)!=5:raise ValueError(f'Expected 5 PDF pages, got {len(reader.pages)}')
    return destination


def shade(cell,fill):
    shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);cell._tc.get_or_add_tcPr().append(shd)


def write_runs(par,text,bold=False):
    import re
    for part in re.split(r'(<b>.*?</b>)',text):
        is_bold=part.startswith('<b>')
        run=par.add_run(part[3:-4] if is_bold else part)
        if bold or is_bold:
            run.bold=True


def build_docx():
    doc=Document();sec=doc.sections[0]
    sec.page_width=Pt(A4[0]);sec.page_height=Pt(A4[1])
    sec.top_margin=Pt(49);sec.bottom_margin=Pt(43);sec.left_margin=sec.right_margin=Pt(42)
    sec.header_distance=Pt(17);sec.footer_distance=Pt(15)
    normal=doc.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(10)
    normal.font.color.rgb=RGBColor.from_string(NAVY)
    normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.12
    for name,size,col in [('Title',32,NAVY),('Heading 1',21,NAVY),('Heading 2',12,NAVY)]:
        st=doc.styles[name];st.font.name='Arial';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string(col);st.font.bold=True
        st.paragraph_format.space_before=Pt(0);st.paragraph_format.space_after=Pt(9);st.paragraph_format.keep_with_next=True
    head=sec.header.paragraphs[0];head.text='LLM ADVICE  /  SPATIAL COOPERATION                                      RESEARCH BRIEF · 29 OCT 2026'
    for run in head.runs:run.font.size=Pt(7.2);run.font.color.rgb=RGBColor.from_string(MUTED)
    foot=sec.footer.paragraphs[0];foot.text='Preliminary evidence · Fixed advice tables · No DeepSeek simulation results                 '
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    for run in foot.runs:run.font.size=Pt(7.2);run.font.color.rgb=RGBColor.from_string(MUTED)
    for i,page in enumerate(PAGES):
        if i:doc.add_page_break()
        for block in page:
            k=block['kind']
            if k=='figure':
                p=doc.add_paragraph();p.paragraph_format.space_after=Pt(3);p.paragraph_format.keep_with_next=True
                p.add_run().add_picture(str(FIG/block['file']),width=Pt(WIDTH))
            elif k in ('table','cards'):
                rows=block['rows'];tab=doc.add_table(rows=len(rows),cols=len(rows[0]));tab.alignment=WD_TABLE_ALIGNMENT.LEFT;tab.autofit=False
                widths=block.get('widths',[1/len(rows[0])]*len(rows[0]))
                for col,w in zip(tab.columns,widths):col.width=Pt(WIDTH*w)
                for ri,row in enumerate(rows):
                    for ci,txt in enumerate(row):
                        cell=tab.cell(ri,ci);cell.width=Pt(WIDTH*widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
                        p=cell.paragraphs[0];p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(5);p.paragraph_format.line_spacing=1.05
                        write_runs(p,txt,bold=ri==0)
                        for run in p.runs:
                            run.font.size=Pt(8.8)
                            if k=='cards' and ri==0:run.font.size=Pt(18);run.font.color.rgb=RGBColor.from_string(TEAL)
                        if ri==0 or k=='cards':shade(cell,PALE)
                        margins=OxmlElement('w:tcMar')
                        for side in ['top','left','bottom','right']:
                            el=OxmlElement('w:'+side);el.set(qn('w:w'),'100' if side in ['left','right'] else '25');el.set(qn('w:type'),'dxa');margins.append(el)
                        cell._tc.get_or_add_tcPr().append(margins)
                    trpr=tab.rows[ri]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
                blank=doc.add_paragraph();blank.paragraph_format.space_after=Pt(2);blank.paragraph_format.line_spacing=Pt(2);blank.add_run().font.size=Pt(2)
            else:
                st={'title':'Title','heading':'Heading 1','subhead':'Heading 2'}.get(k,'Normal')
                p=doc.add_paragraph(style=st);write_runs(p,block['text'])
                if k=='kicker':
                    p.paragraph_format.keep_with_next=True
                    for run in p.runs:run.font.size=Pt(9);run.font.bold=True;run.font.color.rgb=RGBColor.from_string(TEAL)
                if k in ['caption','note','source','meta']:
                    p.paragraph_format.line_spacing=1.08
                    for run in p.runs:run.font.size=Pt(8.2 if k!='source' else 7.6);run.font.color.rgb=RGBColor.from_string(MUTED)
                if k=='lead':
                    for run in p.runs:run.font.size=Pt(10.6)
                if k=='subtitle':
                    for run in p.runs:run.font.size=Pt(11.5);run.font.color.rgb=RGBColor.from_string(MUTED)
    doc.core_properties.title='LLM Advice and Spatial Cooperation'
    doc.core_properties.subject='Preliminary simulation and advice-table results'
    doc.core_properties.author='LLM Advice Research'
    path=OUT/'LLM_Advice_Preliminary_Report.docx';doc.save(path);return path


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for page in PAGES:
        for block in page:
            if block['kind']=='figure':assert (FIG/block['file']).is_file()
    (OUT/'report_content.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=[]
    for i,page in enumerate(PAGES,1):
        text.append(f'PAGE {i}')
        for block in page:
            if block['kind'] in ('table','cards'):text.extend(' | '.join(row) for row in block['rows'])
            elif block['kind']=='figure':text.append(f'[Figure {block["number"]}: {block["file"]}]')
            else:text.append(block['text'].replace('<b>','').replace('</b>',''))
    (OUT/'Report_English.txt').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    print(build_pdf(),flush=True);print(build_docx(),flush=True)


if __name__=='__main__':main()

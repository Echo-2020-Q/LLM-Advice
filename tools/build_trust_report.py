"""Build the illustrated adaptive-trust report from saved figures and audited results.
No model calls or SPD simulations. Run the trust visualization tool first.
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
OUT = ROOT / 'reports/Trust_1030'
FIG = OUT / 'figures'
NAVY, TEAL, MUTED, PALE = '17394F', '008F85', '556B79', 'EEF4F6'
WIDTH = A4[0] - 84


def color(value):
    return colors.HexColor('#' + value.lstrip('#'))


def b(kind, text='', **kw):
    return dict(kind=kind, text=text, **kw)


PAGES = [
    [
        b('kicker', 'RESEARCH UPDATE  /  INDIVIDUAL ADAPTIVE TRUST'),
        b('title', 'When Trust Adapts'),
        b('subtitle', 'Preliminary evidence from GPT advice in the Spatial Prisoner’s Dilemma'),
        b('meta', '30 October 2026  |  77 settings × 2 seeds  |  10,000 steps'),
        b('lead', 'Individual trust expands the observed region of cooperation relative to fixed adoption probability λ = 0.5. The benefit is concentrated at modest temptation to defect; it does not extend to b ≥ 1.05 in this scan, and it does not outperform every fixed policy.'),
        b('cards', rows=[['23 / 70', '0% vs 83.72%', 'Strict sign feedback'], ['Active settings with higher final-window cooperation than fixed λ = 0.5; 47 ties.', 'Two seed outcomes at b = 1.04, P = 1. The mean masks extinction versus survival.', 'Trust can keep increasing at constant payoff because aspiration remains slightly lower.']]),
        b('figure', file='01_cooperation.png', number=1),
        b('caption', 'Figure 1. Cooperation across the adaptive-trust scan. The primary outcome is mean cooperation over rounds 9001–10000. Each setting has two simulation seeds. P is the fixed fraction of eligible positions, not the probability of advice adoption.'),
        b('note', 'Design. 50 × 50 periodic lattice; four interacting neighbors; synchronous updates; initial C probability 0.5; Fermi K = 0.1. Eligible individuals begin at trust T(0) = 0.5, with trust and aspiration learning rates α = β = 0.1 and a three-round payoff window. Advice comes from the same saved GPT table as the earlier fixed-policy scan.'),
    ],
    [
        b('kicker', '01  /  COMPARING ADOPTION POLICIES'),
        b('heading', 'The choice of fixed comparator matters'),
        b('body', 'Across 70 active settings (P > 0), adaptive trust improves mean final-window cooperation over fixed λ = 0.5 in 23 settings and ties in 47. Against the best mean among tested fixed λ in {0, 0.1, 0.5, 1}, it is higher in 14, equal in 29 and lower in 27.'),
        b('figure', file='02_fixed_comparison.png', number=2),
        b('caption', 'Figure 2. Comparisons use the same recorded GPT advice table, matched b, P and seeds, and identical final windows. Differences are percentage points. The best tested fixed policy is a retrospective comparator, not an estimate of an optimal policy.'),
        b('figure', file='03_coverage.png', number=3),
        b('caption', 'Figure 3. Coverage responses include fixed λ = 1. At b = 1.02, P = 0.6, adaptive cooperation is 50.49%, versus 89.72% for fixed λ = 1. At b = 1.01, P = 0.5, the corresponding values are 64.24% and 99.15%. These counterexamples rule out a uniform advantage.'),
    ],
    [
        b('kicker', '02  /  TRUST AND ADVICE USE'),
        b('heading', 'Trust tracks sharply different population outcomes'),
        b('body', 'High-cooperation settings tend to retain high advice use, whereas extinct settings usually have low mean trust. This association alone cannot determine whether trust causes cooperation, responds to it, or both.'),
        b('figure', file='04_trust.png', number=4),
        b('caption', 'Figure 4. Trust and realized advice use refer to eligible individuals only; P = 0 is undefined. A round’s advice use depends on the preceding round’s trust, while the saved trust value is measured after feedback. Whole-population advice use equals P times eligible advice use.'),
        b('figure', file='06_trust_distribution.png', number=5),
        b('caption', 'Figure 5. Final trust distributions at b = 1.02, P = 0.1; b = 1.03, P = 0.7; and b = 1.04, P = 1. Separate seed histograms distinguish within-run heterogeneity from the extinction/survival split in the last setting. Saved aggregates do not identify individual histories.'),
        b('note', 'Distribution caution. Across-person dispersion within one run differs from variation across simulation seeds. The strict high-trust statistic uses T > 0.9; the top histogram bin includes T = 0.9. Neither provides additional independent experimental repeats.'),
    ],
    [
        b('kicker', '03  /  TIME EVOLUTION'),
        b('heading', 'Follow the trajectory before interpreting the endpoint'),
        b('figure', file='05_dynamics.png', number=6),
        b('caption', 'Figure 6. Selected saved trajectories show the joint evolution of cooperation and trust. Curves are descriptive paths from two seeds, not confidence bands or independent time-sample replications. Advice-use curves use a trailing 50-round average.'),
        b('subhead', 'Coverage shifts the observed outcome'),
        b('body', 'On the sampled grid, both seeds reach all-C from P = 0.7 at b = 1.01 and from P = 0.9 at b = 1.02. At b = 1.03, both seeds become extinct for P ≤ 0.6, survive for P ≥ 0.7, and reach all-C at P = 1. Cooperation is nondecreasing with coverage in the final-window outcome for each seed and b in this particular scan.'),
        b('subhead', 'A strategy endpoint is not a trust endpoint'),
        b('body', 'The verified GPT table recommends C/C in the homogeneous all-C neighborhood and D/D in all-D. Together with Fermi imitation, this makes both homogeneous strategy states absorbing in the present model. Trust and aspiration can nevertheless continue to evolve after cooperation stops changing.'),
        b('note', 'Continuing evolution: at b = 1.03, P = 0.7, seed 43, mean cooperation rises from 32.08% in rounds 8001–9000 to 41.71% in rounds 9001–10000 (+9.63 percentage points). The sampled grid does not establish a precise phase boundary or equilibrium.'),
    ],
    [
        b('kicker', '04  /  SENSITIVITY TO SIMULATION SEED'),
        b('heading', 'An intermediate mean can combine extinction and survival'),
        b('lead', 'At b = 1.04, the two seeds produce qualitatively different outcomes at high coverage. Reporting the individual runs is essential: the cell average is not evidence that each run settles at moderate cooperation.'),
        b('figure', file='07_seed_dependence.png', number=7),
        b('caption', 'Figure 7. Seed dependence near the observed survival boundary. Final-window means are reported separately where possible; a seed range is descriptive uncertainty, not a confidence interval.'),
        b('table', widths=[.26,.21,.21,.32], rows=[['Setting', 'Seed 42', 'Seed 43', 'Interpretation'], ['b = 1.04, P = 0.9', '0%', '35.58%', 'Extinction versus a surviving mixed population'], ['b = 1.04, P = 1', '0%', '83.72%', 'Extinction versus high cooperation']]),
        b('body', 'Seed 42 reaches all-D at round 2730 for P = 0.9 and round 3848 for P = 1. At b = 1.04, P = 1, seed 43 ends with mean trust 0.8116 and 64.48% above 0.9; seed 42 ends near 0.0012 with everyone below 0.1. Averaging these runs must not be described as within-run polarization.'),
        b('note', 'With only two seeds per setting, survival probabilities and uncertainty near this boundary remain poorly determined. More independent seeds are the most direct next step here.'),
    ],
    [
        b('kicker', '05  /  PAYOFF AND THE OBSERVATION WINDOW'),
        b('heading', 'Late benefits do not imply immediate benefits'),
        b('figure', file='08_payoff.png', number=8),
        b('caption', 'Figure 8. Payoff comparisons against paired no-advice Fermi controls. Population payoff is raw average per-person payoff, not the three-round feedback signal. Early, whole-horizon and final-window averages answer different questions.'),
        b('table', widths=[.39,.28,.33], rows=[['Window', 'Positive mean difference', 'Negative mean difference'], ['Rounds 1–200', '49 / 70 active settings', '21 / 70'], ['Rounds 1–10000', '60 / 70', '10 / 70'], ['Rounds 9001–10000', 'Nonnegative in all 70', '0 / 70']]),
        b('body', 'The ten negative whole-horizon differences are small in this scan: the most negative is approximately −0.00159 payoff units per person per round. Their existence still prevents a claim that advice uniformly improves welfare over every horizon. Final-window cooperation and cumulative experience should be presented together.'),
        b('subhead', 'What a paired difference measures'),
        b('body', 'A difference subtracts the corresponding no-advice control for the same seed and fixed position group. It includes changes transmitted through neighbors. It is not the direct payoff return of a single adopted recommendation, and the eligible/ineligible groups are not formed according to actual advice use.'),
    ],
    [
        b('kicker', '06  /  MECHANISM AUDIT'),
        b('heading', 'The rule rewards aspiration satisfaction'),
        b('body', 'For eligible individual i at round t, advice is adopted with probability Tᵢ(t−1). After the simultaneous strategy update, qᵢ(t) averages personal raw payoffs over the latest three completed rounds (or all completed rounds when t < 3). The initial state initializes Aᵢ(0), but does not enter that window.'),
        b('table', widths=[.41,.59], rows=[['Condition for an actual adopter', 'Trust update'], ['qᵢ(t) > Aᵢ(t−1)', 'Tᵢ(t) = Tᵢ(t−1) + α [1 − Tᵢ(t−1)]'], ['qᵢ(t) < Aᵢ(t−1)', 'Tᵢ(t) = (1 − α) Tᵢ(t−1)'], ['Equal payoff and aspiration', 'Trust unchanged'], ['No adoption', 'Trust unchanged']]),
        b('body', 'After feedback, every position updates Aᵢ(t) = (1 − β) Aᵢ(t−1) + β qᵢ(t), whether or not it used advice. A valid advice action counts as adoption even if the action stays unchanged. Comparisons use exact signs, with no tolerance or forced exploration.'),
        b('figure', file='09_feedback_caveat.png', number=9),
        b('caption', 'Figure 9. Constant payoff can coexist with continuing positive feedback. At b = 1.01, P = 0.7, seed 42, all-C begins at round 7431; the last 1000 rounds still contain 1,750,000 positive feedback events, while payoff stays 4 and mean aspiration is approximately 3.999999999999998.'),
        b('note', 'The exponential aspiration can approach payoff without equaling it; floating-point residuals reinforce that behavior. Near-unit trust is therefore not independent evidence of ongoing payoff improvements. This qualifies the trust interpretation, but does not establish how cooperation would change under a revised tolerance rule.'),
    ],
    [
        b('kicker', '07  /  EVIDENCE SCOPE AND NEXT EXPERIMENTS'),
        b('heading', 'Resolve the mechanism and the boundary next'),
        b('subhead', '1. Define what neutral feedback means'),
        b('body', 'Retain the current implementation as the benchmark. Compare a documented numerical deadband, and explicitly distinguish aspiration satisfaction from improvement over the preceding round. A three-round average includes earlier decisions, and current payoff also reflects simultaneous neighbor changes; neither is a causal estimate of advice quality.'),
        b('subhead', '2. Add independent seeds where outcomes split'),
        b('body', 'Prioritize at least 10 seeds near b ≈ 1.025–1.045 and P = 0.6–1, especially b = 1.04 at P = 0.9 and 1. This is a practical starting allocation, not a precision guarantee. Report the fraction surviving, time to extinction and separate surviving-run trajectories.'),
        b('subhead', '3. Extend runs that remain informative'),
        b('body', 'Test 20,000–50,000 steps for surviving mixed populations and slow transitions, retaining comparable final windows. Extending a run already at all-D cannot restore cooperation under the unchanged lookup table and absence of mutation.'),
        b('subhead', '4. Separate adaptation from average advice exposure'),
        b('body', 'Retain fixed λ = 0.5 and 1 controls. Add an exposure-matched or homogeneous mean-trust control to distinguish a change in average advice use from individual trust selection. Vary initial trust, α, β and the payoff window in separate, focused ablations.'),
        b('subhead', 'Reproducibility and limits'),
        b('table', widths=[.29,.71], rows=[['Evidence', 'Audit finding'], ['New scan', 'scan_20261029_183749_initial; 154 completed runs; b = 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2; P = 0, 0.1, …, 1.'], ['Advice table', 'Same saved GPT batch and config/results hashes as the 27 October fixed-policy scan; 512 neighborhoods, two recorded replies each, sampled with replacement. No numerical b is supplied to this lookup.'], ['Controls', 'Saved full cooperation and population-payoff trajectories for Fermi controls match across all 154 matched runs, as do initial cooperation rates. Random-number consumption can diverge after intervention.'], ['Missing detail', 'Historical source versions, initial full grids/masks and per-person trust trajectories were not archived. Saved aggregate histograms support distribution analysis, not individual path reconstruction.'], ['Uncertainty', 'Two simulation seeds and one advice table; no significance claims. More seeds do not replace independent sampling of the advice table.']]),
        b('source', 'Source records, plotted tables, trajectory cache and mechanism audit accompany this report in reports/Trust_1030. Analysis used saved outputs only; no new SPD simulations or model calls were made. Future runs should archive the source revision, initial state/mask hashes and selected individual trust/aspiration snapshots.'),
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
    base = dict(fontName='Brief',fontSize=9.6,leading=13.2,textColor=color(NAVY),spaceAfter=7)
    s = {k:ParagraphStyle(k,**base) for k in ['body','lead','note','source','caption','meta','subtitle','table']}
    s['lead'].fontSize=10.2;s['lead'].leading=14;s['lead'].spaceAfter=10
    for k in ['note','source','caption','meta']:
        s[k].fontSize=8.4;s[k].leading=11.4;s[k].textColor=color(MUTED)
    s['source'].fontSize=7.8;s['source'].leading=10.2
    s['caption'].spaceAfter=11
    s['meta'].spaceAfter=11
    s['subtitle'].fontSize=11.5;s['subtitle'].leading=15;s['subtitle'].textColor=color(MUTED)
    s['table'].fontSize=9;s['table'].leading=12;s['table'].spaceAfter=0
    for k,size,leading,after,col in [('kicker',9,12,10,TEAL),('title',32,35,9,NAVY),('heading',21,25,10,NAVY),('subhead',12,16,7,NAVY)]:
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
        card=ParagraphStyle('card',parent=s['table'],fontName='Brief-Bold',fontSize=15,leading=19,textColor=color(TEAL))
        rows[0]=[Paragraph(rich(t),card) for t in block['rows'][0]]
    widths=[WIDTH*w for w in block.get('widths',[1/len(rows[0])]*len(rows[0]))]
    t=Table(rows,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),color(PALE)),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
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
    destination=OUT/'Adaptive_Trust_Report.pdf'
    doc=SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=49,bottomMargin=43,
        title='When Trust Adapts: Preliminary SPD Results',author='LLM Advice Research',subject='Adaptive individual trust: simulation results and mechanism audit')
    def chrome(c,d):
        c.saveState();c.setStrokeColor(color('D5E1E6'));c.setLineWidth(.6)
        c.line(42,A4[1]-32,A4[0]-42,A4[1]-32);c.line(42,31,A4[0]-42,31)
        c.setFont('Brief',7.5);c.setFillColor(color(MUTED))
        c.drawString(42,A4[1]-24,'LLM ADVICE  /  SPATIAL COOPERATION')
        c.drawRightString(A4[0]-42,A4[1]-24,'RESEARCH BRIEF · 30 OCT 2026')
        c.drawString(42,19,'Preliminary evidence · Two seeds per setting · Same GPT advice table')
        c.drawRightString(A4[0]-42,19,f'{d.page} / 8');c.restoreState()
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
    reader=PdfReader(destination)
    if len(reader.pages)!=8:raise ValueError(f'Expected 8 PDF pages, got {len(reader.pages)}')
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
    normal=doc.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(9.6)
    normal.font.color.rgb=RGBColor.from_string(NAVY)
    normal.paragraph_format.space_after=Pt(5);normal.paragraph_format.line_spacing=1.08
    for name,size,col in [('Title',32,NAVY),('Heading 1',21,NAVY),('Heading 2',12,NAVY)]:
        st=doc.styles[name];st.font.name='Arial';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string(col);st.font.bold=True
        st.paragraph_format.space_before=Pt(0);st.paragraph_format.space_after=Pt(9);st.paragraph_format.keep_with_next=True
    head=sec.header.paragraphs[0];head.text='LLM ADVICE  /  SPATIAL COOPERATION                                      RESEARCH BRIEF · 30 OCT 2026'
    for run in head.runs:run.font.size=Pt(7.2);run.font.color.rgb=RGBColor.from_string(MUTED)
    foot=sec.footer.paragraphs[0];foot.text='Preliminary evidence · Two seeds per setting · Same GPT advice table                 '
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
                            if k=='cards' and ri==0:run.font.size=Pt(15);run.font.color.rgb=RGBColor.from_string(TEAL)
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
                    for run in p.runs:run.font.size=Pt(10.2)
                if k=='subtitle':
                    for run in p.runs:run.font.size=Pt(11.5);run.font.color.rgb=RGBColor.from_string(MUTED)
    doc.core_properties.title='When Trust Adapts: Preliminary SPD Results'
    doc.core_properties.subject='Adaptive individual trust: simulation results and mechanism audit'
    doc.core_properties.author='LLM Advice Research'
    path=OUT/'Adaptive_Trust_Report.docx';doc.save(path);return path


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

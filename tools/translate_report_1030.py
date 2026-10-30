"""Translate and typeset the author's 30 October report without changing its source.

Images and the payoff-matrix OMML are preserved from the original DOCX. The
experimental prompt retains its recorded wording. Export the resulting DOCX
through Word for a matching PDF and pagination check.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
ALIAS = Path(str(ROOT).lower())
if ALIAS.exists() and ALIAS.samefile(ROOT):
    ROOT = ALIAS
FOLDER = ROOT / 'reports/Report_1030'
SOURCE = FOLDER / 'Report10.30.docx'
TARGET = FOLDER / 'Report10.30_English.docx'
NAVY, TEAL, MUTED, PALE = '17394F', '008F85', '536A78', 'EEF4F6'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}

SETUP = [
    'We study a spatial Prisoner’s Dilemma on an N × N periodic lattice, with N = 50. Each individual interacts with its four orthogonal neighbors. With rows and columns ordered as C (cooperate) and D (defect), the focal player’s payoff matrix is',
    'Fermi imitation is the default strategy-update rule. A fraction P (0 ≤ P ≤ 1) of individuals is eligible to receive LLM advice. At each update, an eligible individual adopts the recommended action with probability λ (0 ≤ λ ≤ 1).',
    'The evaluated models are identified in the logs as Qwen3.8-27B, DeepSeek v4.1 Flash and GPT-5.5. Qwen requests specified medium reasoning effort, and GPT response metadata reported medium reasoning effort. DeepSeek reasoning effort was not recorded, so a common default across all three models cannot be confirmed.',
    'LLM advice is collected offline for every possible action configuration of a 3 × 3 local grid. Since each of the nine positions can contain either C or D, one complete sampling round (n = 1) consists of 2⁹ = 512 independent queries, yielding one response per configuration.',
]
NEXT = [
    ('Increase advice sampling.', 'Increase the number of responses per configuration to n = 5 and assess response variability under identical local states. We expect this variability to be relatively low; this expectation remains to be tested.'),
    ('Introduce individual adaptation.', 'Make the advice-adoption probability adaptive and individual-specific, denoted by λᵢ. Increase an individual’s adoption probability when its payoff improves after following LLM advice, reflecting greater trust in the model; decrease it when its payoff worsens.'),
    ('Select the simulation horizon.', 'Determine an appropriate range for the total simulation horizon, T.'),
]
CAPTIONS = [
    'Figure 1. Final-window cooperation for (a) Qwen and (b) GPT, shown as percentages. Each cell averages cooperation over steps 9001–10000 and then over the two matched simulation seeds. The four panels within each model correspond to λ = 0, 0.1, 0.5 and 1. These are cooperation levels, not differences between models.',
    'Figure 2. Recorded recommendations conditional on the focal player’s current action. Each row weights its 256 configurations equally. DeepSeek has one reply per configuration, while GPT and Qwen have two; the cell counts retain these different denominators. These are advice frequencies, not realized simulation transition rates.',
    'Figure 3. Advice to cooperate by current action and number of cooperating direct neighbors, pooling corner arrangements. For a current defector with four cooperating direct neighbors, C recommendations are 0/16 for DeepSeek, 5/32 for GPT and 29/32 for Qwen. Small per-state sample sizes limit inference about stable model behavior.',
]
CONTRAST = 'DeepSeek frequently recommends retaining C but rarely recommends D → C. Qwen recommends D → C much more often in cooperative neighborhoods. This contrast suggests a mechanism to test, rather than a demonstrated cause of the simulation gaps.'


def set_color(run, value):
    run.font.color.rgb = RGBColor.from_string(value)


def para(doc, text='', style=None, size=None, bold=False, color=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    if size is not None:run.font.size = Pt(size)
    if bold:run.bold=True
    if color:set_color(run,color)
    return p


def heading(doc, text, level=1):
    return doc.add_paragraph(text, style=f'Heading {level}')


def picture(doc, blob, width):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(io.BytesIO(blob), width=Inches(width))


def caption(doc, text):
    p=para(doc,text,'Caption')
    p.paragraph_format.keep_together=True
    return p


def format_table(doc, source):
    t=doc.add_table(rows=len(source.rows),cols=2)
    t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=[1.48,5.48]
    for col,width in zip(t.columns,widths):col.width=Inches(width)
    for i,row in enumerate(source.rows):
        for j,cell in enumerate(row.cells):
            dest=t.cell(i,j);dest.width=Inches(widths[j]);dest.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
            p=dest.paragraphs[0];p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(4);p.paragraph_format.line_spacing=1.05
            r=p.add_run(cell.text);r.font.size=Pt(9)
            if i==0:
                r.bold=True
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),PALE);dest._tc.get_or_add_tcPr().append(shade)
            borders=OxmlElement('w:tcBorders');bottom=OxmlElement('w:bottom')
            for key,value in [('val','single'),('sz','4'),('color','DCE5EA')]:bottom.set(qn('w:'+key),value)
            borders.append(bottom);dest._tc.get_or_add_tcPr().append(borders)
            margin=OxmlElement('w:tcMar')
            for side in ['left','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'100');el.set(qn('w:type'),'dxa');margin.append(el)
            dest._tc.get_or_add_tcPr().append(margin)
        t.rows[i]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))


def configure(doc):
    sec=doc.sections[0]
    sec.page_width=Inches(8.2677);sec.page_height=Inches(11.6929)
    sec.top_margin=sec.bottom_margin=Inches(.62)
    sec.left_margin=sec.right_margin=Inches(.65)
    sec.header_distance=sec.footer_distance=Inches(.28)
    normal=doc.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(10.2)
    normal.font.color.rgb=RGBColor.from_string(NAVY)
    normal.paragraph_format.line_spacing=1.1;normal.paragraph_format.space_after=Pt(5)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',24),('Heading 1',16),('Heading 2',11)]:
        s=doc.styles[name];s.font.name='Arial';s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor.from_string(NAVY)
        s.paragraph_format.space_before=Pt(4 if name!='Title' else 0);s.paragraph_format.space_after=Pt(7)
        s.paragraph_format.keep_with_next=True;s.paragraph_format.line_spacing=1.05
    cap=doc.styles['Caption'];cap.font.name='Arial';cap.font.size=Pt(8.6);cap.font.italic=False;cap.font.color.rgb=RGBColor.from_string(MUTED)
    cap.paragraph_format.line_spacing=1.05;cap.paragraph_format.space_after=Pt(7)
    lang=OxmlElement('w:lang');lang.set(qn('w:val'),'en-US');normal.element.get_or_add_rPr().append(lang)
    h=sec.header.paragraphs[0];h.text='LLM ADVICE IN SPATIAL GAMES                                            PRELIMINARY RESULTS · 30 OCT 2026'
    for r in h.runs:r.font.name='Arial';r.font.size=Pt(7.2);set_color(r,MUTED)
    f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=f.add_run('Page ');r.font.size=Pt(8);set_color(r,MUTED)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
    doc.core_properties.title='Preliminary Results: LLM Advice in the Spatial Prisoner’s Dilemma'
    doc.core_properties.subject='English translation and formatting of the author’s 30 October 2026 report'
    doc.core_properties.author=''


def main():
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    original=Document(SOURCE)
    assert len(original.inline_shapes)==4 and len(original.tables)==1
    blocks=list(original.iter_inner_content())
    with zipfile.ZipFile(SOURCE) as z:
        images=[z.read(f'word/media/image{i}.png') for i in range(1,5)]
        tree=etree.fromstring(z.read('word/document.xml'))
    matrices=tree.xpath('//m:oMath[m:d/m:e/m:m]',namespaces=NS)
    assert len(matrices)==1

    system_parts=[]
    for i in range(44,72):
        text=blocks[i].text
        if i==44:text=text[text.index('You are a decision advisor'):]
        system_parts.append(text.replace('\xa0',' '))
    system='\n'.join(system_parts).strip()
    audited=(ROOT/'reports/advice_deepseek_gpt_qwen_20261029/data/common_system_prompt.txt').read_text().strip()
    assert ' '.join(system.split())==' '.join(audited.split())
    user='Game state:\n\n'+'\n'.join(blocks[i].text for i in range(73,83)).strip()
    assert user.endswith('"""')
    user=user[:-3]

    doc=Document();configure(doc)
    para(doc,'Preliminary Results','Title')
    para(doc,'LLM Advice in the Spatial Prisoner’s Dilemma',size=12,color=MUTED)
    heading(doc,'1. Experimental setup')
    para(doc,SETUP[0])
    p=para(doc,'A = ');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(deepcopy(matrices[0]));p.add_run(',   1 < b < 2.');p.paragraph_format.space_after=Pt(7)
    for text in SETUP[1:]:para(doc,text)
    heading(doc,'Study design',2)
    format_table(doc,original.tables[0])

    doc.add_page_break()
    heading(doc,'2. Cooperation across the parameter scan')
    para(doc,'(a) Qwen',size=11,bold=True,color=TEAL).paragraph_format.keep_with_next=True
    picture(doc,images[0],6.95)
    para(doc,'(b) GPT',size=11,bold=True,color='287AB5').paragraph_format.keep_with_next=True
    picture(doc,images[1],6.95)
    caption(doc,CAPTIONS[0])

    doc.add_page_break()
    heading(doc,'3. Local advice patterns')
    picture(doc,images[2],6.65)
    caption(doc,CAPTIONS[1])
    picture(doc,images[3],6.65)
    caption(doc,CAPTIONS[2])
    p=para(doc);r=p.add_run('Main contrast. ');r.bold=True;p.add_run(CONTRAST)
    heading(doc,'4. Next round')
    for i,(title,text) in enumerate(NEXT,1):
        p=para(doc);r=p.add_run(f'{i}. {title} ');r.bold=True;p.add_run(text)
        p.paragraph_format.keep_together=True

    doc.add_page_break()
    heading(doc,'Appendix A. Prompt templates')
    p=para(doc,'Each query is independent and contains no previous interaction history. The system prompt and user-message template are reproduced below.');p.paragraph_format.space_after=Pt(5)
    for title,content in [('System prompt',system),('User-message template',user)]:
        heading(doc,title,2)
        for line in content.splitlines():
            if not line.strip():continue
            p=para(doc,line,size=9.1)
            p.paragraph_format.space_after=Pt(2.2);p.paragraph_format.line_spacing=1.02
            if line.startswith('   - '):p.paragraph_format.left_indent=Inches(.12)
            if line in ['The game follows these rules:','Observation format:']:
                p.runs[0].bold=True;p.paragraph_format.keep_with_next=True
    p=para(doc,'Prompt wording is preserved as used, including original grammatical errors. Code quotation delimiters have been removed.',size=8,color=MUTED)
    p.paragraph_format.space_before=Pt(4)
    doc.save(TARGET)

    with zipfile.ZipFile(TARGET) as z:
        chinese=[]
        for n in z.namelist():
            if n.startswith('word/') and n.endswith('.xml'):
                t=etree.fromstring(z.read(n))
                for text in t.xpath('//w:t/text()|//m:t/text()',namespaces=NS):
                    if re.search(r'[\u3400-\u9fff]',text):chinese.append((n,text))
        assert not chinese, chinese
        out_images=[z.read(n) for n in z.namelist() if n.startswith('word/media/') and not n.endswith('/')]
        assert sorted(hashlib.sha256(x).hexdigest() for x in images)==sorted(hashlib.sha256(x).hexdigest() for x in out_images)
        assert z.testzip() is None
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before
    review=FOLDER/'review';review.mkdir(exist_ok=True)
    (review/'english_build_checks.json').write_text(json.dumps({
        'source_sha256':before,'english_docx_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        'source_unchanged':True,'embedded_images_preserved':4,'payoff_matrix_preserved':True,
        'visible_Chinese_text_nodes':0,'system_prompt_matches_audited_text_ignoring_whitespace':True,
        'planned_advice_repeats':5,'adaptive_lambda_proposal_retained':True,
        'unchanged_payoff_case':'Not specified in the original; no rule introduced.'
    },indent=2)+'\n',encoding='utf-8')
    notes='''English edition: translation and editorial notes

Source: Report10.30.docx (left unchanged).
Output: Report10.30_English.docx; a matching Word-rendered PDF is supplied.

All Chinese prose has been translated into English. The four original figures,
their numerical data, the design table and the native Word payoff matrix are
retained. Headings, font sizes, paragraph spacing, table styling and figure
captions have been standardized; blank-paragraph pagination was removed.

The author's next-round plan remains n = 5, individual adaptive advice adoption,
and selection of the total horizon T. The expectation of limited response
variability remains explicitly an expectation. No adoption-update equation or
rule for an unchanged payoff has been invented.

Two scientific wording qualifications were made explicitly:
1. Figure 1 shows absolute cooperation percentages, not percentage-point
   differences. Its caption now states this and labels Qwen above GPT.
2. The draft says all models default to medium reasoning effort. Prior audited
   logs establish a medium request for Qwen and medium response metadata for
   GPT, but no recorded DeepSeek effort. The English body gives that distinction.
   Model names are presented as logged identifiers. The draft's description of
   Qwen and DeepSeek as open-source is not asserted as a verified licensing fact;
   local request logs do not establish licensing or backend identity.

The prompt appendix preserves the experimental wording, including "to maximizes"
and "game.?". Only quote/code delimiters were removed. "Game state:" was restored
at the start of the user-message template to match the recorded template. The
system prompt matches the audited common prompt after whitespace normalization.
'''
    (FOLDER/'English_Edition_Notes.txt').write_text(notes,encoding='utf-8')
    print(TARGET)
    print('Original unchanged; four original images and native payoff matrix preserved; zero Chinese text nodes.')


if __name__=='__main__':main()

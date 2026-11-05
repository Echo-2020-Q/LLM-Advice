"""Build the 1 November report from audited, merged five-answer advice tables.

Consumes local saved data and figures only. Makes no model or simulation calls.
Run merge_five_advice.py, analyze_five_advice.py and plot_five_advice.py first.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
LOWER = Path(str(ROOT).lower())
if LOWER.exists() and LOWER.samefile(ROOT):
    ROOT = LOWER
OUT = ROOT / "reports/1101_3LLM_Analysis"
DATA = OUT / "data"
FIG = OUT / "figures"
MODELS = ["GPT", "DeepSeek", "Qwen"]
WIDTH = A4[0] - 84
NAVY, TEAL, MUTED, PALE = "163348", "087F82", "536977", "EFF5F7"


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(x, places=2):
    return f"{100 * float(x):.{places}f}%"


def pp(x, signed=False):
    return f"{100 * float(x):+.{2}f}" if signed else f"{100 * float(x):.2f}"


def b(kind, text="", **kwargs):
    return dict(kind=kind, text=text, **kwargs)


def content():
    summary = {row["model"]: row for row in read_csv("model_summary.csv")}
    pair = read_csv("model_pair_summary.csv")
    center = {row["model"]: row for row in read_csv("center_pair_summary.csv")}
    batch = {row["model"]: row for row in read_csv("batch_drift_summary.csv")
             if row["center"] == "all"}
    audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert audit["records_validated"] == 7680
    assert audit["states_per_model"] == 512
    assert audit["answers_per_state_per_model"] == 5
    # Confirm the displayed summaries directly from the canonical merged rows.
    with (OUT / "merged/combined_answers.csv").open(encoding="utf-8-sig", newline="") as handle:
        replies = list(csv.DictReader(handle))
    expected_c_totals = {"GPT": 707, "DeepSeek": 1004, "Qwen": 1466}
    expected_examples = {
        "CCCCCCCCC": {"GPT": 5, "DeepSeek": 5, "Qwen": 5},
        "DDDDDDDDD": {"GPT": 0, "DeepSeek": 0, "Qwen": 0},
        "CCCCDCCCC": {"GPT": 2, "DeepSeek": 0, "Qwen": 5},
        "CDCDCDCDC": {"GPT": 0, "DeepSeek": 5, "Qwen": 0},
    }
    for model in MODELS:
        rows = [row for row in replies if row["model_label"] == model]
        counts = {}
        for row in rows:
            counts.setdefault(row["grid_key"], []).append(int(row["is_c"]))
        assert len(rows) == 2560 and len(counts) == 512
        assert all(len(values) == 5 for values in counts.values())
        assert sum(int(row["is_c"]) for row in rows) == int(summary[model]["C_answers"])
        assert int(summary[model]["C_answers"]) == expected_c_totals[model]
        for state, expected in expected_examples.items():
            assert sum(counts[state]) == expected[model]
        agreement = sum((sum(values) * (sum(values) - 1) +
                         (5 - sum(values)) * (4 - sum(values))) / 20
                        for values in counts.values()) / 512
        assert math.isclose(agreement, float(summary[model]["within_state_pair_agreement"]))

    overview = [["Observed metric", *MODELS]]
    for label, field in [
        ("C recommendations / all replies", "p_C"),
        ("C recommendations / current-C replies", "p_C_given_current_C"),
        ("C recommendations / current-D replies", "p_C_given_current_D"),
        ("Recommendation retains current action", "current_action_retention"),
    ]:
        overview.append([label, *[pct(summary[m][field]) for m in MODELS]])

    consistency = [["Observed metric", *MODELS]]
    consistency.extend([
        ["Five replies all agree (states / 512)", *[summary[m]["unanimous_states"] + " / 512" for m in MODELS]],
        ["States with both C and D replies", *[summary[m]["mixed_states"] for m in MODELS]],
        ["Agreement of two distinct saved replies", *[pct(summary[m]["within_state_pair_agreement"]) for m in MODELS]],
    ])

    paired = [["Model pair", "Different majority", "Mean absolute gap"]]
    for row in pair:
        paired.append([
            row["first_model"] + " vs " + row["second_model"],
            f'{row["majority_disagreement_states"]} / 512 ({pct(row["majority_disagreement_fraction"])})',
            pp(row["mean_absolute_p_C_difference"]) + " pp",
        ])

    centered = [["Matched-neighborhood metric", *MODELS]]
    centered.extend([
        ["Mean p_C(center C) − p_C(center D)", *[pp(center[m]["mean_center_C_minus_D_difference"]) + " pp" for m in MODELS]],
        ["Positive / equal / negative pairs", *[" / ".join(center[m][k] for k in ("positive_pairs", "zero_pairs", "negative_pairs")) for m in MODELS]],
    ])

    batches = [["Batch comparison", *MODELS]]
    batches.extend([
        ["Earlier + later replies per state", *[batch[m]["old_repeats_per_state"] + " + " + batch[m]["new_repeats_per_state"] for m in MODELS]],
        ["Overall C change (later − earlier)", *[pp(batch[m]["mean_delta_new_minus_old"], signed=True) + " pp" for m in MODELS]],
        ["Mean absolute within-state C change", *[pp(batch[m]["mean_absolute_state_change"]) + " pp" for m in MODELS]],
    ])

    pages = [
        [
            b("kicker", "RESEARCH UPDATE  /  SAVED ADVICE TABLES"),
            b("title", "Three Models,\nFive Answers per State"),
            b("subtitle", "GPT, DeepSeek and Qwen in the spatial Prisoner’s Dilemma"),
            b("meta", "1 November 2026  |  512 local states × 5 replies × 3 models"),
            b("lead", "The three saved advice tables differ most clearly in whether they encourage a current defector to cooperate. Qwen does this frequently, DeepSeek rarely does, and GPT lies between them. DeepSeek most often preserves the focal player’s current action; GPT shows the highest agreement across its five recorded answers."),
            b("figure", file="01_overview.png", number=1),
            b("caption", "Figure 1. Three distinct summaries. The horizontal categories identify the models; vertical values are percentages. Overall C frequency counts 2,560 replies per model. Action retention counts replies matching the focal player’s current C/D action. Repeat agreement averages the fraction of matching pairs among the 10 distinct reply pairs for each of the 512 states. These metrics measure different properties."),
            b("table", rows=overview, widths=[.46, .18, .18, .18]),
            b("note", "Scope. These are final successful action recommendations, merged from six collection batches. All 512 binary 3×3 grids receive equal weight. The analysis characterizes the saved tables; it does not measure simulation cooperation, payoff, decision accuracy, or a general ranking of model quality."),
        ],
        [
            b("kicker", "01  /  CONDITIONAL RECOMMENDATIONS"),
            b("heading", "The current action changes the comparison"),
            b("figure", file="02_transitions.png", number=2),
            b("caption", "Figure 2. Rows give the focal player’s current action; columns give the advised next action. Each row contains 256 states × 5 replies = 1,280 recommendations and sums to 100%. These are suggested changes, not transitions observed in a simulation. In particular, the D→C entry is 7.03% for GPT, 2.27% for DeepSeek and 48.05% for Qwen."),
            b("figure", file="03_neighbors.png", number=3),
            b("caption", "Figure 3. Horizontal axis: the number of cooperating direct neighbors among the four orthogonal neighbors (0–4). Vertical axis: the fraction of C recommendations, grouped separately by the current center action. For each center action, the five x values pool 16, 64, 96, 64 and 16 states, or 80, 320, 480, 320 and 80 replies. Corners and spatial arrangements are averaged within each group; lines connect descriptive group means."),
            b("body", "When all four direct neighbors cooperate but the focal player currently defects, the C recommendation frequency is 17.50% for GPT, 3.75% for DeepSeek and 93.75% for Qwen (80 replies per model). For current cooperators with four cooperating direct neighbors, all three models recommend C in all 80 saved replies."),
            b("note", "DeepSeek’s higher overall C frequency than GPT mainly reflects stronger preservation of existing cooperators. It does not imply that DeepSeek more often converts current defectors into cooperators."),
        ],
        [
            b("kicker", "02  /  LOCAL SPATIAL CONTEXT"),
            b("heading", "Separate direct neighbors from corner context"),
            b("figure", file="04_context.png", number=4),
            b("caption", "Figure 4. Each panel fixes the model and center action. The horizontal axis counts cooperating direct neighbors; the vertical axis counts cooperating diagonal neighbors. Cell color gives the percentage of C recommendations across matching states and their five replies; numbers round it to whole percent. A cell with d direct and q diagonal cooperators contains binomial(4,d) × binomial(4,q) states and five times as many replies; cell sizes therefore differ."),
            b("subhead", "How to read a cell"),
            b("body", "At d = 2 and q = 2, a fixed-center cell pools 6 × 6 = 36 different grids and 180 replies. At d = 0 and q = 4, it pools just one grid and five replies. A value of 100% in the second cell is five observed C answers, not precise knowledge of a model’s underlying response probability."),
            b("subhead", "The corner cells matter as context"),
            b("body", "For a current defector with exactly two cooperating direct neighbors, Qwen’s C frequency rises from 6.67% with no cooperating corners to 73.33% with four (30 replies at each endpoint). The prompt states that corners do not directly contribute to the focal player’s payoff, but they supply context about some neighbors of its direct neighbors. They do not reveal complete surrounding payoffs or future actions. This is a conditional association, not an isolated corner-payoff effect."),
            b("note", "A striking example is CDC / DCD / CDC: a current cooperator with zero cooperating direct neighbors and four cooperating corners. DeepSeek recommends C in 5/5 replies, while GPT and Qwen recommend C in 0/5. Figure 8 shows the exact grid and other diagnostic examples."),
        ],
        [
            b("kicker", "03  /  REPEAT CONSISTENCY"),
            b("heading", "Five answers expose state-specific variability"),
            b("figure", file="05_consistency.png", number=5),
            b("caption", "Figure 5. Horizontal axis: the number k of C recommendations among the five saved replies to exactly the same grid, from 0 to 5. Vertical axis: the percentage of the 512 states in each category; labels above bars give the state counts. The end categories are unanimous D and unanimous C; interior categories contain both answers. Each model’s six bins total 512 states (100%)."),
            b("table", rows=consistency, widths=[.46, .18, .18, .18]),
            b("subhead", "Unanimity and pair agreement answer different questions"),
            b("body", "Unanimity asks whether all five replies match. Pair agreement gives partial credit to mixed states: for a state with k C answers, it equals [k(k−1) + (5−k)(4−k)] / 20. Thus a 5–0 split has agreement 100%, a 4–1 split 60%, and a 3–2 split 40%. Averaging across states avoids creating arbitrary cross-model pairings by repetition number."),
            b("subhead", "Consistency is not correctness"),
            b("body", "GPT has the largest unanimous fraction (74.61%) and highest pair agreement (87.85%) in these saved replies. Qwen has more mixed states, but also more unanimous-C states. Neither result shows which table leads to better long-run play. A consistent D response and a consistent C response receive the same consistency score."),
            b("note", "The five replies come from two collection batches, so observed variation combines repeat variability with possible batch differences. A unanimous 5/5 outcome does not establish deterministic model behavior. No confidence interval is inferred by treating the 2,560 heterogeneous replies as exchangeable trials."),
        ],
        [
            b("kicker", "04  /  BETWEEN-MODEL DIFFERENCES"),
            b("heading", "Compare the same state, then summarize the gap"),
            b("figure", file="06_model_agreement.png", number=6),
            b("caption", "Figure 6. Each panel compares two models on the same 512 grids. The axes are their observed C frequencies k/5, taking values 0%, 20%, …, 100%; the axis labels name the corresponding models. Each cell counts states with that pair of frequencies, and all cells in a panel sum to 512. The diagonal indicates equal observed frequencies; it does not require identical reply sequences."),
            b("table", rows=paired, widths=[.34, .35, .31]),
            b("subhead", "Frequency gaps preserve more information than majority votes"),
            b("body", "A model’s majority is C when at least three of its five saved replies are C; five replies prevent a tied majority. Two models can share a majority while having very different C frequencies, such as 3/5 versus 5/5. The mean absolute gap averages |p_C,A(s) − p_C,B(s)| over the 512 matched states and is measured in percentage points."),
            b("body", "GPT and Qwen have different majority recommendations on 183/512 states (35.74%), the largest majority-disagreement count of the three pairs. DeepSeek and Qwen have the largest mean absolute frequency gap, 33.05 percentage points. The pair with the greatest difference therefore depends on the comparison metric."),
            b("note", "The repetition labels are identifiers for separate request contexts, not paired random draws across providers. A first-reply-to-first-reply comparison would impose an arbitrary alignment. State-level frequencies and majority comparisons use the actual matched input instead."),
        ],
        [
            b("kicker", "07  /  MATCHED CENTER-ACTION COMPARISON"),
            b("heading", "DeepSeek depends most strongly on the current action"),
            b("figure", file="09_center_effect.png", number=9),
            b("caption", "Figure 9. Hold the eight surrounding C/D entries fixed and compare the two possible center actions. The horizontal value is p_C(center C) minus p_C(center D), in percentage points; the vertical value is the percentage of the 256 paired neighbor configurations at that contrast. Positive values indicate more C advice when the focal player already cooperates. Each difference compares two sets of five saved replies."),
            b("table", rows=centered, widths=[.46, .18, .18, .18]),
            b("subhead", "A descriptive dependence, with a clear comparison unit"),
            b("body", "DeepSeek has a positive center-C minus center-D difference in 248 of the 256 paired neighborhoods and no negative differences. Its mean difference is 73.91 percentage points, compared with 41.17 for GPT and 18.44 for Qwen. This matches the transition matrices: DeepSeek often keeps cooperators cooperating but very rarely recommends C to a current defector."),
            b("subhead", "Why this is not a causal measure of anchoring"),
            b("body", "Changing the center action also changes the focal player’s stated current payoff: with d cooperating direct neighbors, the prompt reports d for C and d×b for D. It can also change the surrounding players’ implied payoffs. The paired analysis controls the eight surrounding action entries, but it does not separate current-action persistence from the payoff information carried by that action."),
            b("note", "The 256 differences are a complete comparison over the designed surrounding configurations. They are not 256 independent simulation experiments, and their distribution should not be read as uncertainty about a single causal effect."),
        ],
        [
            b("kicker", "05  /  CHECKING THE MERGED BATCHES"),
            b("heading", "Stable totals can conceal changes within states"),
            b("figure", file="07_batch_comparison.png", number=7),
            b("caption", "Figure 7. Earlier and later batches are compared using the same input states. Model and current-action categories appear on the horizontal axis; the vertical axis gives the fraction of C replies, as labeled. Separate current-C and current-D results prevent offsetting changes from disappearing into the overall mean. Earlier/later sample sizes differ: GPT and Qwen have 2/3 replies per state; DeepSeek has 1/4."),
            b("table", rows=batches, widths=[.46, .18, .18, .18]),
            b("body", "The overall C frequency changes by less than two percentage points for every model. Yet the mean absolute change in a state’s observed C frequency is about 9.96 points for GPT, 15.43 for DeepSeek and 14.88 for Qwen. These larger within-state differences can cancel in the overall average; they can also arise from small samples and do not by themselves establish temporal drift."),
            b("subhead", "Five replies per state; no observed token-limit truncation"),
            b("body", "The five replies per state have been pooled for every model. All batches use identical system text and matched per-state user prompts. Although GPT/Qwen caps increased from 10,240 to 51,200, the maximum recorded output counts in their old/new batches were only 3,844/3,679 for GPT and 5,139/5,238 for Qwen; DeepSeek recorded 2 in both batches. All 7,680 retained responses finished normally, and even the new responses stayed below the old 10,240 cap. No token-limit truncation was observed, so the cap change is not evidence against pooling or an established explanation of answer differences. Batch labels remain available for provenance."),
            b("note", "DeepSeek’s earlier estimate uses one answer per state, so it can only be 0% or 100%. Its within-state batch gap is not directly comparable as a stability score to the 2-versus-3 designs. The correct conclusion is that broad aggregate patterns persist, with material uncertainty about individual-state frequencies."),
        ],
        [
            b("kicker", "06  /  CONCRETE INPUTS AND SAVED ANSWERS"),
            b("heading", "Read the disagreements in their actual neighborhoods"),
            b("figure", file="08_examples.png", number=8),
            b("caption", "Figure 8. Each 3×3 grid shows a supplied local observation; the middle cell is the focal player. Up, down, left and right cells are direct interaction partners, and the four corners supply additional context. C/D labels denote current actions. The response displays give all five saved recommendations, separated into earlier and later batches, plus the C count. These states illustrate shared endpoints and clear differences, not their frequency in a simulation."),
            b("table", widths=[.46, .18, .18, .18], rows=[
                ["Grid (rows separated by /)", "GPT C / 5", "DeepSeek C / 5", "Qwen C / 5"],
                ["CCC / CCC / CCC", "5 / 5", "5 / 5", "5 / 5"],
                ["DDD / DDD / DDD", "0 / 5", "0 / 5", "0 / 5"],
                ["CCC / CDC / CCC", "2 / 5", "0 / 5", "5 / 5"],
                ["CDC / DCD / CDC", "0 / 5", "5 / 5", "0 / 5"],
            ]),
            b("body", "The isolated defector in an otherwise cooperative window (CCC / CDC / CCC) produces a clear contrast: Qwen advises cooperation in all five replies, DeepSeek in none, and GPT in two. The isolated direct-neighbor pattern in the final row produces a different split, showing why a single global cooperation score cannot summarize the full policy."),
            b("note", "All-C and all-D are unanimous shared endpoints of these saved tables. This is a property of the recorded lookup responses. It does not show how often a spatial simulation will reach either endpoint or how long it will take."),
        ],
        [
            b("kicker", "08  /  METHODS, LIMITS AND REUSE"),
            b("heading", "What the evidence supports—and what to test next"),
            b("subhead", "Data construction and traceability"),
            b("body", "The merge preserves every final successful C/D answer: 512 states × 5 replies = 2,560 per model, 7,680 in total. It assigns new within-model repetition identifiers while retaining original request indices, source batches, record paths and hashes. It does not replace the five answers by a majority vote. All referenced final records, source summaries and request prompts were checked; the existing EmpiricalAdvisor loaded each merged table and reproduced every C/D histogram without running a simulation."),
            b("table", widths=[.29, .71], rows=[
                ["Audit item", "Meaning for interpretation"],
                ["Input space", "All 2⁹ = 512 C/D grids, with 256 current-C and 256 current-D states. This equal weighting is not a simulated state-visitation distribution."],
                ["Requests", "Separate two-message contexts: one system prompt and one supplied observation. No numerical b is given; only 1 < b < 2. Advice targets expected cumulative personal payoff."],
                ["Final replies", f'{audit["archived_attempts_excluded"]} archived prior attempts are excluded; the analysis describes final successful saved replies. It is not an audit of every raw generation attempt.'],
                ["Model metadata", "Names identify logged endpoints. GPT responses record medium reasoning effort; Qwen requests explicitly set medium. DeepSeek reasoning settings are unrecorded. Identical provider behavior or undocumented defaults cannot be assumed."],
                ["Finite repeats", "Five answers produce empirical frequencies in steps of 20%. Zero or five C replies do not establish a true probability of zero or one."],
            ]),
            b("subhead", "Next analysis priorities"),
            b("number", "1  Preserve all three stochastic tables in downstream experiments. Compare them with the same lattice settings, adoption rule and simulation-seed design, and report advice-table uncertainty separately from seed variability."),
            b("number", "2  For a cleaner repeat study, collect additional answers under a fixed, fully recorded protocol. Prioritize states with mixed replies, large model differences and frequent visits in simulations. Record all attempts without treating retries as extra successful repetitions."),
            b("number", "3  Add visitation-weighted summaries after trajectories are available. Track the states where each model is actually consulted, and test whether D→C differences affect cooperation survival, formation of cooperative clusters and population payoff."),
            b("subhead", "Files for inspection and reuse"),
            b("source", "merged/{GPT,DeepSeek,Qwen}/ contains one compatible five-answer advice table per model. merged/combined_answers.csv preserves provenance; data/aligned_states.csv aligns the 512 inputs. data/ contains plotted statistics, audit.json and the source manifest. figures/ supplies standalone plots. Three_LLM_Advice_Explorer.html lets you inspect the states, five replies and batch membership interactively. Analysis_English.txt is the readable text edition; report_content.json preserves the report’s authored blocks."),
            b("note", "No new LLM calls or spatial-game runs were performed for this report. The strongest supported conclusion is a difference in observed recommendation policies under the supplied prompt, with limited repeat sampling and documented collection-batch differences."),
        ],
    ]
    # Present figures in numerical order while giving the center comparison its own page.
    return pages[:5] + [pages[6], pages[7], pages[5], pages[8]]


def color(value):
    return colors.HexColor("#" + value.lstrip("#"))


def register_fonts():
    folder = Path("/mnt/c/Windows/Fonts")
    if (folder / "arial.ttf").exists():
        files = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]
    else:
        folder = Path("/usr/share/fonts/truetype/dejavu")
        files = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"]
    for name, filename in zip(["Report", "Report-Bold", "Report-Italic", "Report-BoldItalic"], files):
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold", italic="Report-Italic", boldItalic="Report-BoldItalic")


def styles():
    base = dict(fontName="Report", fontSize=9.5, leading=13.1,
                textColor=color(NAVY), spaceAfter=8)
    s = {name: ParagraphStyle(name, **base) for name in [
        "body", "lead", "note", "source", "caption", "meta", "subtitle", "table", "number",
    ]}
    s["lead"].fontSize = 10.2
    s["lead"].leading = 14.0
    s["lead"].spaceAfter = 11
    for name in ["note", "source", "caption", "meta"]:
        s[name].fontSize = 8.35
        s[name].leading = 11.3
        s[name].textColor = color(MUTED)
    s["source"].fontSize = 8.0
    s["source"].leading = 10.5
    s["caption"].spaceAfter = 11
    s["meta"].spaceAfter = 11
    s["subtitle"].fontSize = 11.5
    s["subtitle"].leading = 15.5
    s["subtitle"].textColor = color(MUTED)
    s["table"].fontSize = 8.6
    s["table"].leading = 11.3
    s["table"].spaceAfter = 0
    for name, size, leading, after, col in [
        ("kicker", 8.8, 12, 10, TEAL), ("title", 29, 32, 10, NAVY),
        ("heading", 20.5, 24.5, 11, NAVY), ("subhead", 11.4, 15, 6, NAVY),
    ]:
        s[name] = ParagraphStyle(name, fontName="Report-Bold", fontSize=size,
                                 leading=leading, spaceAfter=after,
                                 textColor=color(col), keepWithNext=True)
    return s


def rich(text):
    return escape(str(text)).replace("\n", "<br/>")


def pdf_table(block, style):
    rows = [[Paragraph(("<b>" + rich(t) + "</b>") if i == 0 else rich(t), style["table"])
             for t in row] for i, row in enumerate(block["rows"])]
    table = Table(rows, colWidths=[WIDTH * fraction for fraction in block["widths"]], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), color(PALE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
        ("LINEBELOW", (0, 0), (-1, 0), .6, color("BDD0D8")),
        ("LINEBELOW", (0, 1), (-1, -1), .35, color("DCE6EA")),
    ]))
    return [table, Spacer(1, 12)]


def build_pdf(pages):
    register_fonts()
    style = styles()
    story = []
    page_content_heights = []
    for i, blocks in enumerate(pages):
        if i:
            story.append(PageBreak())
        height = 0.0
        for block in blocks:
            kind = block["kind"]
            if kind == "table":
                objects = pdf_table(block, style)
            elif kind == "figure":
                path = FIG / block["file"]
                with PILImage.open(path) as im:
                    width_px, height_px = im.size
                objects = [Image(str(path), width=WIDTH, height=WIDTH * height_px / width_px), Spacer(1, 4)]
            else:
                objects = [Paragraph(rich(block["text"]), style[kind])]
            for obj in objects:
                _, obj_height = obj.wrap(WIDTH, 10000)
                height += obj_height + obj.getSpaceBefore() + obj.getSpaceAfter()
            story.extend(objects)
        page_content_heights.append(round(height, 1))

    destination = OUT / "Three_LLM_Advice_Analysis.pdf"
    doc = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=48, bottomMargin=43,
                            title="Three Models, Five Answers per State",
                            author="LLM Advice Research",
                            subject="Audited GPT, DeepSeek and Qwen action recommendations across 512 local states")

    def chrome(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(color("D4E0E7"))
        canvas.setLineWidth(.6)
        canvas.line(42, A4[1] - 32, A4[0] - 42, A4[1] - 32)
        canvas.line(42, 31, A4[0] - 42, 31)
        canvas.setFont("Report", 7.3)
        canvas.setFillColor(color(MUTED))
        canvas.drawString(42, A4[1] - 24, "LLM ADVICE  /  FIVE-REPLY COMPARISON")
        canvas.drawRightString(A4[0] - 42, A4[1] - 24, "RESEARCH REPORT · 1 NOV 2026")
        canvas.drawString(42, 19, "512 states per model · Descriptive advice-table analysis")
        canvas.drawRightString(A4[0] - 42, 19, f"{document.page} / {len(pages)}")
        canvas.restoreState()

    doc.build(story, onFirstPage=chrome, onLaterPages=chrome)
    reader = PdfReader(destination)
    if len(reader.pages) != len(pages):
        raise ValueError(f"Expected {len(pages)} pages, got {len(reader.pages)}; estimated heights: {page_content_heights}")
    expected = [next(block["text"] for block in blocks if block["kind"] in ("title", "heading")) for blocks in pages]
    for i, (page, title) in enumerate(zip(reader.pages, expected), 1):
        text = page.extract_text()
        assert " ".join(title.split()) in " ".join(text.split()), (i, title)
        assert "\ufffd" not in text
    return destination, page_content_heights


def write_text(pages):
    lines = ["THREE MODELS, FIVE ANSWERS PER STATE", "GPT, DeepSeek and Qwen in the spatial Prisoner’s Dilemma", "1 November 2026", ""]
    for i, page in enumerate(pages, 1):
        lines.append(f"PAGE {i}")
        for block in page:
            if block["kind"] == "table":
                for row in block["rows"]:
                    lines.append(" | ".join(row))
                lines.append("")
            elif block["kind"] == "figure":
                lines.extend([f'Figure file: figures/{block["file"]}', ""])
            else:
                lines.extend([block["text"], ""])
    (OUT / "Analysis_English.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "report_content.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_preview(path):
    import pypdfium2 as pdfium
    from PIL import ImageDraw
    review = OUT / "review"
    review.mkdir(exist_ok=True)
    pdf = pdfium.PdfDocument(str(path))
    thumbnails = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=1.6)
        rendered = bitmap.to_pil()
        rendered.save(review / f"report_page_{i + 1:02d}.png")
        thumbnail = rendered.copy()
        thumbnail.thumbnail((360, 510))
        thumbnails.append(thumbnail)
        bitmap.close()
        page.close()
    pdf.close()
    sheet = PILImage.new("RGB", (3 * 380, 3 * 548), "#dfe7ec")
    draw = ImageDraw.Draw(sheet)
    for i, thumb in enumerate(thumbnails):
        x, y = (i % 3) * 380 + 10, (i // 3) * 548 + 26
        sheet.paste(thumb, (x, y))
        draw.text((x, y - 18), f"Page {i + 1}", fill="#17394f")
    sheet.save(review / "report_contact_sheet.png")
    return len(thumbnails)


def main():
    pages = content()
    write_text(pages)
    path, heights = build_pdf(pages)
    rendered_pages = render_preview(path)
    metrics = {
        "pdf": str(path.relative_to(ROOT)),
        "pages": len(pages),
        "content_heights_pt": heights,
        "content_frame_height_pt": round(A4[1] - 48 - 43 - 12, 1),
        "figure_count": sum(block["kind"] == "figure" for page in pages for block in page),
        "verified_summary_against_merged_replies": True,
        "verified_four_example_state_C_counts": True,
        "rendered_preview_pages": rendered_pages,
        "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (OUT / "review").mkdir(exist_ok=True)
    (OUT / "review/report_validation.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Build the three-page SafeLeWM explainer. Requires ReportLab and DejaVu fonts.

Run from the repository root: python3 docs/research/build_SafeLeWM.py
All prose and layout are editable here. This script performs no experiments.
"""

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "SafeLeWM.pdf"
INK = colors.HexColor("#183042")
TEAL = colors.HexColor("#176B67")
MUTED = colors.HexColor("#596871")
PALE = colors.HexColor("#EDF5F3")
RULE = colors.HexColor("#D5DFE2")
REGULAR = "DejaVuSans"
BOLD = "DejaVuSans-Bold"


def build():
    for name in (REGULAR, BOLD):
        pdfmetrics.registerFont(TTFont(name, f"/usr/share/fonts/truetype/dejavu/{name}.ttf"))
    pdfmetrics.registerFontFamily(REGULAR, normal=REGULAR, bold=BOLD)
    styles = {
        "body": ParagraphStyle(
            "Body",
            fontName=REGULAR,
            fontSize=10.5,
            leading=14.3,
            textColor=INK,
            spaceAfter=7,
            alignment=TA_LEFT,
        ),
        "heading": ParagraphStyle(
            "Heading",
            fontName=BOLD,
            fontSize=12.8,
            leading=16,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "title": ParagraphStyle(
            "Title",
            fontName=BOLD,
            fontSize=33,
            leading=38,
            textColor=INK,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName=REGULAR,
            fontSize=13,
            leading=17,
            textColor=MUTED,
            spaceAfter=13,
            keepWithNext=True,
        ),
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            fontName=BOLD,
            fontSize=8.5,
            leading=11,
            textColor=TEAL,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "page_title": ParagraphStyle(
            "PageTitle",
            fontName=BOLD,
            fontSize=22,
            leading=27,
            textColor=INK,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName=REGULAR,
            fontSize=8.2,
            leading=10.8,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName=REGULAR,
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=8,
        ),
    }
    width = A4[0] - 38 * mm
    story = []

    def p(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        p(text, "heading")

    def callout(text):
        box = Table([[Paragraph(text, styles["body"])]], colWidths=[width])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 11),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(box)
        story.append(Spacer(1, 5))

    def pipeline(labels=None, caption="Observe what happened, then plan again."):
        drawing = Drawing(width, 68)
        gap = 17
        box_width = (width - 2 * gap) / 3
        labels = labels or [
            ["Propose possible", "action plans"],
            ["Predict goal progress", "and check safety"],
            ["Refine safe plans", "and execute a few actions"],
        ]
        for index, lines in enumerate(labels):
            x = index * (box_width + gap)
            drawing.add(Rect(x, 17, box_width, 43, fillColor=PALE, strokeColor=None))
            for line_index, text in enumerate(lines):
                drawing.add(
                    String(
                        x + box_width / 2,
                        43 - line_index * 13,
                        text,
                        fontName=REGULAR,
                        fontSize=9.5,
                        fillColor=INK,
                        textAnchor="middle",
                    )
                )
            if index < 2:
                left, right = x + box_width + 3, x + box_width + gap - 3
                drawing.add(Line(left, 38, right, 38, strokeColor=TEAL, strokeWidth=1))
                drawing.add(
                    Polygon(
                        [right, 38, right - 4, 41, right - 4, 35],
                        fillColor=TEAL,
                        strokeColor=None,
                    )
                )
        drawing.add(
            String(
                width / 2,
                2,
                caption,
                fontName=REGULAR,
                fontSize=9,
                fillColor=MUTED,
                textAnchor="middle",
            )
        )
        story.append(drawing)

    p("SAFETYDIAL / RESEARCH EXPLAINER / 29 OCTOBER 2026", "eyebrow")
    p("SafeLeWM", "title")
    p("Helping a frozen world model finish tasks safely", "subtitle")
    callout(
        "<b>The aim:</b> reach the original task goal while staying clear of hazards, "
        "without retraining LeWM."
    )

    heading("1. Add a safety requirement to the task")
    p(
        "LeWM is a world model: it predicts how a scene might change under possible "
        "actions. A planner uses those predictions to choose what to do. We keep the "
        "model's weights frozen and introduce virtual keep-out regions into its tasks. "
        "A route that reaches the goal can now be unacceptable because it crosses a hazard."
    )
    p(
        "In our current PushT setup, the pusher and T-block must reach their recorded "
        "goal positions, with the T-block at the desired orientation. The full pusher "
        "disk must keep an extra distance from hazards, called the safety margin. "
        "The T-block is excluded from the hazard rule. "
        "The zones do not physically block motion, and are supplied separately to the "
        "checker rather than painted into LeWM's observations."
    )

    heading("2. Use SafeCEM to search for safe progress")
    p(
        "CEM, the cross-entropy method, searches by sampling action plans, scoring "
        "them and repeatedly refining the promising ones. SafeCEM adds a safety check: "
        "<b>plans that pass the check take priority, then LeWM's goal score chooses "
        "among them.</b> It executes only the first few actions, "
        "observes the result and replans. Our current runs execute five actions per replan."
    )
    pipeline()
    p(
        "LeWM supplies the predicted task progress. For PushT, the safety checker uses "
        "the current pusher position and velocity, its known motion dynamics and the "
        "hazard geometry. If no checked plan is available, the controller stops and the "
        "episode is unfinished. Stopping safely is not counted as completing the task."
    )

    heading("3. Why the default settings can struggle")
    p(
        "The planner has a limited search budget. Once hazards rule out direct routes, "
        "many candidate plans may be rejected, and the remaining choices may make "
        "little progress. Settings that were useful for the original task can therefore "
        "leave SafeCEM safe but unfinished. Tuning changes how it spends its search "
        "effort so that useful safe alternatives are easier to find."
    )
    p(
        "<b>What stays fixed:</b> the learned model, the task goal and the requested "
        "safety margin. The parameters being adjusted belong to the planner.",
        "caption",
    )

    story.append(PageBreak())
    p("Why tune, and what do we know?", "page_title")
    p(
        "The main knobs control how far the planner looks, where it searches and "
        "how much useful search history it keeps."
    )

    p(
        "<b>Lookahead.</b> A shorter horizon means fewer action variables to search "
        "and can reduce accumulated error in the model's task predictions. Replanning "
        "lets short plans build into a longer maneuver. A longer horizon can still be "
        "valuable when useful progress requires looking further ahead."
    )
    p(
        "<b>Sampling spread.</b> Broader exploration can reveal routes around a "
        "hazard. Too little spread can keep proposing similar blocked plans; too much "
        "can waste samples on unsafe or unhelpful actions."
    )
    p(
        "<b>Linking action blocks (temporal correlation).</b> Related changes across "
        "successive action blocks encourage more coherent candidate plans."
    )
    p(
        "<b>Keeping good plans (candidate retention).</b> Reusing promising safe plans between search "
        "iterations and replans reduces the need to rediscover them."
    )
    callout(
        "<b>A useful tradeoff:</b> in the earlier PushT study, shortening lookahead "
        "from 25 to 10 actions allowed 750 candidate plans instead of 300 per iteration "
        "with roughly the same amount of model prediction work. The planner could look less "
        "far ahead while considering more alternatives."
    )

    heading("What the experiments support")
    p(
        "Earlier fixed-preset tuning improved safe completion on reserved cases: "
        "PushT at 12 px rose from <b>2/36 to 19/36</b>, and Cube at 1 cm from "
        "<b>0/36 to 12/36</b>. Those studies used their older layouts and safety "
        "definitions. Several settings changed together, so these results do not "
        "isolate which individual change caused the improvement."
    )
    p(
        "The latest PushT smoke test used one scene, one seed and a manually chosen "
        "preset. Island, rounded-bar and wide-gate runs completed in <b>26, 40 and "
        "40 actions</b>, maintaining 6 px clearance around the full pusher at recorded "
        "physics steps. The shared unchecked reference finished in 17 actions but "
        "crossed the hazards. It used the same preset, so these videos demonstrate "
        "safety checking rather than a new default-versus-tuned comparison."
    )

    heading("The local problem: choose settings automatically")
    p(
        "Different environments can favor different search tradeoffs. We want one "
        "reusable configurator that tries bounded settings on development tasks, "
        "prioritizes safe completion and then efficiency, and freezes the best "
        "evaluated configuration for testing on fresh cases. It may return different "
        "settings for PushT, TwoRoom and Cube. The safety margin stays fixed."
    )
    p(
        "Random search, Bayesian optimization and evolution are possible outer "
        "tuning methods. The output can initially be a fixed configuration; "
        "an execution-time adaptive controller is a separate extension. No automatic "
        "tuning comparison has yet been run on the new three-shape setup."
    )
    story.append(Spacer(1, 5))
    p(
        "<b>Evidence:</b> Principia, task 18; configs/lewm18/protocol.json; "
        "progress/receipts/20261029-pusht-hazard-smoke.json. "
        '<link href="https://huggingface.co/datasets/Sachioster/SafetyDial-PushT/tree/'
        '05040e52cc9043446520eb1136501ef72e32e8c9/campaigns/pusht-hazards-smoke-20261029" '
        'color="#176B67">View the private smoke-test videos and records.</link>',
        "small",
    )

    story.append(PageBreak())
    p("A safety adapter for frozen LeWM", "page_title")
    p("PROPOSED EXTENSION / NOT YET IMPLEMENTED OR VALIDATED", "eyebrow")
    p(
        "The aim is to replace environment-specific predictive checkers with a small "
        "learned safety adapter. It would translate LeWM's features and candidate "
        "actions into a safety estimate for supplied hazards. <b>Only the adapter "
        "would learn; LeWM's original weights would stay frozen.</b>"
    )
    pipeline(
        labels=[
            ["LeWM features, actions", "and hazard description"],
            ["Small safety adapter", "predicts clearance or risk"],
            ["SafeCEM searches", "for safe task progress"],
        ],
        caption="LeWM supplies the goal score. The adapter supplies the safety estimate.",
    )
    p(
        "<b>Inputs and output.</b> The hazard description specifies the region, the "
        "protected body or point, and the required margin. The adapter would predict "
        "the minimum clearance during an action chunk, or its violation risk. Checking "
        "only the endpoint could miss a hazard crossed along the way."
    )

    heading("Learn from the evidence we already have")
    p(
        "Recorded images can be encoded by frozen LeWM, and saved executed states "
        "can supply clearance and violation labels. Because our hazards are virtual, "
        "one recorded trajectory can be labelled against many hazard placements "
        "without new simulator rollouts. Training and test data must be separated "
        "by original trajectory before relabelling."
    )

    heading("Keep the adapter and tuner separate")
    p(
        "The <b>safety adapter</b> learns to assess proposed behavior. The "
        "<b>automatic configurator</b> chooses SafeCEM's search settings using actual "
        "safe-completion outcomes. Initially, the same training procedure could fit "
        "a separate small adapter for each environment, since different LeWM "
        "checkpoints need not share aligned latent features. The requested safety "
        "margin would remain fixed during tuning."
    )
    callout(
        "<b>The intended experience:</b> supply the hazards, automatically calibrate "
        "the adapter, tune the planner, then run. We would still provide the hazard "
        "meaning and reliable labels for actual behavior."
    )

    heading("The first test is reliable safety prediction")
    p(
        "Latent goal distance is not automatically physical hazard clearance. "
        "Earlier probes showed that accurate readouts of real observations can "
        "still fail on imagined futures. This adapter must therefore be checked "
        "against actual outcomes on held-out trajectories and plans of the kind "
        "CEM proposes. Conservative error allowances would need calibration; a "
        "safety prediction alone is not a guarantee. We would measure completion "
        "alongside violations so that excessive caution is not mistaken for success."
    )
    p(
        '<b>Related reading:</b> <link href="https://arxiv.org/abs/2509.19555" '
        'color="#176B67">AnySafe (2025)</link> studies latent safety filters conditioned '
        "on user-specified constraints. It motivates this direction but does not "
        "validate the proposed LeWM adapter.",
        "small",
    )

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(19 * mm, 15 * mm, A4[0] - 19 * mm, 15 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont(REGULAR, 8)
        canvas.drawString(19 * mm, 10 * mm, "SafeLeWM | Working explanation | 29 October 2026")
        canvas.drawRightString(A4[0] - 19 * mm, 10 * mm, f"{doc.page} / 3")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title="SafeLeWM",
        author="Sachin Mohan",
        subject="Frozen LeWM, safe planning, automatic configuration and a proposed safety adapter",
    )
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == "__main__":
    build()

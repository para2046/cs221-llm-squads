"""Build the 3-slide CS221 deck in the style of NavLM_milestone3.pptx (python-pptx)."""
import copy, json, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

TEMPLATE = "/home/users/para2046/cs221/project/NavLM_milestone3.pptx"
OUT = sys.argv[1]
DATA = json.load(open(sys.argv[2]))  # live numbers

RED, NAVY, GRAY, INK = "C00000", "1F3A68", "555555", "1A1A1A"
LIGHT, BORDER, WHITE = "F6F6F6", "D0D0D0", "FFFFFF"
FONT = "Cambria"
HEADER = "LLM Squads in StarCraft II:  Communication, Memory & Post-Training"
FOOTER = "CS 221  ·  Project Progress  ·  November 2026"

prs = Presentation(TEMPLATE)
# drop the template's slides, keep its master/theme
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
BLANK = [l for l in prs.slide_layouts if l.name == "Blank"][0]


def rgb(h):
    return RGBColor.from_string(h)


def rect(s, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, name=None, lw=0.75):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.08
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line); sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    if name:
        sh.name = name
    return sh


def text(s, x, y, w, h, paras, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         italic=False, name=None, space_after=2, margin=0.04):
    """paras: str | list of (str | list of (text, {opts}))"""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.02)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, o in runs:
            r = p.add_run(); r.text = t
            f = r.font; f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic); f.color.rgb = rgb(o.get("color", color))
    if name:
        tb.name = name
    return tb


def frame(s, n, section, title, subtitle=None):
    rect(s, 0, 0, 10, 0.42, RED, name="Header band")
    rect(s, 0, 0.42, 10, 0.012, NAVY, name="Header rule")
    text(s, 0.4, 0.07, 6.6, 0.28, HEADER, size=10, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.0, 0.07, 2.6, 0.28, section, size=10, color=WHITE, bold=True, align=PP_ALIGN.RIGHT,
         anchor=MSO_ANCHOR.MIDDLE)
    if title:
        text(s, 0.4, 0.58, 9.2, 0.55, title, size=26, color=NAVY, bold=True, name="Title")
        rect(s, 0.4, 1.25, 9.2, 0.012, NAVY, name="Title rule")
    if subtitle:
        text(s, 0.4, 1.33, 9.2, 0.34, subtitle, size=13, color=GRAY, name="Subtitle")
    text(s, 0.4, 6.95, 6.0, 0.25, FOOTER, size=9, color=GRAY)
    text(s, 7.5, 6.95, 2.1, 0.25, f"{n} / 3", size=9, color=GRAY, align=PP_ALIGN.RIGHT)


def label(s, x, y, w, t):
    text(s, x, y, w, 0.28, t, size=11, color=NAVY, bold=True)
    rect(s, x + 0.04, y + 0.29, 0.6, 0.012, NAVY)


# ---------------- slide 1: problem ----------------
s = prs.slides.add_slide(BLANK)
frame(s, 1, "1.  Problem", "Can a Squad of Small LLMs Fight as a Team?",
      "Each of 5 Protoss units runs its own Gemma 4 call on text, with no shared brain and no tactics in the prompt.")
label(s, 0.4, 1.8, 4.4, "ONE LLM CALL PER UNIT, EVERY STEP")
steps = [
    ("Text observation, per unit", "own state  ·  visible allies & enemies  ·  its last 5 steps"),
    ("Gemma 4  (E4B ≈ 4B  or  31B)", "5 independent calls per step, one per unit"),
    ("JSON order + team message", '{"action": "attack_e3", "msg": "finishing e3"}'),
    ("SMACv2 Protoss 5v5", "vs SC2 built-in AI, difficulty 7  ·  random units & spawns"),
]
y = 2.22
for i, (a, b) in enumerate(steps):
    rect(s, 0.4, y, 4.4, 0.62, WHITE, BORDER, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 0.5, y + 0.05, 4.2, 0.28, a, size=12, color=NAVY, bold=True)
    text(s, 0.5, y + 0.32, 4.2, 0.26, b, size=10, color=GRAY)
    if i < len(steps) - 1:
        rect(s, 2.51, y + 0.65, 0.18, 0.17, NAVY, shape=MSO_SHAPE.DOWN_ARROW)
    y += 0.85
rect(s, 0.4, 5.66, 4.4, 1.0, LIGHT, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 0.5, 5.71, 4.2, 0.28, "COMMUNICATION CHANNELS", size=10, color=NAVY, bold=True)
text(s, 0.5, 5.98, 4.2, 0.66, [
    [("C0  ", {"bold": True, "color": NAVY}), ("no channel", {})],
    [("C1  ", {"bold": True, "color": NAVY}), ("+ teammates' state and sightings (JSON)", {})],
    [("C2  ", {"bold": True, "color": NAVY}), ("+ a free message of ≤ 8 words", {})]], size=10, space_after=0)

pic = s.shapes.add_picture(DATA["frame"], Inches(5.15), Inches(1.8), width=Inches(4.45))
pic.line.color.rgb = rgb(BORDER); pic.line.width = Pt(0.75)
text(s, 5.15, 4.33, 4.45, 0.25, "31B squad in C2, step 18: each unit's order (orange) and team chat (white).",
     size=9, color=GRAY, italic=True)
label(s, 5.15, 4.66, 4.45, "RESEARCH QUESTIONS")
qs = [("Q1", "Communication", "Does a team channel help, and does it depend on model size?"),
      ("Q2", "Memory", "Does a short history of each unit's own steps help?"),
      ("Q3", "Post-training", "Can 4B learn from 31B (SFT), and can multi-agent RL push it further?")]
y = 5.1
for q, h, b in qs:
    c = rect(s, 5.15, y + 0.04, 0.42, 0.42, NAVY, shape=MSO_SHAPE.OVAL)
    text(s, 5.15, y + 0.04, 0.42, 0.42, q, size=11, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, margin=0)
    text(s, 5.67, y, 3.93, 0.52, [[(h + ".  ", {"bold": True, "color": NAVY}), (b, {})]], size=11)
    y += 0.58

# ---------------- slide 2: demo ----------------
s = prs.slides.add_slide(BLANK)
frame(s, 2, "2.  Demo", None)
text(s, 0.4, 1.9, 9.2, 1.5, "DEMO", size=88, color=NAVY, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.4, 3.4, 9.2, 0.4, "Same scenario (seed 100046), same opening positions, four different squads",
     size=14, color=GRAY, align=PP_ALIGN.CENTER)
cards = [("31B  ·  C0", "no channel", "Loss"), ("31B  ·  C2", "with messages", "Win"),
         ("4B  ·  untrained", "C2", "Loss"), ("4B  ·  after SFT", "AWR, C2", "Win")]
x = 0.4
for a, b, r in cards:
    rect(s, x, 4.3, 2.1, 1.1, WHITE, BORDER, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, 4.38, 2.1, 0.3, a, size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    text(s, x, 4.68, 2.1, 0.26, b, size=10, color=GRAY, align=PP_ALIGN.CENTER)
    text(s, x, 4.95, 2.1, 0.36, r, size=14, bold=True, color=(NAVY if r == "Win" else RED), align=PP_ALIGN.CENTER)
    x += 2.37

# ---------------- slide 3: setup, results, in progress ----------------
s = prs.slides.add_slide(BLANK)
frame(s, 3, "3.  Setup & Results", "Setup, Results & Experiments in Progress",
      "Learning from 31B lifts 4B from 17 % to 53 %; whether multi-agent RL adds more is being tested now.")
stats = [("64", "fixed scenarios, paired by seed (McNemar)"), ("Level 7", "SC2 built-in AI opponent"),
         ("60 %", "31B teacher with messages (C2)"), ("53.1 %", "best 4B: SFT on 31B games")]
x = 0.4
for big, small in stats:
    rect(s, x, 1.8, 2.1, 0.66, WHITE, BORDER, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, 1.83, 2.1, 0.32, big, size=15, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    text(s, x, 2.15, 2.1, 0.28, small, size=9, color=GRAY, align=PP_ALIGN.CENTER)
    x += 2.37

label(s, 0.4, 2.62, 4.5, "WIN RATE VS BUILT-IN AI  (%)")
cats = ["31B · C0 no channel", "31B · C1 + states", "31B · C2 + messages", "4B · untrained", "4B · no history",
        "4B · DAgger (SFT)", "4B · AWR (SFT)"]
vals = [25, 50, 60, 17.2, 3.1, 37.5, 53.1]
cd = CategoryChartData(); cd.categories = list(reversed(cats)); cd.add_series("Win rate", list(reversed(vals)))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(3.0), Inches(4.5), Inches(3.0), cd)
ch = gf.chart; ch.has_legend = False; ch.has_title = False
plot = ch.plots[0]; plot.gap_width = 45; plot.has_data_labels = True
dl = plot.data_labels; dl.number_format = '0.#'; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(10); dl.font.name = FONT; dl.font.color.rgb = rgb(INK)
ser = plot.series[0]
for i, c in enumerate(reversed(cats)):
    pt = ser.points[i]; pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(NAVY if c.startswith("31B") else RED)
va = ch.value_axis; va.maximum_scale = 70; va.minimum_scale = 0; va.has_major_gridlines = False
va.visible = False
ca = ch.category_axis; ca.tick_labels.font.size = Pt(10); ca.tick_labels.font.name = FONT
ca.tick_labels.font.color.rgb = rgb(INK); ca.format.line.color.rgb = rgb(BORDER)
text(s, 0.4, 6.05, 4.5, 0.6, "31B: 40 games per cell (C1 p = .037, C2 p = .003 vs C0).  4B: same 64 seeds; "
     "AWR p < 10⁻⁴, DAgger p = .011, no history p = .012 vs untrained.", size=9, color=GRAY, italic=True)

label(s, 5.15, 2.62, 4.45, "IN PROGRESS:  MULTI-AGENT RL FROM 53 % SFT")
cards = [
    ("MAPPO", "shared LoRA actor  ·  centralized critic on global state  ·  GAE  ·  PPO + KL to SFT",
     DATA["mappo_line"], DATA["mappo_64"]),
    ("GRPO self-play", "both teams share one policy  ·  group-normalized return + per-unit credit",
     DATA["grpo_line"], DATA["grpo_64"]),
]
y = 3.0
for h, how, line16, line64 in cards:
    rect(s, 5.15, y, 4.45, 1.32, WHITE, BORDER, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 5.25, y + 0.05, 4.25, 0.28, h, size=12, color=NAVY, bold=True)
    text(s, 5.25, y + 0.32, 4.25, 0.42, how, size=10, color=GRAY)
    text(s, 5.25, y + 0.74, 4.25, 0.55, [[("16-game checks:  ", {"bold": True, "color": NAVY}), (line16, {})],
                                         [("64-game test:  ", {"bold": True, "color": NAVY}), (line64, {})]],
         size=10, space_after=1)
    y += 1.42
rect(s, 5.15, 5.86, 4.45, 0.8, LIGHT, NAVY, MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 5.25, 5.9, 4.25, 0.74, [[("Verified:  ", {"bold": True, "color": NAVY}),
                                 ("sampled = recomputed log-probs, frozen SFT reference, per-unit credits sum to 0.", {})],
                                [("So far:  ", {"bold": True, "color": NAVY}), (DATA["so_far"], {})]],
     size=10, space_after=2)

prs.save(OUT)
print("saved", OUT)

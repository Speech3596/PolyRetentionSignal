"""
PolyRetentionSignal Design System / Theme Tokens
All color, spacing, and style constants are centralized here.
"""
from __future__ import annotations

# ── Main Colors ──────────────────────────────────────────────────────────────
BLUE1 = "#004F9F"
BLUE2 = "#263985"

# ── Sub Colors ───────────────────────────────────────────────────────────────
PURPLE1 = "#703F8A"
PURPLE2 = "#502968"
YELLOW1 = "#F7CF39"
YELLOW2 = "#E9A134"
SKYBLUE1 = "#69BDE4"
SKYBLUE2 = "#00A4D3"

# ── Neutral ──────────────────────────────────────────────────────────────────
WHITE = "#FFFFFF"
BG_LIGHT = "#F5F7FA"
CARD_BG = "#FFFFFF"
TEXT_DARK = "#1A1F36"
TEXT_MUTED = "#6B7280"
BORDER_LIGHT = "#E5E7EB"
SUCCESS = "#10B981"
DANGER = "#EF4444"

# ── Risk badge colors ────────────────────────────────────────────────────────
RISK_HIGH_BG = YELLOW2
RISK_HIGH_TEXT = WHITE
RISK_MEDIUM_BG = YELLOW1
RISK_MEDIUM_TEXT = TEXT_DARK
RISK_LOW_BG = SKYBLUE1
RISK_LOW_TEXT = WHITE

# ── Chart palette (ordered) ──────────────────────────────────────────────────
CHART_PALETTE = [BLUE1, SKYBLUE2, PURPLE1, YELLOW2, BLUE2, SKYBLUE1, PURPLE2, YELLOW1]
RETAINED_COLOR = BLUE1
CHURNED_COLOR = YELLOW2

# ── Plotly layout defaults ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(family="Pretendard, Noto Sans KR, sans-serif", color=TEXT_DARK),
    paper_bgcolor=WHITE,
    plot_bgcolor=BG_LIGHT,
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=CHART_PALETTE,
)


def inject_custom_css():
    """Return a Streamlit-compatible CSS string to inject via st.markdown."""
    return f"""
<style>
/* ── Global overrides ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] {{
    background-color: {BG_LIGHT};
}}
section[data-testid="stSidebar"] {{
    background-color: {BLUE2};
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] label {{
    color: {WHITE} !important;
}}

/* ── Header ───────────────────────────────────────────────── */
h1 {{
    color: {BLUE1} !important;
    font-weight: 700 !important;
}}
h2, h3 {{
    color: {BLUE2} !important;
}}

/* ── KPI card ─────────────────────────────────────────────── */
.kpi-card {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-top: 4px solid {BLUE1};
    text-align: center;
}}
.kpi-card .kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {BLUE1};
    margin: 4px 0;
}}
.kpi-card .kpi-label {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
}}
.kpi-card.purple {{
    border-top-color: {PURPLE1};
}}
.kpi-card.purple .kpi-value {{
    color: {PURPLE1};
}}
.kpi-card.sky {{
    border-top-color: {SKYBLUE2};
}}
.kpi-card.sky .kpi-value {{
    color: {SKYBLUE2};
}}
.kpi-card.yellow {{
    border-top-color: {YELLOW2};
}}
.kpi-card.yellow .kpi-value {{
    color: {YELLOW2};
}}

/* ── Section card ─────────────────────────────────────────── */
.section-card {{
    background: {CARD_BG};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}

/* ── Warning / info boxes ─────────────────────────────────── */
.warn-box {{
    background: #FEF3C7;
    border-left: 4px solid {YELLOW2};
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 0.9rem;
}}
.info-box {{
    background: #EFF6FF;
    border-left: 4px solid {BLUE1};
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 0.9rem;
}}
.error-box {{
    background: #FEE2E2;
    border-left: 4px solid {DANGER};
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 0.9rem;
}}

/* ── Risk badges ──────────────────────────────────────────── */
.badge-high {{
    background: {RISK_HIGH_BG};
    color: {RISK_HIGH_TEXT};
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 600;
}}
.badge-medium {{
    background: {RISK_MEDIUM_BG};
    color: {RISK_MEDIUM_TEXT};
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 600;
}}
.badge-low {{
    background: {RISK_LOW_BG};
    color: {RISK_LOW_TEXT};
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: 600;
}}

/* ── Styled table headers ─────────────────────────────────── */
.styled-table thead th {{
    background: {BLUE1} !important;
    color: {WHITE} !important;
    font-weight: 600;
}}

/* ── Tabs ─────────────────────────────────────────────────── */
button[data-baseweb="tab"] {{
    font-weight: 600 !important;
}}

/* ── Primary button ───────────────────────────────────────── */
.stButton > button[kind="primary"] {{
    background-color: {BLUE1} !important;
    border-color: {BLUE1} !important;
}}

/* ── Tooltip helper ───────────────────────────────────────── */
.help-tip {{
    display: inline-block;
    cursor: help;
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    margin-left: 4px;
}}
</style>
"""


def kpi_card_html(label: str, value, variant: str = "") -> str:
    cls = f"kpi-card {variant}".strip()
    return f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'


def section_header(title: str, help_text: str = "") -> str:
    tip = f' <span class="help-tip" title="{help_text}">&#9432;</span>' if help_text else ""
    return f'<h3 style="margin-top:12px;">{title}{tip}</h3>'

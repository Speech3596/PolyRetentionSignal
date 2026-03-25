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

/* ── Landing page ────────────────────────────────────────── */
.landing-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 70vh;
    padding: 40px 20px;
}}
.landing-logo {{
    max-width: 280px;
    margin-bottom: 16px;
}}
.landing-title {{
    font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: {BLUE1};
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}}
.landing-subtitle {{
    font-size: 1rem;
    color: {TEXT_MUTED};
    margin-bottom: 36px;
    letter-spacing: 1px;
}}
.upload-card {{
    background: {CARD_BG};
    border-radius: 14px;
    padding: 28px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid {BLUE1};
    min-height: 180px;
}}
.upload-card h4 {{
    color: {BLUE1};
    margin-bottom: 8px;
    font-weight: 700;
}}
.upload-card .upload-hint {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    margin-bottom: 12px;
}}

/* ── Loading overlay ─────────────────────────────────────── */
.loading-overlay {{
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(26, 31, 54, 0.82);
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
}}
.poly-anim-text {{
    font-family: 'Arial Black', 'Helvetica Neue', sans-serif;
    font-size: 100px;
    font-weight: 900;
    letter-spacing: -2px;
    background: linear-gradient(90deg,
        {BLUE1} 0%, {BLUE1} var(--fill),
        rgba(0,79,159,0.18) var(--fill), rgba(0,79,159,0.18) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: polyFillSweep 2.2s ease-in-out infinite;
}}
@keyframes polyFillSweep {{
    0%   {{ --fill: 0%;   }}
    50%  {{ --fill: 100%; }}
    100% {{ --fill: 0%;   }}
}}
@supports not (animation-timeline: view()) {{
    .poly-anim-text {{
        animation: polyFillFallback 2.2s ease-in-out infinite;
    }}
    @keyframes polyFillFallback {{
        0%   {{ background-position: 200% center; }}
        50%  {{ background-position: 0% center;   }}
        100% {{ background-position: 200% center; }}
    }}
}}
.poly-anim-text {{
    background-size: 200% 100%;
    animation: polyFillFallback 2.2s ease-in-out infinite;
}}
@keyframes polyFillFallback {{
    0%   {{ background-position: 200% center; }}
    50%  {{ background-position: 0% center;   }}
    100% {{ background-position: 200% center; }}
}}
.loading-msg {{
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    margin-top: 18px;
    letter-spacing: 2px;
}}

/* ── File accordion ──────────────────────────────────────── */
.file-accordion {{
    background: {CARD_BG};
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    cursor: pointer;
    user-select: none;
    border-left: 4px solid {BLUE1};
}}
.file-accordion summary {{
    font-weight: 600;
    color: {BLUE1};
    font-size: 0.92rem;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.file-accordion summary::before {{
    content: "\\25B6";
    font-size: 0.7rem;
    transition: transform 0.2s;
}}
.file-accordion[open] summary::before {{
    transform: rotate(90deg);
}}
.file-accordion .file-list {{
    margin-top: 8px;
    padding-left: 18px;
    font-size: 0.84rem;
    color: {TEXT_MUTED};
    line-height: 1.7;
}}

/* ── Condition badge ─────────────────────────────────────── */
.condition-bar {{
    background: linear-gradient(135deg, {BLUE2}, {BLUE1});
    color: {WHITE};
    padding: 10px 18px;
    border-radius: 8px;
    font-size: 0.85rem;
    margin-bottom: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
}}
.condition-bar span {{
    background: rgba(255,255,255,0.18);
    padding: 2px 10px;
    border-radius: 4px;
    font-weight: 600;
}}

/* ── Hide sidebar when not needed ────────────────────────── */
.no-sidebar section[data-testid="stSidebar"] {{
    display: none !important;
}}
.no-sidebar .stMainBlockContainer {{
    max-width: 1100px;
    margin: 0 auto;
}}

/* ── Filter container ────────────────────────────────────── */
.filter-container {{
    background: {CARD_BG};
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    padding: 20px 28px;
    border-top: 3px solid {BLUE1};
    margin-bottom: 16px;
}}
.filter-header {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}}
.filter-label {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {TEXT_DARK};
    margin-bottom: 2px;
}}
.filter-summary {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    text-align: center;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid {BORDER_LIGHT};
}}
.filter-btn {{
    font-size: 0.72rem;
    color: {BLUE1};
    cursor: pointer;
    text-decoration: none;
    font-weight: 600;
    border: none;
    background: none;
    padding: 0;
}}
.filter-btn:hover {{
    text-decoration: underline;
}}

/* ── Multiselect tag (chip) override ─────────────────────── */
[data-baseweb="tag"] {{
    background-color: {BLUE1} !important;
    border-radius: 20px !important;
    padding: 2px 12px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: white !important;
    border: none !important;
}}
[data-baseweb="tag"] span {{
    color: white !important;
}}
[data-baseweb="tag"] [role="presentation"] {{
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.9rem !important;
}}

/* ── Dropdown popover ────────────────────────────────────── */
[data-baseweb="popover"] {{
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
    border: 1px solid {BORDER_LIGHT} !important;
    overflow: hidden !important;
}}
[data-baseweb="menu"] li {{
    font-size: 0.88rem !important;
    padding: 10px 16px !important;
    transition: background 0.15s !important;
}}
[data-baseweb="menu"] li:hover {{
    background-color: #EFF6FF !important;
    color: {BLUE1} !important;
}}

/* ── Select input box ────────────────────────────────────── */
[data-baseweb="select"] > div:first-child {{
    border-radius: 10px !important;
    border: 1.5px solid {BORDER_LIGHT} !important;
    background: white !important;
    min-height: 44px !important;
    transition: border-color 0.2s !important;
}}
[data-baseweb="select"] > div:first-child:hover {{
    border-color: {BLUE1} !important;
}}

/* ── Filter checkbox (전체 toggle) ─────────────────────── */
.filter-container [data-testid="stCheckbox"] {{
    margin-top: 0px !important;
}}
.filter-container [data-testid="stCheckbox"] label {{
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: {BLUE1} !important;
    gap: 4px !important;
}}
.filter-container [data-testid="stCheckbox"] label span[data-testid="stCheckbox-label"] {{
    color: {BLUE1} !important;
}}
.filter-container [data-testid="stCheckbox"] div[role="checkbox"] {{
    border-color: {BLUE1} !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 4px !important;
}}
.filter-container [data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {{
    background-color: {BLUE1} !important;
    border-color: {BLUE1} !important;
}}

/* ── Multiselect clear icon ───────────────────────────── */
[data-baseweb="select"] svg {{
    color: {TEXT_MUTED} !important;
}}
[data-baseweb="select"] svg:hover {{
    color: {BLUE1} !important;
}}

/* ── Analysis sub-tabs styling ────────────────────────── */
button[data-baseweb="tab"] {{
    font-size: 0.82rem !important;
    padding: 8px 12px !important;
}}
</style>
"""


def kpi_card_html(label: str, value, variant: str = "") -> str:
    cls = f"kpi-card {variant}".strip()
    return f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'


def section_header(title: str, help_text: str = "") -> str:
    tip = f' <span class="help-tip" title="{help_text}">&#9432;</span>' if help_text else ""
    return f'<h3 style="margin-top:12px;">{title}{tip}</h3>'

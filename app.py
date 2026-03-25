from __future__ import annotations

import base64, traceback, time, pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from retentionsignal_core import (
    apply_tpi_formula,
    build_item_stats,
    build_student_summary,
    build_tpi_matrix,
    detect_file_kind,
    make_default_formula,
    read_single_exam,
    read_student_info,
    to_csv_bytes,
    ALIAS_TO_COLUMN,
)
from analysis_core import (
    TENURE_LABELS_INTERNAL,
    assign_tenure_bucket,
    extract_cohort,
    build_cohort_student_level,
    build_comparison_table,
    point_biserial_table,
    quantile_tenure_crosstab,
    univariate_logistic_table,
    run_all_multivariate_models,
    correlation_matrix,
    correlation_pairs_table,
    run_survival_analysis,
    compute_risk_scores,
    build_monthly_flow,
    campus_summary,
    build_integrated_summary,
    to_csv_bytes as ac_to_csv_bytes,
    build_zip_package,
)
from theme import (
    inject_custom_css,
    kpi_card_html,
    section_header,
    PLOTLY_LAYOUT,
    BLUE1, BLUE2, PURPLE1, PURPLE2, YELLOW1, YELLOW2, SKYBLUE1, SKYBLUE2,
    RETAINED_COLOR, CHURNED_COLOR, CHART_PALETTE,
    TEXT_MUTED, WHITE,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Poly Retention Signal", layout="wide", initial_sidebar_state="collapsed")
st.markdown(inject_custom_css(), unsafe_allow_html=True)
# Hide sidebar completely
st.markdown("<style>section[data-testid='stSidebar']{display:none !important;}</style>", unsafe_allow_html=True)

# ── Poly logo helper ─────────────────────────────────────────────────────────
_LOGO_PNG = pathlib.Path(__file__).parent / "bi_poly.png"
_LOGO_SVG = pathlib.Path(__file__).parent / "poly_logo.svg"

def _logo_html(width: int = 260) -> str:
    for path, mime in [(_LOGO_PNG, "image/png"), (_LOGO_SVG, "image/svg+xml")]:
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            return f'<img src="data:{mime};base64,{b64}" width="{width}" style="display:block;margin:0 auto;">'
    return f'<div style="text-align:center;font-family:Arial Black,sans-serif;font-size:72px;font-weight:900;color:{BLUE1};">Poly</div>'

LOADING_OVERLAY_HTML = f"""
<div class="loading-overlay">
  <div class="poly-anim-text">Poly</div>
  <div class="loading-msg">데이터 분석 중 ...</div>
</div>
"""

# ── Session state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "data_loaded": False,
    "processing": False,
    "raw_df": None,
    "item_df": None,
    "summary_df": None,
    "student_info_df": None,
    "exam_file_names": [],
    "student_file_name": "",
    "tpi_matrix": None,
    "analysis_results": None,
    "formula_used": None,
    "enabled_weights_used": None,
    "include_na_students": True,
    "stat_exam_types": None,
    "stat_months": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_error(msg: str):
    st.markdown(f'<div class="error-box">{msg}</div>', unsafe_allow_html=True)

def safe_warn(msg: str):
    st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)

def safe_info(msg: str):
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

def validate_uploads(exam_objs, student_obj):
    errors: list[str] = []
    valid_exams = []
    for f in exam_objs or []:
        name = getattr(f, "name", str(f))
        kind = detect_file_kind(name, expected_ext=".xlsx")
        if kind != "exam":
            errors.append(f"시험 업로드 칸에 시험 파일이 아닌 항목: {name}")
        else:
            valid_exams.append(f)
    if student_obj is not None:
        name = getattr(student_obj, "name", str(student_obj))
        kind = detect_file_kind(name, expected_ext=".csv")
        if kind != "student":
            errors.append(f"학생 업로드 칸에 학생 파일이 아닌 항목: {name}")
    return valid_exams, student_obj, errors


@st.cache_data(show_spinner=False)
def load_all(exam_objs, student_obj):
    exam_frames = [read_single_exam(f) for f in exam_objs]
    raw = pd.concat(exam_frames, ignore_index=True) if exam_frames else pd.DataFrame()
    student_info = read_student_info(student_obj) if student_obj is not None else pd.DataFrame()
    item = build_item_stats(raw) if not raw.empty else pd.DataFrame()
    summary = build_student_summary(raw, student_info) if not raw.empty else pd.DataFrame()
    return raw, item, summary, student_info


def _condition_badge(formula: str, exam_types: list, months: list, campuses: list | None = None, include_na: bool = True):
    """Show current TPI conditions as a styled bar."""
    parts = [
        f"<span>TPI: {formula[:60]}{'…' if len(formula)>60 else ''}</span>",
        f"<span>시험유형: {', '.join(exam_types)}</span>",
        f"<span>월: {', '.join(str(m) for m in months)}</span>",
    ]
    if campuses:
        parts.append(f"<span>캠퍼스: {', '.join(campuses[:5])}</span>")
    parts.append(f"<span>N/A 학생: {'포함' if include_na else '제외'}</span>")
    st.markdown(f'<div class="condition-bar">{"".join(parts)}</div>', unsafe_allow_html=True)


def _tenure_breakdown(df: pd.DataFrame, metric_col: str, value_func="mean"):
    """Return {전체, 6개월 이하, ..., 24개월 초과} → value."""
    labels = ["전체"] + list(TENURE_LABELS_INTERNAL)
    out = {}
    for lb in labels:
        sub = df if lb == "전체" else df[df["재원기간구간"] == lb]
        vals = sub[metric_col].dropna() if metric_col in sub.columns else pd.Series(dtype=float)
        if value_func == "mean":
            out[lb] = round(vals.mean(), 2) if len(vals) > 0 else "N/A"
        elif value_func == "count":
            out[lb] = len(vals)
        elif value_func == "churn_rate":
            n = len(sub)
            c = int(sub["target_churn_sep"].sum()) if n > 0 and "target_churn_sep" in sub.columns else 0
            out[lb] = round(c / n * 100, 1) if n > 0 else "N/A"
    return out


def _build_tenure_summary_table(sl: pd.DataFrame, metrics: list[str], suffix: str = "_8월"):
    """Build a table: rows=metrics, cols=tenure groups, values=mean."""
    rows = []
    for m in metrics:
        col = f"{m}{suffix}"
        if col not in sl.columns:
            continue
        row = {"지표": m}
        row.update(_tenure_breakdown(sl, col))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_analysis(summary_df, student_info_df, formula, selected_exam_types, selected_months, include_na):
    """Run TPI + cohort + all statistical analyses. Returns dict of results."""
    # Filter summary by selected conditions
    view = summary_df[
        summary_df["시험유형"].isin(selected_exam_types) & summary_df["월"].isin(selected_months)
    ].copy()

    if view.empty:
        return {"error": "선택된 조건에 해당하는 데이터가 없습니다."}

    try:
        full_tpi_df = apply_tpi_formula(view, formula)
    except Exception as e:
        return {"error": f"TPI 수식 오류: {e}"}

    # Also compute TPI on full data for cohort extraction (needs March+Aug)
    full_summary_tpi = apply_tpi_formula(summary_df, formula)

    cohort_df, cohort_meta = extract_cohort(full_summary_tpi, student_info_df)
    if cohort_df.empty:
        return {"error": cohort_meta.get("error", "코호트 추출 실패")}

    student_level = build_cohort_student_level(cohort_df, student_info_df)
    if student_level.empty:
        return {"error": "학생 수준 데이터 생성 실패"}

    # N/A handling: if exclude, drop students with any NaN in key 8월 metrics
    if not include_na:
        key_cols = [c for c in student_level.columns if c.endswith("_8월") or c.endswith("_3월")]
        before = len(student_level)
        student_level = student_level.dropna(subset=key_cols).copy()
        excluded = before - len(student_level)
        cohort_meta["na_excluded"] = excluded
        # Recompute meta counts
        cohort_meta["cohort_size"] = len(student_level)
        cohort_meta["retained_count"] = int((student_level["target_churn_sep"] == 0).sum())
        cohort_meta["churned_count"] = int((student_level["target_churn_sep"] == 1).sum())
        cohort_meta["churn_rate"] = float(student_level["target_churn_sep"].mean() * 100) if len(student_level) > 0 else 0

    base_metrics = ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR", "TPI"]
    aug_metrics = [f"{m}_8월" for m in base_metrics]
    mar_metrics = [f"{m}_3월" for m in base_metrics]
    delta_metrics = [f"{m}_변화량" for m in base_metrics]

    comp_aug = build_comparison_table(student_level, base_metrics, "_8월")
    comp_mar = build_comparison_table(student_level, base_metrics, "_3월")
    comp_delta = build_comparison_table(student_level, base_metrics, "_변화량")
    pb_aug = point_biserial_table(student_level, aug_metrics)
    pb_mar = point_biserial_table(student_level, mar_metrics)
    pb_delta = point_biserial_table(student_level, delta_metrics)

    crosstabs = {}
    for m in base_metrics:
        col = f"{m}_8월"
        if col in student_level.columns:
            ct = quantile_tenure_crosstab(student_level, col)
            if not ct.empty:
                crosstabs[m] = ct

    uni_aug = univariate_logistic_table(student_level, aug_metrics)
    uni_mar = univariate_logistic_table(student_level, mar_metrics)
    uni_delta = univariate_logistic_table(student_level, delta_metrics)
    multi_models = run_all_multivariate_models(student_level)
    corr_m = correlation_matrix(student_level, aug_metrics)
    corr_p = correlation_pairs_table(student_level, aug_metrics)
    pb_all = point_biserial_table(student_level, aug_metrics + mar_metrics + delta_metrics)
    survival = run_survival_analysis(student_level)
    risk_df = compute_risk_scores(student_level)

    monthly_flows = {}
    for m in base_metrics:
        mf = build_monthly_flow(cohort_df, m)
        if not mf.empty:
            monthly_flows[m] = mf

    campus_df = campus_summary(risk_df)
    integrated = build_integrated_summary(pb_aug, uni_aug, multi_models, comp_aug)

    # TPI matrix (for display in TPI tab)
    tpi_df_view = apply_tpi_formula(
        summary_df[summary_df["시험유형"].isin(selected_exam_types) & summary_df["월"].isin(selected_months)].copy(),
        formula,
    )
    tpi_matrix = build_tpi_matrix(tpi_df_view)

    return {
        "cohort_meta": cohort_meta,
        "cohort_df": cohort_df,
        "student_level": student_level,
        "comp_aug": comp_aug, "comp_mar": comp_mar, "comp_delta": comp_delta,
        "pb_aug": pb_aug, "pb_mar": pb_mar, "pb_delta": pb_delta, "pb_all": pb_all,
        "crosstabs": crosstabs,
        "uni_aug": uni_aug, "uni_mar": uni_mar, "uni_delta": uni_delta,
        "multi_models": multi_models,
        "corr_matrix": corr_m, "corr_pairs": corr_p,
        "survival": survival,
        "risk_df": risk_df,
        "monthly_flows": monthly_flows,
        "campus_df": campus_df,
        "integrated": integrated,
        "tpi_matrix": tpi_matrix,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.data_loaded:
    # ── Show loading overlay if processing ──
    if st.session_state.processing:
        st.markdown(LOADING_OVERLAY_HTML, unsafe_allow_html=True)

        try:
            raw_df, item_df, summary_df, student_info_df = load_all(
                st.session_state._pending_exams, st.session_state._pending_student
            )
            if summary_df.empty:
                safe_error("처리 가능한 데이터가 없습니다. 파일을 확인해 주세요.")
                st.session_state.processing = False
                st.stop()

            st.session_state.raw_df = raw_df
            st.session_state.item_df = item_df
            st.session_state.summary_df = summary_df
            st.session_state.student_info_df = student_info_df
            st.session_state.data_loaded = True
            st.session_state.processing = False
            del st.session_state._pending_exams
            del st.session_state._pending_student
            time.sleep(1.0)
            st.rerun()
        except Exception:
            st.session_state.processing = False
            safe_error("데이터 로딩 중 오류가 발생했습니다. 파일 형식을 확인해 주세요.")
            st.stop()

    # ── Landing page ──
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    st.markdown(_logo_html(260), unsafe_allow_html=True)
    st.markdown('<div class="landing-title" style="text-align:center;">Poly Retention Signal</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle" style="text-align:center;">* 퇴원생 성적 기반 코호트 분석 *</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_l, col_gap, col_r = st.columns([5, 1, 5])

    with col_l:
        st.markdown('<div class="upload-card"><h4>성적 데이터 업로드</h4><p class="upload-hint">시험 성적 .xlsx 여러 개 업로드 가능 / 학생 데이터 .csv 1개만 가능</p></div>', unsafe_allow_html=True)
        exam_files = st.file_uploader(
            "시험 데이터 (.xlsx)",
            type=["xlsx"],
            accept_multiple_files=True,
            key="upload_exam",
            label_visibility="collapsed",
        )

    with col_r:
        st.markdown('<div class="upload-card"><h4>학생 데이터 업로드</h4><p class="upload-hint">학생 마스터 .csv 파일 1개</p></div>', unsafe_allow_html=True)
        student_file = st.file_uploader(
            "학생 데이터 (.csv)",
            type=["csv"],
            accept_multiple_files=False,
            key="upload_student",
            label_visibility="collapsed",
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # 분석 button centered
    _, btn_col, _ = st.columns([4, 3, 4])
    with btn_col:
        analyze_clicked = st.button("분 석", type="primary", use_container_width=True)

    if analyze_clicked:
        if not exam_files:
            safe_error("시험 데이터를 업로드해 주세요.")
            st.stop()
        if student_file is None:
            safe_error("학생 데이터를 업로드해 주세요.")
            st.stop()

        exam_files, student_file, validation_errors = validate_uploads(exam_files, student_file)
        if validation_errors:
            for msg in validation_errors:
                safe_error(msg)
            st.stop()

        # Store file names and pending uploads
        st.session_state.exam_file_names = [f.name for f in exam_files]
        st.session_state.student_file_name = student_file.name
        st.session_state._pending_exams = exam_files
        st.session_state._pending_student = student_file
        st.session_state.processing = True
        st.rerun()

    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION (data loaded)
# ═════════════════════════════════════════════════════════════════════════════

raw_df = st.session_state.raw_df
item_df = st.session_state.item_df
summary_df = st.session_state.summary_df
student_info_df = st.session_state.student_info_df

# ── File accordion (top) ────────────────────────────────────────────────────
exam_names = st.session_state.exam_file_names
student_name = st.session_state.student_file_name
file_count = len(exam_names) + (1 if student_name else 0)

accordion_html = f"""
<details class="file-accordion">
  <summary>업로드된 파일 ({file_count}개)</summary>
  <div class="file-list">
"""
for fn in exam_names:
    accordion_html += f"📊 {fn}<br>"
if student_name:
    accordion_html += f"👤 {student_name}<br>"
accordion_html += """
  </div>
</details>
"""
col_acc, col_reset = st.columns([9, 1])
with col_acc:
    st.markdown(accordion_html, unsafe_allow_html=True)
with col_reset:
    if st.button("새 분석", key="btn_reset"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Filter controls (dropdowns) ─────────────────────────────────────────────
with st.container():
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])
    with fc1:
        all_exam_types = sorted(summary_df["시험유형"].dropna().unique().tolist())
        selected_exam_types = st.multiselect(
            "시험유형", options=all_exam_types, default=all_exam_types, key="filter_exam_type"
        )
    with fc2:
        month_opts = sorted(
            summary_df.loc[summary_df["시험유형"].isin(selected_exam_types), "월"]
            .dropna().astype(int).unique().tolist()
        )
        selected_months = st.multiselect("월", options=month_opts, default=month_opts, key="filter_month")
    with fc3:
        if "캠퍼스" in summary_df.columns:
            campus_opts = sorted(summary_df["캠퍼스"].dropna().unique().tolist())
            selected_campuses = st.multiselect("캠퍼스", options=campus_opts, default=campus_opts, key="filter_campus")
        else:
            selected_campuses = []
    with fc4:
        selected_students = st.multiselect(
            "학생코드 (선택)",
            options=summary_df["학생코드"].dropna().astype(str).unique().tolist(),
            default=[], key="filter_student",
        )

# Apply filters
view_df = summary_df[
    summary_df["시험유형"].isin(selected_exam_types) & summary_df["월"].isin(selected_months)
].copy()
if selected_campuses and "캠퍼스" in view_df.columns:
    view_df = view_df[view_df["캠퍼스"].isin(selected_campuses)].copy()
if selected_students:
    view_df = view_df[view_df["학생코드"].astype(str).isin(selected_students)].copy()

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["학생 성적표", "문항 정답률", "TPI 계산", "통계 분석"])

# ═══ TAB 1: 학생 성적표 ═══
with tab1:
    st.subheader("학생 성적표")
    st.dataframe(view_df, use_container_width=True, height=600)
    st.download_button(
        "학생 성적표 CSV 다운로드",
        data=view_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="Poly_Student_Summary.csv",
        mime="text/csv",
        key="btn_dl_student",
    )

# ═══ TAB 2: 문항 정답률 ═══
with tab2:
    st.subheader("문항 정답률")
    item_view = item_df[
        item_df["exam_type"].isin(selected_exam_types) & item_df["month_num"].isin(selected_months)
    ].copy()
    item_view = item_view.rename(columns={"exam_type": "시험유형", "year": "연도", "month_num": "월"})
    st.dataframe(item_view, use_container_width=True, height=600)
    st.download_button(
        "문항 정답률 CSV 다운로드",
        data=to_csv_bytes(item_view),
        file_name="PolyRetentionSignal_item_accuracy.csv",
        mime="text/csv",
        key="btn_dl_item",
    )

# ═══ TAB 3: TPI 계산 ═══
with tab3:
    st.subheader("TPI 계산")
    st.markdown("사용 가능한 별칭: `P`, `T`, `BCV`, `CI`, `QR`, `CT`, `CCV`, `CQR`, `CV`  ·  연산자: `+`, `-`, `*`, `/`, `**`, `( )`")

    formula_mode = st.radio(
        "수식 입력 방식", ["가중치(비율) 기반 자동 생성", "자유 수식 직접 입력"], horizontal=True, key="formula_mode"
    )

    if formula_mode == "가중치(비율) 기반 자동 생성":
        default_weights = {"P": 20, "T": 20, "BCV": 15, "CI": 15, "QR": 10, "CT": 10, "CCV": 5, "CQR": 5, "CV": 0}
        enabled_weights = {}
        wcols = st.columns(5)
        aliases = list(default_weights.keys())
        for i, alias in enumerate(aliases):
            with wcols[i % 5]:
                default_on = alias != "CV"
                enabled = st.checkbox(f"{alias} 채택", value=default_on, key=f"en_{alias}")
                weight = st.number_input(f"{alias} 비율%", min_value=0.0, max_value=100.0, value=float(default_weights[alias]), step=1.0, key=f"wt_{alias}")
                enabled_weights[alias] = weight if enabled else 0.0
        formula = make_default_formula(enabled_weights)
        st.text_input("생성된 TPI 수식", value=formula, disabled=True, key="formula_display")
    else:
        formula = st.text_input(
            "TPI 수식",
            value="(P*20.0 + T*20.0 + BCV*15.0 + CI*15.0 + QR*10.0 + CT*10.0 + CCV*5.0 + CQR*5.0) / 100.0",
            key="formula_free",
        )

    # TPI 실행 row
    tpi_col1, tpi_col2, tpi_col3 = st.columns([2, 2, 6])
    with tpi_col1:
        tpi_run = st.button("TPI 실행", type="primary", key="btn_tpi_run")
    with tpi_col2:
        include_na = st.checkbox("N/A 학생 포함", value=st.session_state.include_na_students, key="chk_include_na",
                                  help="일부 월 시험 데이터 미존재 학생을 통계 분석에 포함할지 여부")

    if tpi_run:
        st.session_state.include_na_students = include_na
        with st.spinner("TPI 계산 및 코호트 분석 수행 중..."):
            results = run_analysis(summary_df, student_info_df, formula, selected_exam_types, selected_months, include_na)

        if isinstance(results, dict) and "error" in results and isinstance(results["error"], str):
            safe_error(results["error"])
        else:
            st.session_state.analysis_results = results
            st.session_state.tpi_matrix = results.get("tpi_matrix")
            st.session_state.formula_used = formula
            st.session_state.enabled_weights_used = enabled_weights.copy() if formula_mode == "가중치(비율) 기반 자동 생성" else None
            st.session_state.stat_exam_types = selected_exam_types[:]
            st.session_state.stat_months = selected_months[:]
            st.success("TPI 계산 및 코호트 분석 완료")

    # Show TPI matrix
    if st.session_state.tpi_matrix is not None:
        st.dataframe(st.session_state.tpi_matrix, use_container_width=True, height=650)
        st.download_button(
            "TPI 매트릭스 CSV 다운로드",
            data=to_csv_bytes(st.session_state.tpi_matrix),
            file_name="PolyRetentionSignal_TPI_matrix.csv",
            mime="text/csv",
            key="btn_dl_tpi",
        )

    with st.expander("TPI 지표 별칭(Alias) 가이드"):
        st.markdown("""
- **P**: P-Score (전체 정답률)  · **T**: T-Score (표준점수 기반 0-100)
- **BCV**: B.CV (과목별 편차 역수)  · **CI**: CI (난이도 가중 성취 지표)
- **QR**: QR (백분위 순위)  · **CT**: C.T-Score (캠퍼스 내 T-Score)
- **CCV**: C.CV (캠퍼스 내 변동성)  · **CQR**: C.QR (캠퍼스 내 백분위)  · **CV**: CV (전체 시험 변동성)
""")

# ═══ TAB 4: 통계 분석 ═══
with tab4:
    ar = st.session_state.analysis_results

    if ar is None:
        st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;padding:60px;color:{TEXT_MUTED};font-size:1.15rem;">'
            'TPI 설정 및 실행을 진행해야 합니다</div>',
            unsafe_allow_html=True,
        )
    elif isinstance(ar, dict) and "error" in ar and isinstance(ar["error"], str):
        safe_error(ar["error"])
    else:
        # ── Analysis condition controls inside statistics tab ──
        st.markdown(section_header("분석 조건 설정"), unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns([2, 2, 2])
        with sc1:
            stat_exam_opts = sorted(summary_df["시험유형"].dropna().unique().tolist())
            stat_exam_sel = st.multiselect(
                "시험유형 (통계)", options=stat_exam_opts,
                default=st.session_state.stat_exam_types or stat_exam_opts,
                key="stat_exam_filter",
            )
        with sc2:
            stat_month_opts = sorted(summary_df["월"].dropna().astype(int).unique().tolist())
            stat_month_sel = st.multiselect(
                "월 (통계)", options=stat_month_opts,
                default=st.session_state.stat_months or stat_month_opts,
                key="stat_month_filter",
            )
        with sc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            reanalyze = st.button("분석", type="primary", key="btn_reanalyze")

        if reanalyze:
            with st.spinner("재분석 중..."):
                new_results = run_analysis(
                    summary_df, student_info_df,
                    st.session_state.formula_used,
                    stat_exam_sel, stat_month_sel,
                    st.session_state.include_na_students,
                )
            if isinstance(new_results, dict) and "error" in new_results:
                safe_error(new_results["error"])
            else:
                st.session_state.analysis_results = new_results
                st.session_state.tpi_matrix = new_results.get("tpi_matrix")
                st.session_state.stat_exam_types = stat_exam_sel
                st.session_state.stat_months = stat_month_sel
                ar = new_results
                st.success("재분석 완료")

        st.divider()

        # ── Banner ──
        st.markdown(
            '<div class="info-box"><strong>2025년 3월 + 8월 재원 학생 코호트 기준, 9월 재원/퇴원 비교 분석</strong><br>'
            '본 분석은 인과 증명이 아니라, 9월 퇴원과 함께 관찰되는 통계적 방향성 탐색을 목적으로 합니다.</div>',
            unsafe_allow_html=True,
        )

        meta = ar["cohort_meta"]
        sl = ar["student_level"]
        cohort_df = ar["cohort_df"]

        ANALYSIS_TABS = [
            "1. TPI 지수 구성",
            "2. 종합 요약",
            "3. 재원 vs 퇴원 비교",
            "4. 지표별 퇴원 연결",
            "5. TPI 심층 분석",
            "6. 월별 흐름",
            "7. 위험 신호 탐지",
            "8. 캠퍼스/시험유형",
            "9. 상관분석",
            "10. 회귀분석",
            "11. 생존분석",
            "12. 통합 통계",
            "13. 다운로드",
        ]
        atabs = st.tabs(ANALYSIS_TABS)

        _formula = st.session_state.formula_used or ""
        _s_etypes = st.session_state.stat_exam_types or []
        _s_months = st.session_state.stat_months or []
        _inc_na = st.session_state.include_na_students

        # ─── TAB 1: TPI 지수 구성 ────────────────────────────────────────
        with atabs[0]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("TPI 지수 구성 정의")
            st.markdown(f"**현재 적용된 TPI 수식:** `{_formula}`")

            ew = st.session_state.enabled_weights_used
            direction_map = {
                "P": "높을수록 좋음", "T": "높을수록 좋음", "BCV": "높을수록 좋음 (역방향 보정됨)",
                "CI": "높을수록 좋음", "QR": "높을수록 좋음", "CT": "높을수록 좋음",
                "CCV": "높을수록 좋음 (역방향 보정됨)", "CQR": "높을수록 좋음", "CV": "높을수록 좋음 (역방향 보정됨)",
            }
            norm_map = {
                "P": "0~100 clip", "T": "Z-score→T-score 0~100", "BCV": "100 - raw_CV, 0~100 clip",
                "CI": "난이도 가중 0~100", "QR": "백분위 0~100", "CT": "캠퍼스 내 T-score 0~100",
                "CCV": "캠퍼스 내 100-rawCV", "CQR": "캠퍼스 내 백분위 0~100", "CV": "100 - raw_CV 0~100",
            }
            config_rows = []
            all_aliases = ["P", "T", "BCV", "CI", "QR", "CT", "CCV", "CQR", "CV"]
            for alias in all_aliases:
                if ew is not None:
                    included = ew.get(alias, 0) > 0
                    weight = ew.get(alias, 0)
                else:
                    included = alias in (_formula or "")
                    weight = "-"
                config_rows.append({
                    "지표": alias, "실제컬럼": ALIAS_TO_COLUMN.get(alias, alias),
                    "포함여부": "포함" if included else "제외", "가중치": weight,
                    "방향성": direction_map.get(alias, ""), "정규화방식": norm_map.get(alias, ""),
                })
            config_df = pd.DataFrame(config_rows)
            st.dataframe(config_df, use_container_width=True, hide_index=True)

            if ew:
                total_w = sum(v for v in ew.values() if v > 0)
                st.markdown(f"**가중치 합계:** {total_w}")

            st.markdown("---")
            st.markdown("**분석 코호트 정의:**")
            st.markdown(f"- 코호트 규모: {meta['cohort_size']}명 (유지: {meta['retained_count']}명, 퇴원: {meta['churned_count']}명)")
            if meta.get("na_excluded"):
                st.markdown(f"- N/A 제외 학생: {meta['na_excluded']}명")

            st.markdown("**재원기간 구간 정의:**")
            for label in TENURE_LABELS_INTERNAL:
                st.markdown(f"- {label}")

        # ─── TAB 2: 종합 요약 ────────────────────────────────────────────
        with atabs[1]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("종합 요약")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi_card_html("코호트 학생 수", meta["cohort_size"]), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi_card_html("9월 재원", meta["retained_count"], "sky"), unsafe_allow_html=True)
            with k3:
                st.markdown(kpi_card_html("9월 퇴원", meta["churned_count"], "yellow"), unsafe_allow_html=True)
            with k4:
                st.markdown(kpi_card_html("퇴원율", f"{meta['churn_rate']:.1f}%", "purple"), unsafe_allow_html=True)

            st.markdown("")
            k5, k6, k7 = st.columns(3)
            avg_tpi = sl["TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
            churned_tpi = sl.loc[sl["target_churn_sep"] == 1, "TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
            retained_tpi = sl.loc[sl["target_churn_sep"] == 0, "TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
            with k5:
                st.markdown(kpi_card_html("평균 TPI (8월)", f"{avg_tpi:.1f}" if pd.notna(avg_tpi) else "N/A"), unsafe_allow_html=True)
            with k6:
                st.markdown(kpi_card_html("퇴원군 평균 TPI", f"{churned_tpi:.1f}" if pd.notna(churned_tpi) else "N/A", "yellow"), unsafe_allow_html=True)
            with k7:
                st.markdown(kpi_card_html("유지군 평균 TPI", f"{retained_tpi:.1f}" if pd.notna(retained_tpi) else "N/A", "sky"), unsafe_allow_html=True)

            # Tenure breakdown summary
            st.markdown("---")
            st.markdown("**재원기간 구간별 요약**")
            tenure_rows = []
            for label in ["전체"] + list(TENURE_LABELS_INTERNAL):
                sub = sl if label == "전체" else sl[sl["재원기간구간"] == label]
                n = len(sub)
                churned = int(sub["target_churn_sep"].sum()) if n > 0 else 0
                rate = round(churned / n * 100, 1) if n > 0 else 0
                avg = sub["TPI_8월"].mean() if "TPI_8월" in sub.columns and n > 0 else np.nan
                tenure_rows.append({"구간": label, "학생수": n, "퇴원수": churned, "퇴원율(%)": rate, "평균TPI": round(avg, 2) if pd.notna(avg) else "N/A"})
            st.dataframe(pd.DataFrame(tenure_rows), use_container_width=True, hide_index=True)

            # Charts
            dc1, dc2 = st.columns(2)
            with dc1:
                fig = go.Figure(data=[go.Pie(
                    labels=["재원", "퇴원"], values=[meta["retained_count"], meta["churned_count"]],
                    hole=0.5, marker_colors=[RETAINED_COLOR, CHURNED_COLOR],
                )])
                fig.update_layout(title="9월 재원/퇴원 비율", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with dc2:
                if not ar["comp_aug"].empty:
                    comp = ar["comp_aug"]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="유지군", x=comp["지표"], y=comp["유지군_평균"], marker_color=RETAINED_COLOR))
                    fig.add_trace(go.Bar(name="퇴원군", x=comp["지표"], y=comp["퇴원군_평균"], marker_color=CHURNED_COLOR))
                    fig.update_layout(barmode="group", title="유지군 vs 퇴원군 (8월)", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

        # ─── TAB 3: 재원 vs 퇴원 비교 ────────────────────────────────────
        with atabs[2]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("재원 vs 9월 퇴원 비교")
            st.caption(f"코호트 n={meta['cohort_size']} | 유지군 n={meta['retained_count']} | 퇴원군 n={meta['churned_count']}")

            for label, comp_df in [("8월 값 비교", ar["comp_aug"]), ("3월 값 비교", ar["comp_mar"]), ("3→8 변화량 비교", ar["comp_delta"])]:
                st.markdown(f"**{label}**")
                if comp_df is not None and not comp_df.empty:
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                else:
                    safe_warn(f"{label} 데이터가 없습니다.")

            # Tenure breakdown table (8월)
            st.markdown("---")
            st.markdown("**재원기간 구간별 8월 지표 평균**")
            tb = _build_tenure_summary_table(sl, ["P-Score", "T-Score", "B.CV", "CI", "QR", "TPI"], "_8월")
            if not tb.empty:
                st.dataframe(tb, use_container_width=True, hide_index=True)

            # Box plot
            if "TPI_8월" in sl.columns:
                sl_plot = sl.copy()
                sl_plot["그룹"] = sl_plot["target_churn_sep"].map({0: "유지", 1: "퇴원"})
                fig = px.box(sl_plot, x="그룹", y="TPI_8월", color="그룹",
                             color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR})
                fig.update_layout(title="유지군 vs 퇴원군 TPI 분포 (8월)", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # Effect size
            if not ar["comp_aug"].empty and "효과크기_d" in ar["comp_aug"].columns:
                st.markdown("**효과크기 (8월)**")
                comp = ar["comp_aug"]
                fig = px.bar(comp, x="지표", y="효과크기_d", color="효과크기_d",
                             color_continuous_scale=["#2196F3", "#FFC107", "#F44336"], title="Cohen's d")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        # ─── TAB 4: 지표별 퇴원 연결 분석 ────────────────────────────────
        with atabs[3]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("지표별 퇴원 연결 분석")

            base_metrics = ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR", "TPI"]

            st.markdown("**단변량 로지스틱 회귀 (8월)**")
            if ar["uni_aug"] is not None and not ar["uni_aug"].empty:
                st.dataframe(ar["uni_aug"], use_container_width=True, hide_index=True)
            else:
                safe_warn("데이터가 없습니다.")

            st.markdown("**포인트-바이시리얼 상관 (8월)**")
            if ar["pb_aug"] is not None and not ar["pb_aug"].empty:
                st.dataframe(ar["pb_aug"], use_container_width=True, hide_index=True)
            else:
                safe_warn("데이터가 없습니다.")

            st.markdown("---")
            st.markdown("**분위수 × 재원기간 구간 교차표**")
            selected_ct = st.selectbox("교차표 지표 선택", base_metrics, key="ct_metric")
            if selected_ct in ar["crosstabs"]:
                ct = ar["crosstabs"][selected_ct]
                st.dataframe(ct, use_container_width=True, hide_index=True)
                rate_cols = [c for c in ct.columns if c.endswith("_퇴원율")]
                if rate_cols:
                    hm_data = ct[ct["분위"] != "전체"][["분위"] + rate_cols].set_index("분위")
                    hm_data.columns = [c.replace("_퇴원율", "") for c in hm_data.columns]
                    fig = px.imshow(hm_data.astype(float).values, x=hm_data.columns.tolist(), y=hm_data.index.tolist(),
                                   color_continuous_scale="YlOrRd", aspect="auto",
                                   labels=dict(x="재원기간구간", y="분위", color="퇴원율(%)"))
                    fig.update_layout(title=f"{selected_ct} 분위수 × 재원기간 퇴원율", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                safe_warn(f"{selected_ct} 교차표 생성 불가")

        # ─── TAB 5: TPI 심층 분석 ────────────────────────────────────────
        with atabs[4]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("TPI 심층 분석")

            if "TPI_8월" in sl.columns:
                sl_plot = sl.copy()
                sl_plot["그룹"] = sl_plot["target_churn_sep"].map({0: "유지", 1: "퇴원"})

                fig = px.histogram(sl_plot, x="TPI_8월", color="그룹", barmode="overlay", nbins=30,
                                   color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR}, opacity=0.7)
                fig.update_layout(title="TPI 분포 (유지/퇴원 중첩)", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                # TPI quintile churn
                if "TPI" in ar["crosstabs"]:
                    st.markdown("**TPI 분위수별 퇴원율**")
                    st.dataframe(ar["crosstabs"]["TPI"], use_container_width=True, hide_index=True)

                # Bottom students
                st.markdown("---")
                for pct, label in [(10, "하위 10%"), (20, "하위 20%"), (30, "하위 30%")]:
                    threshold = sl["TPI_8월"].quantile(pct / 100)
                    if pd.notna(threshold):
                        bottom = sl[sl["TPI_8월"] <= threshold].copy()
                        with st.expander(f"TPI {label} ({len(bottom)}명)", expanded=(pct == 20)):
                            dcols = ["캠퍼스", "학생명", "학생코드", "시험유형", "TPI_8월", "TPI_3월", "TPI_변화량",
                                     "P-Score_8월", "CI_8월", "QR_8월", "tenure_aug_fixed", "재원기간구간", "target_churn_sep"]
                            dcols = [c for c in dcols if c in bottom.columns]
                            bd = bottom[dcols].sort_values("TPI_8월").reset_index(drop=True)
                            bd = bd.rename(columns={"target_churn_sep": "9월상태(1=퇴원)", "tenure_aug_fixed": "재원기간"})
                            st.dataframe(bd, use_container_width=True, hide_index=True)

                # TPI vs tenure scatter
                if "tenure_aug_fixed" in sl.columns:
                    fig = px.scatter(sl_plot, x="tenure_aug_fixed", y="TPI_8월", color="그룹",
                                     color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR}, opacity=0.6, title="TPI vs 재원기간")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # TPI by campus
                if "캠퍼스" in sl.columns:
                    fig = px.box(sl_plot, x="캠퍼스", y="TPI_8월", color="그룹",
                                 color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR}, title="캠퍼스별 TPI")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                safe_warn("TPI_8월 데이터가 없습니다.")

        # ─── TAB 6: 월별 흐름 ────────────────────────────────────────────
        with atabs[5]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("월별 흐름 분석")

            flows = ar.get("monthly_flows", {})
            if flows:
                flow_metric = st.selectbox("지표 선택", list(flows.keys()), key="flow_metric")
                mf = flows[flow_metric]
                if not mf.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=mf["월"], y=mf["전체_평균"], name="전체", line=dict(color=BLUE2, width=2, dash="dash")))
                    fig.add_trace(go.Scatter(x=mf["월"], y=mf["유지군_평균"], name="유지군", line=dict(color=RETAINED_COLOR, width=2)))
                    fig.add_trace(go.Scatter(x=mf["월"], y=mf["퇴원군_평균"], name="퇴원군", line=dict(color=CHURNED_COLOR, width=2)))
                    fig.update_layout(title=f"{flow_metric} 월별 추이 (3~8월)", xaxis_title="월", yaxis_title=flow_metric, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(mf, use_container_width=True, hide_index=True)

                # Monthly heatmap
                st.markdown("---")
                st.markdown("**월 × 지표 히트맵 (퇴원군)**")
                hm_rows = []
                for m_name, mf_data in flows.items():
                    if not mf_data.empty:
                        row = {"지표": m_name}
                        for _, r in mf_data.iterrows():
                            row[f"{int(r['월'])}월"] = r["퇴원군_평균"]
                        hm_rows.append(row)
                if hm_rows:
                    hm_df = pd.DataFrame(hm_rows).set_index("지표")
                    fig = px.imshow(hm_df.astype(float).values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
                                   color_continuous_scale="Blues", aspect="auto")
                    fig.update_layout(title="퇴원군 월별 지표 히트맵", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # Change table
                st.markdown("---")
                st.markdown("**3→8 변화량 표**")
                change_cols = ["학생명", "학생코드", "TPI_3월", "TPI_8월", "TPI_변화량", "TPI_3~8평균", "TPI_3~8변동성", "target_churn_sep", "재원기간구간"]
                change_cols = [c for c in change_cols if c in sl.columns]
                if change_cols:
                    change_df = sl[change_cols].rename(columns={"target_churn_sep": "9월상태(1=퇴원)"})
                    if "TPI_변화량" in sl.columns:
                        change_df = change_df.sort_values("TPI_변화량")
                    st.dataframe(change_df, use_container_width=True, height=400, hide_index=True)
            else:
                safe_warn("월별 흐름 데이터가 없습니다.")

        # ─── TAB 7: 위험 신호 탐지 ───────────────────────────────────────
        with atabs[6]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("학생 위험 신호 탐지")
            risk_df = ar.get("risk_df", pd.DataFrame())

            if not risk_df.empty:
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(kpi_card_html("High 위험", int((risk_df["위험도"] == "High").sum()), "yellow"), unsafe_allow_html=True)
                with r2:
                    st.markdown(kpi_card_html("Medium 위험", int((risk_df["위험도"] == "Medium").sum()), "purple"), unsafe_allow_html=True)
                with r3:
                    st.markdown(kpi_card_html("Low 위험", int((risk_df["위험도"] == "Low").sum()), "sky"), unsafe_allow_html=True)

                with st.expander("위험점수 산정 기준"):
                    st.markdown("""
- 8월 TPI 하위 20%: +3점  ·  8월 CI 하위 20%: +2점  ·  8월 QR 하위 20%: +2점  ·  8월 BCV 하위 20%: +1점
- TPI 3→8 하락: +2점  ·  CI 3→8 하락: +1점  ·  QR 3→8 하락: +1점  ·  짧은 재원기간+저성과: +1점
- **High**: 7점 이상 | **Medium**: 4~6점 | **Low**: 3점 이하
""")

                # Tenure breakdown for risk
                st.markdown("**재원기간 구간별 위험 분포**")
                risk_tenure_rows = []
                for label in ["전체"] + list(TENURE_LABELS_INTERNAL):
                    sub = risk_df if label == "전체" else risk_df[risk_df["재원기간구간"] == label]
                    n = len(sub)
                    h = int((sub["위험도"] == "High").sum()) if n > 0 else 0
                    m_risk = int((sub["위험도"] == "Medium").sum()) if n > 0 else 0
                    lo = int((sub["위험도"] == "Low").sum()) if n > 0 else 0
                    risk_tenure_rows.append({"구간": label, "학생수": n, "High": h, "Medium": m_risk, "Low": lo})
                st.dataframe(pd.DataFrame(risk_tenure_rows), use_container_width=True, hide_index=True)

                risk_filter = st.multiselect("위험도 필터", ["High", "Medium", "Low"], default=["High", "Medium"], key="risk_filter")
                filtered_risk = risk_df[risk_df["위험도"].isin(risk_filter)].copy()
                dcols = ["캠퍼스", "학생명", "학생코드", "TPI_3월", "TPI_8월", "TPI_변화량",
                         "tenure_aug_fixed", "재원기간구간", "target_churn_sep", "위험점수", "위험도", "위험사유"]
                dcols = [c for c in dcols if c in filtered_risk.columns]
                show_risk = filtered_risk[dcols].sort_values("위험점수", ascending=False).reset_index(drop=True)
                show_risk = show_risk.rename(columns={"target_churn_sep": "9월상태(1=퇴원)", "tenure_aug_fixed": "재원기간"})
                st.dataframe(show_risk, use_container_width=True, height=500, hide_index=True)
                st.caption(f"표시: {len(show_risk)}명 / 코호트: {meta['cohort_size']}명")
            else:
                safe_warn("위험 데이터 없음")

        # ─── TAB 8: 캠퍼스/시험유형 ──────────────────────────────────────
        with atabs[7]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("캠퍼스/시험유형 분해")
            campus_df = ar.get("campus_df", pd.DataFrame())

            if not campus_df.empty:
                st.dataframe(campus_df, use_container_width=True, hide_index=True)
                if "평균TPI" in campus_df.columns and "퇴원율" in campus_df.columns:
                    fig = px.scatter(campus_df, x="평균TPI", y="퇴원율", size="학생수", text="캠퍼스",
                                     color="퇴원율", color_continuous_scale="YlOrRd", title="캠퍼스별 평균TPI vs 퇴원율")
                    fig.update_traces(textposition="top center")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("**시험유형(MT/LT) 비교**")
                if "시험유형" in sl.columns:
                    et_rows = []
                    for etype in ["MT", "LT"]:
                        sub = sl[sl["시험유형"].str.contains(etype, na=False)]
                        if len(sub) > 0:
                            cn = int(sub["target_churn_sep"].sum())
                            et_rows.append({"시험유형": etype, "학생수": len(sub), "퇴원수": cn, "퇴원율(%)": round(cn / len(sub) * 100, 1)})
                    if et_rows:
                        st.dataframe(pd.DataFrame(et_rows), use_container_width=True, hide_index=True)
            else:
                safe_warn("캠퍼스 데이터 없음")

        # ─── TAB 9: 상관분석 ─────────────────────────────────────────────
        with atabs[8]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("상관분석")

            corr_m = ar.get("corr_matrix", pd.DataFrame())
            if corr_m is not None and not corr_m.empty:
                fig = px.imshow(corr_m.values, x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="8월 지표 Pearson 상관행렬")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                safe_warn("상관행렬 데이터 없음")

            corr_p = ar.get("corr_pairs", pd.DataFrame())
            if corr_p is not None and not corr_p.empty:
                st.markdown("**지표 쌍별 상관계수**")
                st.dataframe(corr_p, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**target_churn_sep 포인트-바이시리얼 상관**")
            pb_all = ar.get("pb_all", pd.DataFrame())
            if pb_all is not None and not pb_all.empty:
                st.dataframe(pb_all, use_container_width=True, hide_index=True)
            else:
                safe_warn("데이터 없음")

        # ─── TAB 10: 회귀분석 ────────────────────────────────────────────
        with atabs[9]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("회귀분석")

            # Tenure breakdown table for regression context
            st.markdown("**재원기간 구간별 8월 지표 평균 (회귀 배경)**")
            tb_reg = _build_tenure_summary_table(sl, ["TPI", "P-Score", "CI", "QR", "B.CV"], "_8월")
            if not tb_reg.empty:
                st.dataframe(tb_reg, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**A. 단변량 로지스틱 회귀 (8월)**")
            if ar["uni_aug"] is not None and not ar["uni_aug"].empty:
                st.dataframe(ar["uni_aug"], use_container_width=True, hide_index=True)
            else:
                safe_warn("데이터 없음")

            st.markdown("**단변량 로지스틱 회귀 (3월)**")
            if ar["uni_mar"] is not None and not ar["uni_mar"].empty:
                st.dataframe(ar["uni_mar"], use_container_width=True, hide_index=True)
            else:
                safe_warn("데이터 없음")

            st.markdown("**단변량 로지스틱 회귀 (변화량)**")
            if ar["uni_delta"] is not None and not ar["uni_delta"].empty:
                st.dataframe(ar["uni_delta"], use_container_width=True, hide_index=True)
            else:
                safe_warn("데이터 없음")

            st.markdown("---")
            st.markdown("**B. 다변량 로지스틱 회귀 모델 비교**")
            multi = ar.get("multi_models", [])
            if multi:
                perf_rows = []
                for m in multi:
                    perf_rows.append({
                        "모델": m["모델"], "AUC": m["AUC"], "Accuracy": m["Accuracy"],
                        "Precision": m["Precision"], "Recall": m["Recall"], "F1": m["F1"],
                        "유효표본수": m["유효표본수"], "해석": m["해석"],
                    })
                perf_df = pd.DataFrame(perf_rows)
                st.dataframe(perf_df, use_container_width=True, hide_index=True)

                valid_perf = perf_df.dropna(subset=["AUC"])
                if not valid_perf.empty:
                    fig = px.bar(valid_perf, x="모델", y="AUC", color="AUC",
                                 color_continuous_scale="Blues", title="모델별 AUC 비교")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                for m in multi:
                    if m.get("coefs"):
                        with st.expander(f"{m['모델']} - 계수 상세"):
                            coef_rows = [{"변수": k, "계수": v["계수"], "오즈비": v["오즈비"]} for k, v in m["coefs"].items()]
                            st.dataframe(pd.DataFrame(coef_rows), use_container_width=True, hide_index=True)
            else:
                safe_warn("다변량 모델 데이터 없음")

        # ─── TAB 11: 생존분석 ────────────────────────────────────────────
        with atabs[10]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("생존분석")
            st.caption("재원기간을 보조적으로 활용한 생존분석")

            surv = ar.get("survival", {})
            if surv.get("error"):
                safe_warn(surv["error"])
            else:
                km_data = surv.get("km_data", {})
                if km_data:
                    st.markdown("**Kaplan-Meier 생존곡선 (TPI 그룹별)**")
                    fig = go.Figure()
                    colors = {"하위": CHURNED_COLOR, "중위": SKYBLUE2, "상위": RETAINED_COLOR}
                    for grp_name, sf in km_data.items():
                        fig.add_trace(go.Scatter(x=sf.index, y=sf.iloc[:, 0], name=grp_name, line=dict(color=colors.get(grp_name, BLUE1))))
                    fig.update_layout(title="Kaplan-Meier 생존곡선", xaxis_title="재원기간(개월)", yaxis_title="생존확률", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    safe_warn("KM 곡선 데이터 없음")

                cox_table = surv.get("cox_table")
                if cox_table is not None and not cox_table.empty:
                    st.markdown("**Cox 비례위험 모델**")
                    st.dataframe(cox_table, use_container_width=True, hide_index=True)
                    st.caption("HR > 1: 퇴원 위험 증가 방향, HR < 1: 감소 방향")
                else:
                    safe_warn("Cox 모델 데이터 없음")

            st.markdown("---")
            st.caption("생존분석은 보조분석입니다.")

        # ─── TAB 12: 통합 통계 ───────────────────────────────────────────
        with atabs[11]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("통합 통계 분석")

            integrated = ar.get("integrated", pd.DataFrame())
            if not integrated.empty:
                st.dataframe(integrated, use_container_width=True, hide_index=True)
            else:
                safe_warn("통합 요약 생성 불가")

            st.markdown("---")
            st.markdown("**단변량 회귀 핵심 요약**")
            if ar["uni_aug"] is not None and not ar["uni_aug"].empty and "AUC" in ar["uni_aug"].columns:
                top5 = ar["uni_aug"].nlargest(5, "AUC", keep="first")
                st.dataframe(top5, use_container_width=True, hide_index=True)

            st.markdown("**다변량 모델 성능 비교**")
            if ar.get("multi_models"):
                perf_rows = [{"모델": m["모델"], "AUC": m["AUC"], "F1": m["F1"], "유효표본수": m["유효표본수"]}
                             for m in ar["multi_models"] if pd.notna(m.get("AUC"))]
                if perf_rows:
                    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

            # Tenure summary
            st.markdown("---")
            st.markdown("**재원기간 구간별 종합 요약**")
            tenure_rows = []
            for label in ["전체"] + list(TENURE_LABELS_INTERNAL):
                sub = sl if label == "전체" else sl[sl["재원기간구간"] == label]
                n = len(sub)
                churned = int(sub["target_churn_sep"].sum()) if n > 0 else 0
                rate = round(churned / n * 100, 1) if n > 0 else 0
                avg_tpi = sub["TPI_8월"].mean() if "TPI_8월" in sub.columns and n > 0 else np.nan
                avg_ci = sub["CI_8월"].mean() if "CI_8월" in sub.columns and n > 0 else np.nan
                avg_qr = sub["QR_8월"].mean() if "QR_8월" in sub.columns and n > 0 else np.nan
                tenure_rows.append({
                    "구간": label, "학생수": n, "퇴원수": churned, "퇴원율(%)": rate,
                    "평균TPI": round(avg_tpi, 2) if pd.notna(avg_tpi) else "N/A",
                    "평균CI": round(avg_ci, 2) if pd.notna(avg_ci) else "N/A",
                    "평균QR": round(avg_qr, 2) if pd.notna(avg_qr) else "N/A",
                })
            st.dataframe(pd.DataFrame(tenure_rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            safe_info("통계적으로 유의하더라도 표본이 작거나 효과크기가 약한 경우 실무 적용에 주의가 필요합니다.")

        # ─── TAB 13: 다운로드 ────────────────────────────────────────────
        with atabs[12]:
            _condition_badge(_formula, _s_etypes, _s_months, include_na=_inc_na)
            st.subheader("결과 다운로드")
            st.caption("코호트 기준 결과")

            download_files = {}

            # 1. Config
            config_csv = ac_to_csv_bytes(config_df)
            download_files["01_TPI지수구성표.csv"] = config_csv
            st.download_button("1. TPI 지수 구성표", data=config_csv, file_name="TPI지수구성표.csv", mime="text/csv", key="dl_config")

            # 2. Student detail
            if not sl.empty:
                sl_csv = ac_to_csv_bytes(sl)
                download_files["02_학생별_상세.csv"] = sl_csv
                st.download_button("2. 학생별 상세 결과", data=sl_csv, file_name="학생별_상세.csv", mime="text/csv", key="dl_sl")

            # 3. Cohort
            cs_csv = ac_to_csv_bytes(pd.DataFrame([meta]))
            download_files["03_코호트_요약.csv"] = cs_csv
            st.download_button("3. 코호트 요약", data=cs_csv, file_name="코호트_요약.csv", mime="text/csv", key="dl_cohort")

            # 4. Comparison
            if not ar["comp_aug"].empty:
                comp_csv = ac_to_csv_bytes(ar["comp_aug"])
                download_files["04_비교표.csv"] = comp_csv
                st.download_button("4. 유지 vs 퇴원 비교표", data=comp_csv, file_name="비교표.csv", mime="text/csv", key="dl_comp")

            # 5. Cross tables
            if ar["crosstabs"]:
                all_ct = pd.concat([ct.assign(지표명=m) for m, ct in ar["crosstabs"].items()], ignore_index=True)
                ct_csv = ac_to_csv_bytes(all_ct)
                download_files["05_교차표.csv"] = ct_csv
                st.download_button("5. 분위수×재원기간 교차표", data=ct_csv, file_name="교차표.csv", mime="text/csv", key="dl_ct")

            # 6. Monthly flow
            if ar["monthly_flows"]:
                all_mf = pd.concat([mf.assign(지표=m) for m, mf in ar["monthly_flows"].items()], ignore_index=True)
                mf_csv = ac_to_csv_bytes(all_mf)
                download_files["06_월별흐름.csv"] = mf_csv
                st.download_button("6. 월별 흐름", data=mf_csv, file_name="월별흐름.csv", mime="text/csv", key="dl_mf")

            # 7. Risk
            if not ar["risk_df"].empty:
                risk_csv = ac_to_csv_bytes(ar["risk_df"])
                download_files["07_위험학생.csv"] = risk_csv
                st.download_button("7. 위험학생 리스트", data=risk_csv, file_name="위험학생.csv", mime="text/csv", key="dl_risk")

            # 8. Campus
            if not ar["campus_df"].empty:
                camp_csv = ac_to_csv_bytes(ar["campus_df"])
                download_files["08_캠퍼스.csv"] = camp_csv
                st.download_button("8. 캠퍼스 요약", data=camp_csv, file_name="캠퍼스.csv", mime="text/csv", key="dl_campus")

            # 9. Correlation
            if ar["corr_pairs"] is not None and not ar["corr_pairs"].empty:
                corr_csv = ac_to_csv_bytes(ar["corr_pairs"])
                download_files["09_상관분석.csv"] = corr_csv
                st.download_button("9. 상관분석", data=corr_csv, file_name="상관분석.csv", mime="text/csv", key="dl_corr")

            # 10. Regression
            if ar["uni_aug"] is not None and not ar["uni_aug"].empty:
                reg_csv = ac_to_csv_bytes(ar["uni_aug"])
                download_files["10_회귀_단변량.csv"] = reg_csv
                st.download_button("10. 회귀 (단변량)", data=reg_csv, file_name="회귀_단변량.csv", mime="text/csv", key="dl_reg_uni")

            if ar["multi_models"]:
                perf_dl = pd.DataFrame([{k: v for k, v in m.items() if k != "coefs"} for m in ar["multi_models"]])
                multi_csv = ac_to_csv_bytes(perf_dl)
                download_files["10_회귀_다변량.csv"] = multi_csv
                st.download_button("10. 회귀 (다변량)", data=multi_csv, file_name="회귀_다변량.csv", mime="text/csv", key="dl_reg_multi")

            # 11. Survival
            surv = ar.get("survival", {})
            if surv.get("cox_table") is not None and not surv["cox_table"].empty:
                surv_csv = ac_to_csv_bytes(surv["cox_table"])
                download_files["11_생존분석.csv"] = surv_csv
                st.download_button("11. 생존분석", data=surv_csv, file_name="생존분석.csv", mime="text/csv", key="dl_surv")

            # 12. Integrated
            if not integrated.empty:
                int_csv = ac_to_csv_bytes(integrated)
                download_files["12_통합통계.csv"] = int_csv
                st.download_button("12. 통합 통계", data=int_csv, file_name="통합통계.csv", mime="text/csv", key="dl_integrated")

            # ZIP
            st.markdown("---")
            if download_files:
                zip_bytes = build_zip_package(download_files)
                st.download_button(
                    "전체 결과 ZIP 다운로드", data=zip_bytes,
                    file_name="PolyRetentionSignal_전체결과.zip", mime="application/zip",
                    key="dl_zip", type="primary",
                )

from __future__ import annotations

import traceback
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
)

st.set_page_config(page_title="PolyRetentionSignal", layout="wide")
st.markdown(inject_custom_css(), unsafe_allow_html=True)
st.title("PolyRetentionSignal")

st.markdown(
    """
    - 시험 데이터 업로드: `.xlsx` 여러 개 가능
    - 학생 데이터 업로드: `.csv` 1개만 가능
    - 업로드 즉시 분석 가능 구조
    - 학생 데이터의 우측 끝에서 두번째 열 = 재원기간, 마지막 열 = 재원여부(1 재원 / 0 퇴원)
    """
)

# ── Sidebar: Data Upload ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("데이터 입력")
    exam_files = st.file_uploader(
        "시험 데이터 업로드 (.xlsx, 여러 개 가능)",
        type=["xlsx"],
        accept_multiple_files=True,
        help="시험 성적 원본 파일 여러 개를 업로드한다.",
    )
    student_file = st.file_uploader(
        "학생 데이터 업로드 (.csv, 1개만 가능)",
        type=["csv"],
        accept_multiple_files=False,
        help="학생 마스터 1개만 업로드한다.",
    )


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
            errors.append(f"시험 업로드 칸에 시험 파일이 아닌 항목이 들어왔다: {name}")
        else:
            valid_exams.append(f)
    if student_obj is not None:
        name = getattr(student_obj, "name", str(student_obj))
        kind = detect_file_kind(name, expected_ext=".csv")
        if kind != "student":
            errors.append(f"학생 업로드 칸에 학생 파일이 아닌 항목이 들어왔다: {name}")
    return valid_exams, student_obj, errors


@st.cache_data(show_spinner=False)
def load_all(exam_objs, student_obj):
    exam_frames = [read_single_exam(f) for f in exam_objs]
    raw = pd.concat(exam_frames, ignore_index=True) if exam_frames else pd.DataFrame()
    student_info = read_student_info(student_obj) if student_obj is not None else pd.DataFrame()
    item = build_item_stats(raw) if not raw.empty else pd.DataFrame()
    summary = build_student_summary(raw, student_info) if not raw.empty else pd.DataFrame()
    return raw, item, summary, student_info


exam_files, student_file, validation_errors = validate_uploads(exam_files, student_file)
if validation_errors:
    for msg in validation_errors:
        safe_error(msg)
    st.stop()

if not exam_files:
    safe_info("시험 데이터 업로드가 필요합니다.")
    st.stop()
if student_file is None:
    safe_info("학생 데이터 업로드가 필요합니다.")
    st.stop()

try:
    raw_df, item_df, summary_df, student_info_df = load_all(exam_files, student_file)
except Exception:
    safe_error("데이터 로딩 중 오류가 발생했습니다. 파일 형식을 확인해 주세요.")
    st.stop()

if summary_df.empty:
    safe_warn("처리 가능한 데이터가 없습니다.")
    st.stop()

# ── Filters ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    selected_exam_types = st.multiselect(
        "시험유형",
        options=sorted(summary_df["시험유형"].dropna().unique().tolist()),
        default=sorted(summary_df["시험유형"].dropna().unique().tolist()),
    )
with col2:
    month_opts = sorted(
        summary_df.loc[summary_df["시험유형"].isin(selected_exam_types), "월"]
        .dropna().astype(int).unique().tolist()
    )
    selected_months = st.multiselect("월", options=month_opts, default=month_opts)
with col3:
    # Level filter (campus_type if available)
    if "캠퍼스" in summary_df.columns:
        campus_opts = sorted(summary_df["캠퍼스"].dropna().unique().tolist())
        selected_campuses = st.multiselect("캠퍼스 필터", options=campus_opts, default=campus_opts)
    else:
        selected_campuses = []
with col4:
    selected_students = st.multiselect(
        "학생코드 필터(선택)",
        options=summary_df["학생코드"].dropna().astype(str).unique().tolist(),
        default=[],
    )

view_df = summary_df[
    summary_df["시험유형"].isin(selected_exam_types) & summary_df["월"].isin(selected_months)
].copy()
if selected_campuses and "캠퍼스" in view_df.columns:
    view_df = view_df[view_df["캠퍼스"].isin(selected_campuses)].copy()
if selected_students:
    view_df = view_df[view_df["학생코드"].astype(str).isin(selected_students)].copy()

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["학생 성적표", "문항 정답률", "TPI 계산"])

with tab1:
    st.subheader("학생 성적표")
    st.dataframe(view_df, use_container_width=True, height=600)
    csv_data = view_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="학생 성적표 CSV 다운로드",
        data=csv_data,
        file_name="Poly_Student_Summary.csv",
        mime="text/csv",
        key="btn_download_student_v2",
    )

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
        key="btn_download_item",
    )

with tab3:
    st.subheader("TPI 계산")
    st.markdown("사용 가능한 별칭: `P`, `T`, `BCV`, `CI`, `QR`, `CT`, `CCV`, `CQR`, `CV`")
    st.markdown("사용 가능한 연산자: `+`, `-`, `*`, `/`, `**` (거듭제곱), `( )` (괄호)")

    formula_mode = st.radio(
        "수식 입력 방식 선택",
        options=["가중치(비율) 기반 자동 생성", "자유 수식 직접 입력"],
        horizontal=True,
    )

    if formula_mode == "가중치(비율) 기반 자동 생성":
        st.write("아래에서 채택할 지표와 비율을 설정하면 자동으로 수식이 생성됩니다.")
        default_weights = {
            "P": 20, "T": 20, "BCV": 15, "CI": 15, "QR": 10,
            "CT": 10, "CCV": 5, "CQR": 5, "CV": 0,
        }
        enabled_weights = {}
        wcols = st.columns(5)
        aliases = list(default_weights.keys())
        for i, alias in enumerate(aliases):
            with wcols[i % 5]:
                default_on = alias != "CV"  # CV off by default
                enabled = st.checkbox(f"{alias} 채택", value=default_on, key=f"en_{alias}")
                weight = st.number_input(
                    f"{alias} 비율%",
                    min_value=0.0, max_value=100.0,
                    value=float(default_weights[alias]),
                    step=1.0, key=f"wt_{alias}",
                )
                enabled_weights[alias] = weight if enabled else 0.0
        formula = make_default_formula(enabled_weights)
        st.text_input("생성된 TPI 수식", value=formula, disabled=True, key="formula_generated")
    else:
        st.write("원하는 대로 수식을 직접 작성할 수 있습니다.")
        formula = st.text_input(
            "TPI 수식 입력창",
            value="(P*20.0 + T*20.0 + BCV*15.0 + CI*15.0 + QR*10.0 + CT*10.0 + CCV*5.0 + CQR*5.0) / 100.0",
            help="예시: P*0.3 + T*0.7",
            key="formula_free",
        )

    # Session state init
    for key in ["tpi_matrix", "analysis_results", "formula_used", "enabled_weights_used"]:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.button("TPI 계산 실행", type="primary"):
        try:
            with st.spinner("TPI 계산 및 코호트 분석 수행 중..."):
                # 1. TPI calculation
                tpi_df = apply_tpi_formula(view_df, formula)
                tpi_matrix = build_tpi_matrix(tpi_df)
                st.session_state.tpi_matrix = tpi_matrix

                # Store formula info
                if formula_mode == "가중치(비율) 기반 자동 생성":
                    st.session_state.enabled_weights_used = enabled_weights.copy()
                else:
                    st.session_state.enabled_weights_used = None
                st.session_state.formula_used = formula

                # 2. Apply TPI to full summary (all months, not just filtered view)
                full_tpi_df = apply_tpi_formula(summary_df, formula)

                # 3. Extract cohort
                cohort_df, cohort_meta = extract_cohort(full_tpi_df, student_info_df)

                if cohort_df.empty:
                    st.session_state.analysis_results = {"error": cohort_meta.get("error", "코호트 추출 실패")}
                else:
                    # 4. Build student-level data
                    student_level = build_cohort_student_level(cohort_df, student_info_df)

                    if student_level.empty:
                        st.session_state.analysis_results = {"error": "학생 수준 데이터 생성 실패"}
                    else:
                        # 5. Run all analyses
                        base_metrics = ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR", "TPI"]
                        aug_metrics = [f"{m}_8월" for m in base_metrics]
                        mar_metrics = [f"{m}_3월" for m in base_metrics]
                        delta_metrics = [f"{m}_변화량" for m in base_metrics]

                        # Comparison tables
                        comp_aug = build_comparison_table(student_level, base_metrics, "_8월")
                        comp_mar = build_comparison_table(student_level, base_metrics, "_3월")
                        comp_delta = build_comparison_table(student_level, base_metrics, "_변화량")

                        # Point-biserial
                        pb_aug = point_biserial_table(student_level, aug_metrics)
                        pb_mar = point_biserial_table(student_level, mar_metrics)
                        pb_delta = point_biserial_table(student_level, delta_metrics)

                        # Quantile × tenure cross tables
                        crosstabs = {}
                        for m in base_metrics:
                            col = f"{m}_8월"
                            if col in student_level.columns:
                                ct = quantile_tenure_crosstab(student_level, col)
                                if not ct.empty:
                                    crosstabs[m] = ct

                        # Univariate logistic
                        uni_aug = univariate_logistic_table(student_level, aug_metrics)
                        uni_mar = univariate_logistic_table(student_level, mar_metrics)
                        uni_delta = univariate_logistic_table(student_level, delta_metrics)

                        # Multivariate logistic
                        multi_models = run_all_multivariate_models(student_level)

                        # Correlation
                        corr_matrix = correlation_matrix(student_level, aug_metrics)
                        corr_pairs = correlation_pairs_table(student_level, aug_metrics)
                        pb_all = point_biserial_table(student_level, aug_metrics + mar_metrics + delta_metrics)

                        # Survival
                        survival = run_survival_analysis(student_level)

                        # Risk scores
                        risk_df = compute_risk_scores(student_level)

                        # Monthly flow
                        monthly_flows = {}
                        for m in base_metrics:
                            mf = build_monthly_flow(cohort_df, m)
                            if not mf.empty:
                                monthly_flows[m] = mf

                        # Campus summary
                        campus_df = campus_summary(risk_df)

                        # Integrated summary
                        integrated = build_integrated_summary(pb_aug, uni_aug, multi_models, comp_aug)

                        st.session_state.analysis_results = {
                            "cohort_meta": cohort_meta,
                            "cohort_df": cohort_df,
                            "student_level": student_level,
                            "comp_aug": comp_aug,
                            "comp_mar": comp_mar,
                            "comp_delta": comp_delta,
                            "pb_aug": pb_aug,
                            "pb_mar": pb_mar,
                            "pb_delta": pb_delta,
                            "pb_all": pb_all,
                            "crosstabs": crosstabs,
                            "uni_aug": uni_aug,
                            "uni_mar": uni_mar,
                            "uni_delta": uni_delta,
                            "multi_models": multi_models,
                            "corr_matrix": corr_matrix,
                            "corr_pairs": corr_pairs,
                            "survival": survival,
                            "risk_df": risk_df,
                            "monthly_flows": monthly_flows,
                            "campus_df": campus_df,
                            "integrated": integrated,
                        }
                st.success("TPI 계산 및 코호트 분석 완료")
        except Exception:
            safe_error("TPI 계산 중 오류가 발생했습니다. 수식과 데이터를 확인해 주세요.")

    # Show TPI matrix
    if st.session_state.tpi_matrix is not None:
        st.dataframe(st.session_state.tpi_matrix, use_container_width=True, height=650)
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button(
                "TPI 매트릭스 CSV 다운로드",
                data=to_csv_bytes(st.session_state.tpi_matrix),
                file_name="PolyRetentionSignal_TPI_matrix.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_tpi",
            )

    with st.expander("TPI 지표 별칭(Alias) 가이드"):
        st.markdown("""
        - **P**: P-Score (전체 정답률)
        - **T**: T-Score (표준점수 기반 0-100)
        - **BCV**: B.CV (과목별 편차 역수 - 균형도)
        - **CI**: CI (정답 문항의 난이도 가중치 기반 성취 지표)
        - **QR**: QR (백분위 순위)
        - **CT**: C.T-Score (캠퍼스 내 T-Score)
        - **CCV**: C.CV (캠퍼스 내 P-Score 변동성)
        - **CQR**: C.QR (캠퍼스 내 백분위 순위)
        - **CV**: CV (전체 시험 변동성)
        """)

    # ── Analysis Hub ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown(section_header("코호트 기반 9월 이탈 방향성 분석 허브"), unsafe_allow_html=True)

    ar = st.session_state.analysis_results
    if ar is None:
        safe_info("TPI 계산 실행 버튼을 클릭하면 코호트 분석이 자동으로 수행됩니다.")
    elif isinstance(ar, dict) and "error" in ar and isinstance(ar["error"], str):
        safe_error(ar["error"])
    else:
        # Banner
        st.markdown(
            '<div class="info-box"><strong>2025년 3월 GT2 재원 + 2025년 8월 GT2 재원 학생 코호트 기준, '
            '2025년 9월 재원/퇴원 비교 분석</strong><br>'
            '본 분석은 인과 증명이 아니라, 9월 퇴원과 함께 관찰되는 통계적 방향성 탐색을 목적으로 합니다.</div>',
            unsafe_allow_html=True,
        )

        meta = ar["cohort_meta"]
        sl = ar["student_level"]
        cohort_df = ar["cohort_df"]

        ANALYSIS_TABS = [
            "TPI 지수 구성",
            "종합 요약",
            "재원 vs 9월 퇴원 비교",
            "지표별 퇴원 연결 분석",
            "TPI 심층 분석",
            "월별 흐름 분석",
            "학생 위험 신호 탐지",
            "캠퍼스/시험유형 분해",
            "상관분석",
            "회귀분석",
            "생존분석",
            "통합 통계 분석",
            "결과 다운로드",
        ]
        atabs = st.tabs(ANALYSIS_TABS)

        # ─── TAB 0: TPI 지수 구성 ───────────────────────────────────────────
        with atabs[0]:
            st.subheader("TPI 지수 구성 정의")
            st.markdown(f"**현재 적용된 TPI 수식:** `{st.session_state.formula_used}`")

            # Build definition table
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
            rows = []
            all_aliases = ["P", "T", "BCV", "CI", "QR", "CT", "CCV", "CQR", "CV"]
            for alias in all_aliases:
                if ew is not None:
                    included = ew.get(alias, 0) > 0
                    weight = ew.get(alias, 0)
                else:
                    included = alias in (st.session_state.formula_used or "")
                    weight = "-"
                rows.append({
                    "지표": alias,
                    "실제컬럼": ALIAS_TO_COLUMN.get(alias, alias),
                    "포함여부": "포함" if included else "제외",
                    "가중치": weight,
                    "방향성": direction_map.get(alias, ""),
                    "정규화방식": norm_map.get(alias, ""),
                })
            config_df = pd.DataFrame(rows)
            st.dataframe(config_df, use_container_width=True, hide_index=True)

            if ew:
                total_w = sum(v for v in ew.values() if v > 0)
                st.markdown(f"**가중치 합계:** {total_w}")

            st.markdown("---")
            st.markdown("**현재 필터 조건:**")
            st.markdown(f"- 시험유형: {', '.join(selected_exam_types)}")
            st.markdown(f"- 월: {', '.join(str(m) for m in selected_months)}")
            if selected_campuses:
                st.markdown(f"- 캠퍼스: {', '.join(selected_campuses)}")

            st.markdown("**분석 코호트 정의:**")
            st.markdown("- 2025년 3월 GT2 재원 + 2025년 8월 GT2 재원 학생")
            st.markdown(f"- 코호트 규모: {meta['cohort_size']}명")

            st.markdown("**재원기간 구간 정의 (2025년 8월 기준 고정):**")
            for label in TENURE_LABELS_INTERNAL:
                st.markdown(f"- {label}")

        # ─── TAB 1: 종합 요약 ───────────────────────────────────────────────
        with atabs[1]:
            st.subheader("종합 요약")

            # KPI Cards
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
            with k5:
                avg_tpi = sl["TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
                st.markdown(kpi_card_html("평균 TPI (8월)", f"{avg_tpi:.1f}" if pd.notna(avg_tpi) else "N/A"), unsafe_allow_html=True)
            with k6:
                churned_tpi = sl.loc[sl["target_churn_sep"] == 1, "TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
                st.markdown(kpi_card_html("퇴원군 평균 TPI", f"{churned_tpi:.1f}" if pd.notna(churned_tpi) else "N/A", "yellow"), unsafe_allow_html=True)
            with k7:
                retained_tpi = sl.loc[sl["target_churn_sep"] == 0, "TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
                st.markdown(kpi_card_html("유지군 평균 TPI", f"{retained_tpi:.1f}" if pd.notna(retained_tpi) else "N/A", "sky"), unsafe_allow_html=True)

            # Additional metric averages
            extra_metrics = ["CI", "QR", "B.CV", "T-Score", "P-Score"]
            em_cols = st.columns(len(extra_metrics))
            for i, m in enumerate(extra_metrics):
                col_name = f"{m}_8월"
                with em_cols[i]:
                    val = sl[col_name].mean() if col_name in sl.columns else np.nan
                    st.metric(f"평균 {m}", f"{val:.1f}" if pd.notna(val) else "N/A")

            # Donut chart
            st.markdown("")
            dc1, dc2 = st.columns(2)
            with dc1:
                fig = go.Figure(data=[go.Pie(
                    labels=["재원", "퇴원"],
                    values=[meta["retained_count"], meta["churned_count"]],
                    hole=0.5,
                    marker_colors=[RETAINED_COLOR, CHURNED_COLOR],
                )])
                fig.update_layout(title="9월 재원/퇴원 비율", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with dc2:
                # Retained vs Churned avg metric comparison
                if not ar["comp_aug"].empty:
                    comp = ar["comp_aug"]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="유지군", x=comp["지표"], y=comp["유지군_평균"], marker_color=RETAINED_COLOR))
                    fig.add_trace(go.Bar(name="퇴원군", x=comp["지표"], y=comp["퇴원군_평균"], marker_color=CHURNED_COLOR))
                    fig.update_layout(barmode="group", title="유지군 vs 퇴원군 평균 지표 비교 (8월)", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            # Summary text
            st.markdown("---")
            st.markdown(f"**요약:** 코호트 {meta['cohort_size']}명 중 {meta['churned_count']}명({meta['churn_rate']:.1f}%)이 9월 퇴원으로 관찰되었습니다. "
                        f"퇴원군의 평균 TPI({churned_tpi:.1f if pd.notna(churned_tpi) else 'N/A'})는 "
                        f"유지군({retained_tpi:.1f if pd.notna(retained_tpi) else 'N/A'})보다 낮은 경향이 관찰됩니다.")
            st.caption("* 위 결과는 통계적 방향성이며, 인과관계를 의미하지 않습니다.")

        # ─── TAB 2: 재원 vs 9월 퇴원 비교 ───────────────────────────────────
        with atabs[2]:
            st.subheader("재원 vs 9월 퇴원 비교")
            st.caption(f"코호트 n={meta['cohort_size']} | 유지군 n={meta['retained_count']} | 퇴원군 n={meta['churned_count']}")

            for label, comp_df in [("8월 값 비교", ar["comp_aug"]), ("3월 값 비교", ar["comp_mar"]), ("3→8 변화량 비교", ar["comp_delta"])]:
                st.markdown(f"**{label}**")
                if not comp_df.empty:
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                else:
                    safe_warn(f"{label} 데이터가 없습니다.")

            # Box plots
            if "TPI_8월" in sl.columns:
                st.markdown("**8월 TPI 분포 비교**")
                sl_plot = sl.copy()
                sl_plot["그룹"] = sl_plot["target_churn_sep"].map({0: "유지", 1: "퇴원"})
                fig = px.box(sl_plot, x="그룹", y="TPI_8월", color="그룹",
                             color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR})
                fig.update_layout(title="유지군 vs 퇴원군 TPI 분포 (8월)", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # Difference heatmap
            if not ar["comp_aug"].empty:
                st.markdown("**효과크기 히트맵 (8월)**")
                comp = ar["comp_aug"]
                if "효과크기_d" in comp.columns:
                    fig = px.bar(comp, x="지표", y="효과크기_d", color="효과크기_d",
                                 color_continuous_scale=["#2196F3", "#FFC107", "#F44336"],
                                 title="Cohen's d 효과크기")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

        # ─── TAB 3: 지표별 퇴원 연결 분석 ───────────────────────────────────
        with atabs[3]:
            st.subheader("지표별 퇴원 연결 분석")
            st.caption(f"코호트 n={meta['cohort_size']}")

            # Overall comparison table
            st.markdown("**전체 지표 비교표**")
            if not ar["uni_aug"].empty:
                st.dataframe(ar["uni_aug"], use_container_width=True, hide_index=True)

            # Point-biserial
            st.markdown("**포인트-바이시리얼 상관 (8월 지표)**")
            if not ar["pb_aug"].empty:
                st.dataframe(ar["pb_aug"], use_container_width=True, hide_index=True)

            # Quantile × Tenure cross tables
            st.markdown("---")
            st.markdown("**분위수 × 재원기간 구간 교차표**")
            base_metrics = ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR", "TPI"]

            selected_ct_metric = st.selectbox("교차표 지표 선택", base_metrics, key="ct_metric_select")
            if selected_ct_metric in ar["crosstabs"]:
                ct = ar["crosstabs"][selected_ct_metric]
                st.dataframe(ct, use_container_width=True, hide_index=True)

                # Heatmap for churn rates
                rate_cols = [c for c in ct.columns if c.endswith("_퇴원율")]
                if rate_cols:
                    hm_data = ct[ct["분위"] != "전체"][["분위"] + rate_cols].set_index("분위")
                    hm_data.columns = [c.replace("_퇴원율", "") for c in hm_data.columns]
                    fig = px.imshow(hm_data.values, x=hm_data.columns.tolist(), y=hm_data.index.tolist(),
                                   color_continuous_scale="YlOrRd", aspect="auto",
                                   labels=dict(x="재원기간구간", y="분위", color="퇴원율(%)"))
                    fig.update_layout(title=f"{selected_ct_metric} 분위수 × 재원기간 퇴원율 히트맵", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                safe_warn(f"{selected_ct_metric}에 대한 교차표를 생성할 수 없습니다.")

        # ─── TAB 4: TPI 심층 분석 ───────────────────────────────────────────
        with atabs[4]:
            st.subheader("TPI 심층 분석")

            if "TPI_8월" in sl.columns:
                # Distribution
                sl_plot = sl.copy()
                sl_plot["그룹"] = sl_plot["target_churn_sep"].map({0: "유지", 1: "퇴원"})

                fig = px.histogram(sl_plot, x="TPI_8월", color="그룹", barmode="overlay", nbins=30,
                                   color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR},
                                   opacity=0.7)
                fig.update_layout(title="TPI 분포 (유지군/퇴원군 중첩)", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

                # TPI quintile churn rate
                if "TPI" in ar["crosstabs"]:
                    st.markdown("**TPI 분위수별 9월 퇴원율**")
                    st.dataframe(ar["crosstabs"]["TPI"], use_container_width=True, hide_index=True)

                # Bottom students
                st.markdown("---")
                for pct, label in [(10, "하위 10%"), (20, "하위 20%"), (30, "하위 30%")]:
                    threshold = sl["TPI_8월"].quantile(pct / 100)
                    if pd.notna(threshold):
                        bottom = sl[sl["TPI_8월"] <= threshold].copy()
                        with st.expander(f"TPI {label} 학생 ({len(bottom)}명)", expanded=(pct == 20)):
                            display_cols = ["캠퍼스", "학생명", "학생코드", "시험유형", "TPI_8월", "TPI_3월", "TPI_변화량",
                                            "P-Score_8월", "T-Score_8월", "B.CV_8월", "CI_8월", "QR_8월",
                                            "C.T-Score_8월", "C.CV_8월", "C.QR_8월",
                                            "tenure_aug_fixed", "재원기간구간", "target_churn_sep"]
                            display_cols = [c for c in display_cols if c in bottom.columns]
                            bottom_display = bottom[display_cols].sort_values("TPI_8월").reset_index(drop=True)
                            bottom_display = bottom_display.rename(columns={"target_churn_sep": "9월상태(1=퇴원)", "tenure_aug_fixed": "재원기간"})
                            st.dataframe(bottom_display, use_container_width=True, hide_index=True)

                # TPI vs tenure
                if "tenure_aug_fixed" in sl.columns:
                    fig = px.scatter(sl_plot, x="tenure_aug_fixed", y="TPI_8월", color="그룹",
                                     color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR},
                                     opacity=0.6, title="TPI vs 재원기간")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # TPI by campus
                if "캠퍼스" in sl.columns:
                    fig = px.box(sl_plot, x="캠퍼스", y="TPI_8월", color="그룹",
                                 color_discrete_map={"유지": RETAINED_COLOR, "퇴원": CHURNED_COLOR},
                                 title="캠퍼스별 TPI 분포")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

        # ─── TAB 5: 월별 흐름 분석 ──────────────────────────────────────────
        with atabs[5]:
            st.subheader("월별 흐름 분석")

            flows = ar["monthly_flows"]
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

                # Monthly heatmap across metrics
                st.markdown("---")
                st.markdown("**월 × 지표 히트맵 (퇴원군 평균)**")
                hm_rows = []
                for m_name, mf_data in flows.items():
                    if not mf_data.empty:
                        row = {"지표": m_name}
                        for _, r in mf_data.iterrows():
                            row[f"{int(r['월'])}월"] = r["퇴원군_평균"]
                        hm_rows.append(row)
                if hm_rows:
                    hm_df = pd.DataFrame(hm_rows).set_index("지표")
                    fig = px.imshow(hm_df.values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
                                   color_continuous_scale="Blues", aspect="auto")
                    fig.update_layout(title="퇴원군 월별 지표 히트맵", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # Change table
                st.markdown("---")
                st.markdown("**3→8 변화량 표**")
                change_cols = ["학생명", "학생코드", "TPI_3월", "TPI_8월", "TPI_변화량", "TPI_3~8평균", "TPI_3~8변동성", "target_churn_sep", "재원기간구간"]
                change_cols = [c for c in change_cols if c in sl.columns]
                if change_cols:
                    change_df = sl[change_cols].rename(columns={"target_churn_sep": "9월상태(1=퇴원)"}).sort_values("TPI_변화량" if "TPI_변화량" in sl.columns else change_cols[0])
                    st.dataframe(change_df, use_container_width=True, height=400, hide_index=True)
            else:
                safe_warn("월별 흐름 데이터가 없습니다.")

        # ─── TAB 6: 학생 위험 신호 탐지 ─────────────────────────────────────
        with atabs[6]:
            st.subheader("학생 위험 신호 탐지")
            risk_df = ar["risk_df"]

            if not risk_df.empty:
                # Summary cards
                r1, r2, r3 = st.columns(3)
                with r1:
                    high_n = int((risk_df["위험도"] == "High").sum())
                    st.markdown(kpi_card_html("High 위험", high_n, "yellow"), unsafe_allow_html=True)
                with r2:
                    med_n = int((risk_df["위험도"] == "Medium").sum())
                    st.markdown(kpi_card_html("Medium 위험", med_n, "purple"), unsafe_allow_html=True)
                with r3:
                    low_n = int((risk_df["위험도"] == "Low").sum())
                    st.markdown(kpi_card_html("Low 위험", low_n, "sky"), unsafe_allow_html=True)

                # Risk scoring rules info
                with st.expander("위험점수 산정 기준"):
                    st.markdown("""
                    - 8월 TPI 하위 20%: +3점
                    - 8월 CI 하위 20%: +2점
                    - 8월 QR 하위 20%: +2점
                    - 8월 BCV 하위 20%: +1점
                    - TPI 3→8 하락: +2점
                    - CI 3→8 하락: +1점
                    - QR 3→8 하락: +1점
                    - 짧은 재원기간 + 저성과: +1점
                    - **High**: 7점 이상 | **Medium**: 4~6점 | **Low**: 3점 이하
                    """)

                # Filter by risk level
                risk_filter = st.multiselect("위험도 필터", ["High", "Medium", "Low"], default=["High", "Medium"], key="risk_filter")
                filtered_risk = risk_df[risk_df["위험도"].isin(risk_filter)].copy()

                display_cols = ["캠퍼스", "학생명", "학생코드", "TPI_3월", "TPI_8월", "TPI_변화량",
                                "tenure_aug_fixed", "재원기간구간", "target_churn_sep", "위험점수", "위험도", "위험사유"]
                display_cols = [c for c in display_cols if c in filtered_risk.columns]
                show_risk = filtered_risk[display_cols].sort_values("위험점수", ascending=False).reset_index(drop=True)
                show_risk = show_risk.rename(columns={"target_churn_sep": "9월상태(1=퇴원)", "tenure_aug_fixed": "재원기간"})
                st.dataframe(show_risk, use_container_width=True, height=500, hide_index=True)
                st.caption(f"표시: {len(show_risk)}명 / 코호트: {meta['cohort_size']}명")
            else:
                safe_warn("위험 신호 데이터가 없습니다.")

        # ─── TAB 7: 캠퍼스/시험유형 분해 ────────────────────────────────────
        with atabs[7]:
            st.subheader("캠퍼스/시험유형 분해 분석")
            campus_df = ar["campus_df"]

            if not campus_df.empty:
                st.dataframe(campus_df, use_container_width=True, hide_index=True)

                # Scatter: avg TPI vs churn rate
                if "평균TPI" in campus_df.columns and "퇴원율" in campus_df.columns:
                    fig = px.scatter(campus_df, x="평균TPI", y="퇴원율", size="학생수", text="캠퍼스",
                                     color="퇴원율", color_continuous_scale="YlOrRd",
                                     title="캠퍼스별 평균 TPI vs 9월 퇴원율")
                    fig.update_traces(textposition="top center")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # MT vs LT comparison
                st.markdown("---")
                st.markdown("**시험유형(MT/LT) 비교**")
                if "시험유형" in sl.columns:
                    for etype in ["MT", "LT"]:
                        sub = sl[sl["시험유형"].str.contains(etype, na=False)]
                        if len(sub) > 0:
                            churned_n = int(sub["target_churn_sep"].sum())
                            st.metric(f"{etype} - 퇴원율", f"{churned_n/len(sub)*100:.1f}% ({churned_n}/{len(sub)})")
            else:
                safe_warn("캠퍼스 데이터가 없습니다.")

        # ─── TAB 8: 상관분석 ────────────────────────────────────────────────
        with atabs[8]:
            st.subheader("상관분석")

            # Correlation heatmap
            corr_m = ar["corr_matrix"]
            if not corr_m.empty:
                fig = px.imshow(corr_m.values, x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                               title="8월 지표 간 Pearson 상관행렬")
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # Correlation pairs
            corr_p = ar["corr_pairs"]
            if not corr_p.empty:
                st.markdown("**지표 쌍별 상관계수**")
                st.dataframe(corr_p, use_container_width=True, hide_index=True)

            # Point-biserial with churn
            st.markdown("---")
            st.markdown("**target_churn_sep와의 포인트-바이시리얼 상관**")
            pb_all = ar["pb_all"]
            if not pb_all.empty:
                st.dataframe(pb_all, use_container_width=True, hide_index=True)

        # ─── TAB 9: 회귀분석 ────────────────────────────────────────────────
        with atabs[9]:
            st.subheader("회귀분석")
            st.caption("로지스틱 회귀 중심의 방향성 분석")

            # A. Univariate summary
            st.markdown("**A. 단변량 로지스틱 회귀 (8월 지표)**")
            if not ar["uni_aug"].empty:
                st.dataframe(ar["uni_aug"], use_container_width=True, hide_index=True)

            st.markdown("**단변량 로지스틱 회귀 (3월 지표)**")
            if not ar["uni_mar"].empty:
                st.dataframe(ar["uni_mar"], use_container_width=True, hide_index=True)

            st.markdown("**단변량 로지스틱 회귀 (변화량)**")
            if not ar["uni_delta"].empty:
                st.dataframe(ar["uni_delta"], use_container_width=True, hide_index=True)

            # B. Multivariate
            st.markdown("---")
            st.markdown("**B. 다변량 로지스틱 회귀 모델 비교**")
            multi = ar["multi_models"]
            if multi:
                perf_rows = []
                for m in multi:
                    perf_rows.append({
                        "모델": m["모델"],
                        "AUC": m["AUC"],
                        "Accuracy": m["Accuracy"],
                        "Precision": m["Precision"],
                        "Recall": m["Recall"],
                        "F1": m["F1"],
                        "유효표본수": m["유효표본수"],
                        "해석": m["해석"],
                    })
                perf_df = pd.DataFrame(perf_rows)
                st.dataframe(perf_df, use_container_width=True, hide_index=True)

                # AUC comparison chart
                valid_perf = perf_df.dropna(subset=["AUC"])
                if not valid_perf.empty:
                    fig = px.bar(valid_perf, x="모델", y="AUC", color="AUC",
                                 color_continuous_scale="Blues", title="모델별 AUC 비교")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # Show coefficients for each model
                for m in multi:
                    if m.get("coefs"):
                        with st.expander(f"{m['모델']} - 계수 상세"):
                            coef_rows = [{"변수": k, "계수": v["계수"], "오즈비": v["오즈비"]} for k, v in m["coefs"].items()]
                            st.dataframe(pd.DataFrame(coef_rows), use_container_width=True, hide_index=True)

        # ─── TAB 10: 생존분석 ───────────────────────────────────────────────
        with atabs[10]:
            st.subheader("생존분석")
            st.caption("재원기간을 보조적으로 활용한 생존분석 관점의 차이 탐색")

            surv = ar["survival"]
            if surv.get("error"):
                safe_warn(surv["error"])
            else:
                # Kaplan-Meier curves
                km_data = surv.get("km_data", {})
                if km_data:
                    st.markdown("**Kaplan-Meier 생존곡선 (TPI 그룹별)**")
                    fig = go.Figure()
                    colors = {"하위": CHURNED_COLOR, "중위": SKYBLUE2, "상위": RETAINED_COLOR}
                    for grp_name, sf in km_data.items():
                        fig.add_trace(go.Scatter(
                            x=sf.index, y=sf.iloc[:, 0],
                            name=grp_name, line=dict(color=colors.get(grp_name, BLUE1)),
                        ))
                    fig.update_layout(title="Kaplan-Meier 생존곡선", xaxis_title="재원기간(개월)", yaxis_title="생존확률", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                # Cox PH
                cox_table = surv.get("cox_table")
                if cox_table is not None and not cox_table.empty:
                    st.markdown("**Cox 비례위험 모델 결과**")
                    st.dataframe(cox_table, use_container_width=True, hide_index=True)
                    st.caption("* 위험비(HR) > 1: 해당 변수 증가 시 퇴원 위험 증가 방향, HR < 1: 퇴원 위험 감소 방향")

            st.markdown("---")
            st.caption("생존분석은 보조분석입니다. 해석은 '위험률 차이', '연결 강도', '통계적 방향성' 표현을 사용합니다.")

        # ─── TAB 11: 통합 통계 분석 ──────────────────────────────────────────
        with atabs[11]:
            st.subheader("통합 통계 분석")
            st.caption("개별 통계 탭에 흩어진 핵심 결과를 한 화면에서 종합")

            integrated = ar["integrated"]
            if not integrated.empty:
                st.dataframe(integrated, use_container_width=True, hide_index=True)
            else:
                safe_warn("통합 요약을 생성할 수 없습니다.")

            # Top univariate results
            st.markdown("---")
            st.markdown("**단변량 회귀 핵심 요약**")
            if not ar["uni_aug"].empty and "AUC" in ar["uni_aug"].columns:
                top5 = ar["uni_aug"].nlargest(5, "AUC", keep="first")
                st.dataframe(top5, use_container_width=True, hide_index=True)

            # Model performance comparison
            st.markdown("**다변량 모델 성능 비교**")
            if ar["multi_models"]:
                perf_rows = [{"모델": m["모델"], "AUC": m["AUC"], "F1": m["F1"], "유효표본수": m["유효표본수"]} for m in ar["multi_models"] if pd.notna(m.get("AUC"))]
                if perf_rows:
                    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

            # Tenure-specific patterns
            st.markdown("---")
            st.markdown("**재원기간 구간별 퇴원율 요약**")
            if "재원기간구간" in sl.columns:
                tenure_summary = []
                for label in TENURE_LABELS_INTERNAL:
                    sub = sl[sl["재원기간구간"] == label]
                    n = len(sub)
                    churned = int(sub["target_churn_sep"].sum()) if n > 0 else 0
                    rate = round(churned / n * 100, 1) if n > 0 else 0
                    avg_tpi = sub["TPI_8월"].mean() if "TPI_8월" in sub.columns and n > 0 else np.nan
                    tenure_summary.append({"재원기간구간": label, "학생수": n, "퇴원수": churned, "퇴원율": rate, "평균TPI": round(avg_tpi, 2) if pd.notna(avg_tpi) else ""})
                # Total
                total_n = len(sl)
                total_churned = int(sl["target_churn_sep"].sum())
                total_rate = round(total_churned / total_n * 100, 1) if total_n > 0 else 0
                total_tpi = sl["TPI_8월"].mean() if "TPI_8월" in sl.columns else np.nan
                tenure_summary.append({"재원기간구간": "전체", "학생수": total_n, "퇴원수": total_churned, "퇴원율": total_rate, "평균TPI": round(total_tpi, 2) if pd.notna(total_tpi) else ""})
                st.dataframe(pd.DataFrame(tenure_summary), use_container_width=True, hide_index=True)

            st.markdown("---")
            safe_info("통계적으로 유의하더라도 표본이 작거나 효과크기가 약한 경우 실무 적용에 주의가 필요합니다.")

        # ─── TAB 12: 결과 다운로드 ──────────────────────────────────────────
        with atabs[12]:
            st.subheader("결과 다운로드")
            st.caption("모든 다운로드 데이터는 코호트 기준 결과만 포함됩니다.")

            download_files = {}

            # 1. TPI config
            config_csv = ac_to_csv_bytes(config_df)
            download_files["01_TPI지수구성표.csv"] = config_csv
            st.download_button("1. TPI 지수 구성표 CSV", data=config_csv, file_name="TPI지수구성표.csv", mime="text/csv", key="dl_config")

            # 2. Student detail
            if not sl.empty:
                sl_csv = ac_to_csv_bytes(sl)
                download_files["02_학생별_상세_TPI_결과.csv"] = sl_csv
                st.download_button("2. 학생별 상세 TPI 결과 CSV", data=sl_csv, file_name="학생별_상세_TPI_결과.csv", mime="text/csv", key="dl_student_detail")

            # 3. Cohort summary
            cohort_sum = pd.DataFrame([meta])
            cs_csv = ac_to_csv_bytes(cohort_sum)
            download_files["03_코호트_요약.csv"] = cs_csv
            st.download_button("3. 코호트 요약 CSV", data=cs_csv, file_name="코호트_요약.csv", mime="text/csv", key="dl_cohort_sum")

            # 4. Comparison
            if not ar["comp_aug"].empty:
                comp_csv = ac_to_csv_bytes(ar["comp_aug"])
                download_files["04_유지군vs퇴원군_비교표.csv"] = comp_csv
                st.download_button("4. 유지군 vs 퇴원군 비교표 CSV", data=comp_csv, file_name="유지군vs퇴원군_비교표.csv", mime="text/csv", key="dl_comp")

            # 5. Cross tables
            for m_name, ct_data in ar["crosstabs"].items():
                ct_csv = ac_to_csv_bytes(ct_data)
                fname = f"05_분위수x재원기간_교차표_{m_name}.csv"
                download_files[fname] = ct_csv
            if ar["crosstabs"]:
                all_ct = pd.concat([ct.assign(지표명=m) for m, ct in ar["crosstabs"].items()], ignore_index=True)
                ct_all_csv = ac_to_csv_bytes(all_ct)
                st.download_button("5. 지표별 분위수×재원기간 교차표 CSV (통합)", data=ct_all_csv, file_name="분위수x재원기간_교차표_전체.csv", mime="text/csv", key="dl_ct")

            # 6. Monthly flow
            if ar["monthly_flows"]:
                all_mf = pd.concat([mf.assign(지표=m) for m, mf in ar["monthly_flows"].items()], ignore_index=True)
                mf_csv = ac_to_csv_bytes(all_mf)
                download_files["06_월별흐름분석.csv"] = mf_csv
                st.download_button("6. 월별 흐름 분석 CSV", data=mf_csv, file_name="월별흐름분석.csv", mime="text/csv", key="dl_mf")

            # 7. Risk students
            if not ar["risk_df"].empty:
                risk_csv = ac_to_csv_bytes(ar["risk_df"])
                download_files["07_위험학생_리스트.csv"] = risk_csv
                st.download_button("7. 위험학생 리스트 CSV", data=risk_csv, file_name="위험학생_리스트.csv", mime="text/csv", key="dl_risk")

            # 8. Campus summary
            if not ar["campus_df"].empty:
                camp_csv = ac_to_csv_bytes(ar["campus_df"])
                download_files["08_캠퍼스_요약.csv"] = camp_csv
                st.download_button("8. 캠퍼스 요약 CSV", data=camp_csv, file_name="캠퍼스_요약.csv", mime="text/csv", key="dl_campus")

            # 9. Correlation
            if not ar["corr_pairs"].empty:
                corr_csv = ac_to_csv_bytes(ar["corr_pairs"])
                download_files["09_상관분석_결과.csv"] = corr_csv
                st.download_button("9. 상관분석 결과 CSV", data=corr_csv, file_name="상관분석_결과.csv", mime="text/csv", key="dl_corr")

            # 10. Regression
            if not ar["uni_aug"].empty:
                reg_csv = ac_to_csv_bytes(ar["uni_aug"])
                download_files["10_회귀분석_단변량.csv"] = reg_csv
                st.download_button("10. 회귀 결과 CSV (단변량)", data=reg_csv, file_name="회귀분석_단변량.csv", mime="text/csv", key="dl_reg_uni")

            if ar["multi_models"]:
                perf_dl = pd.DataFrame([{k: v for k, v in m.items() if k != "coefs"} for m in ar["multi_models"]])
                multi_csv = ac_to_csv_bytes(perf_dl)
                download_files["10_회귀분석_다변량.csv"] = multi_csv
                st.download_button("10. 회귀 결과 CSV (다변량)", data=multi_csv, file_name="회귀분석_다변량.csv", mime="text/csv", key="dl_reg_multi")

            # 11. Survival
            surv = ar["survival"]
            if surv.get("cox_table") is not None and not surv["cox_table"].empty:
                surv_csv = ac_to_csv_bytes(surv["cox_table"])
                download_files["11_생존분석_결과.csv"] = surv_csv
                st.download_button("11. 생존분석 결과 CSV", data=surv_csv, file_name="생존분석_결과.csv", mime="text/csv", key="dl_surv")

            # 12. Integrated
            if not ar["integrated"].empty:
                int_csv = ac_to_csv_bytes(ar["integrated"])
                download_files["12_통합통계분석_요약.csv"] = int_csv
                st.download_button("12. 통합 통계 분석 요약 CSV", data=int_csv, file_name="통합통계분석_요약.csv", mime="text/csv", key="dl_integrated")

            # 13. ZIP package
            st.markdown("---")
            st.markdown("**전체 결과 패키지 다운로드**")
            if download_files:
                zip_bytes = build_zip_package(download_files)
                st.download_button(
                    "전체 결과 ZIP 다운로드",
                    data=zip_bytes,
                    file_name="PolyRetentionSignal_분석결과_전체.zip",
                    mime="application/zip",
                    key="dl_zip",
                    type="primary",
                )

"""
PolyRetentionSignal — Analysis Core
Cohort extraction, statistics, regression, survival analysis, risk scoring, downloads.
All statistical analysis functions are separated from UI rendering.
"""
from __future__ import annotations

import io
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Tenure bucket helpers ────────────────────────────────────────────────────

TENURE_BINS = [0, 6, 12, 18, 24, float("inf")]
TENURE_LABELS_INTERNAL = [
    "6개월 이하",
    "12개월 이하",
    "18개월 이하",
    "24개월 이하",
    "25개월 이상",
]

def assign_tenure_bucket(months: pd.Series) -> pd.Series:
    """Assign tenure bucket labels from continuous months."""
    return pd.cut(
        months,
        bins=TENURE_BINS,
        labels=TENURE_LABELS_INTERNAL,
        right=True,
        include_lowest=True,
    ).astype(str)

# ── Cohort extraction ────────────────────────────────────────────────────────

def get_valid_student_codes(student_info: pd.DataFrame) -> set:
    """Return the set of student_codes present in the student CSV."""
    if student_info is None or student_info.empty:
        return set()
    codes = student_info["student_code"].dropna()
    return set(codes.astype(int).tolist())


def extract_cohort(
    summary_df: pd.DataFrame,
    student_info: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract the analysis cohort:
    1. Students enrolled in March 2025 GT2
    2. Students enrolled in August 2025 GT2
    3. Only students present in student CSV
    4. Determine September churn status
    Returns (cohort_df, meta_dict).
    """
    valid_codes = get_valid_student_codes(student_info)
    if not valid_codes:
        return pd.DataFrame(), {"error": "학생 CSV에 유효한 학생코드가 없습니다."}

    df = summary_df.copy()
    df["학생코드_int"] = pd.to_numeric(df["학생코드"], errors="coerce")

    # Students present in student CSV
    df_valid = df[df["학생코드_int"].isin(valid_codes)].copy()

    if df_valid.empty:
        return pd.DataFrame(), {"error": "시험 데이터와 학생 CSV 간 일치하는 학생이 없습니다."}

    # Build per-student enrollment status by month from student_info
    # student_info has: student_code, enrollment_months, is_enrolled, etc.
    # We need month-level presence. We use exam data to see which months students appeared.
    # Cohort rule: present in March GT2 (exam_type in GT2 series), present in Aug GT2
    # Since exam_type is MT/LT, GT2 = the overall exam system.
    # "GT2 재원" means they are enrolled at that point. We check enrollment status from student data.
    # However student_info only gives a single snapshot. Let's check if they have exam records in March and Aug.

    # Step 1: Find students who have exam records in year=2025, month=3
    march_students = set(
        df_valid.loc[
            (df_valid["연도"].astype(int) == 2025) & (df_valid["월"].astype(int) == 3),
            "학생코드_int",
        ]
        .dropna()
        .unique()
    )

    # Step 2: Find students who have exam records in year=2025, month=8
    aug_students = set(
        df_valid.loc[
            (df_valid["연도"].astype(int) == 2025) & (df_valid["월"].astype(int) == 8),
            "학생코드_int",
        ]
        .dropna()
        .unique()
    )

    # Cohort: students in both March and August 2025
    cohort_codes = march_students & aug_students & valid_codes

    if not cohort_codes:
        return pd.DataFrame(), {
            "error": "2025년 3월과 8월 모두 시험 기록이 있는 학생이 없습니다. 데이터를 확인하세요.",
            "march_count": len(march_students),
            "aug_count": len(aug_students),
        }

    # Step 3: Determine September status
    # Check if student has exam records in Sep 2025 OR is_enrolled=1 in student_info
    sep_students = set(
        df_valid.loc[
            (df_valid["연도"].astype(int) == 2025) & (df_valid["월"].astype(int) == 9),
            "학생코드_int",
        ]
        .dropna()
        .unique()
    )

    # Use student_info enrollment status as primary source for Sep status
    # If student_info says is_enrolled=1, they are retained. If 0, they churned.
    # But we also check: if they have Sep exam records, they are retained.
    enrolled_map = dict(
        zip(
            student_info["student_code"].astype(int),
            student_info["is_enrolled"].astype(int),
        )
    )

    cohort_records = []
    for code in cohort_codes:
        # Get enrollment status from student_info
        is_enrolled = enrolled_map.get(int(code), None)
        # Also check if they appear in Sep exams
        in_sep = int(code) in sep_students

        if is_enrolled == 1 or in_sep:
            churn = 0  # retained
        elif is_enrolled == 0:
            churn = 1  # churned
        else:
            # Cannot determine - skip
            continue

        cohort_records.append({"학생코드_int": int(code), "target_churn_sep": churn})

    if not cohort_records:
        return pd.DataFrame(), {"error": "9월 상태를 판정할 수 있는 학생이 없습니다."}

    cohort_status = pd.DataFrame(cohort_records)

    # Build cohort dataframe with all exam records for these students
    cohort_all_records = df_valid[df_valid["학생코드_int"].isin(cohort_status["학생코드_int"])].copy()
    cohort_all_records = cohort_all_records.merge(cohort_status, on="학생코드_int", how="left")

    # Get tenure from student_info (Aug 2025 fixed)
    tenure_map = dict(
        zip(
            student_info["student_code"].astype(int),
            student_info["enrollment_months"],
        )
    )
    cohort_all_records["tenure_aug_fixed"] = cohort_all_records["학생코드_int"].map(tenure_map)
    cohort_all_records["재원기간구간"] = assign_tenure_bucket(
        pd.to_numeric(cohort_all_records["tenure_aug_fixed"], errors="coerce")
    )

    meta = {
        "total_valid_students": len(valid_codes),
        "march_students": len(march_students),
        "aug_students": len(aug_students),
        "cohort_size": len(cohort_status),
        "retained_count": int((cohort_status["target_churn_sep"] == 0).sum()),
        "churned_count": int((cohort_status["target_churn_sep"] == 1).sum()),
        "churn_rate": float(cohort_status["target_churn_sep"].mean() * 100),
    }

    return cohort_all_records, meta


def build_cohort_student_level(
    cohort_df: pd.DataFrame,
    student_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a student-level summary for the cohort with March values, Aug values,
    deltas, averages, variability.
    """
    if cohort_df.empty:
        return pd.DataFrame()

    metric_cols = ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR", "TPI"]
    available_metrics = [c for c in metric_cols if c in cohort_df.columns]

    id_cols = ["학생코드_int", "target_churn_sep", "tenure_aug_fixed", "재원기간구간"]
    # Also get student name and campus from first record
    extra_cols = ["캠퍼스", "학생명", "시험유형"]

    records = []
    for code, grp in cohort_df.groupby("학생코드_int"):
        rec = {
            "학생코드": int(code),
            "target_churn_sep": grp["target_churn_sep"].iloc[0],
            "tenure_aug_fixed": grp["tenure_aug_fixed"].iloc[0],
            "재원기간구간": grp["재원기간구간"].iloc[0],
            "캠퍼스": grp["캠퍼스"].iloc[0] if "캠퍼스" in grp.columns else "",
            "학생명": grp["학생명"].iloc[0] if "학생명" in grp.columns else "",
        }

        # Get exam types
        exam_types = grp["시험유형"].unique().tolist() if "시험유형" in grp.columns else []
        rec["시험유형"] = ", ".join(str(e) for e in exam_types)

        for m in available_metrics:
            # March values (year=2025, month=3)
            mar_vals = grp.loc[
                (grp["연도"].astype(int) == 2025) & (grp["월"].astype(int) == 3), m
            ].dropna()
            mar_val = mar_vals.mean() if len(mar_vals) > 0 else np.nan

            # August values
            aug_vals = grp.loc[
                (grp["연도"].astype(int) == 2025) & (grp["월"].astype(int) == 8), m
            ].dropna()
            aug_val = aug_vals.mean() if len(aug_vals) > 0 else np.nan

            rec[f"{m}_3월"] = mar_val
            rec[f"{m}_8월"] = aug_val

            # Delta and rate
            if pd.notna(mar_val) and pd.notna(aug_val):
                rec[f"{m}_변화량"] = aug_val - mar_val
                rec[f"{m}_변화율"] = ((aug_val - mar_val) / mar_val * 100) if mar_val != 0 else np.nan
            else:
                rec[f"{m}_변화량"] = np.nan
                rec[f"{m}_변화율"] = np.nan

            # 3~8 average and variability (all months between 3 and 8)
            range_vals = grp.loc[
                (grp["연도"].astype(int) == 2025)
                & (grp["월"].astype(int) >= 3)
                & (grp["월"].astype(int) <= 8),
                m,
            ].dropna()
            rec[f"{m}_3~8평균"] = range_vals.mean() if len(range_vals) > 0 else np.nan
            rec[f"{m}_3~8변동성"] = range_vals.std() if len(range_vals) > 1 else np.nan

            # Slope if 3+ observations
            if len(range_vals) >= 3:
                months_idx = grp.loc[range_vals.index, "월"].astype(int).values
                try:
                    slope = np.polyfit(months_idx, range_vals.values, 1)[0]
                    rec[f"{m}_추세기울기"] = slope
                except Exception:
                    rec[f"{m}_추세기울기"] = np.nan
            else:
                rec[f"{m}_추세기울기"] = np.nan

        records.append(rec)

    result = pd.DataFrame(records)
    # Round numeric columns
    for col in result.select_dtypes(include=[np.number]).columns:
        result[col] = result[col].round(2)
    return result


# ── Descriptive statistics & comparison ──────────────────────────────────────

def group_comparison(
    student_df: pd.DataFrame,
    metric_col: str,
    group_col: str = "target_churn_sep",
) -> Dict[str, Any]:
    """Compare retained vs churned for a single metric."""
    from scipy import stats as sp_stats

    retained = student_df.loc[student_df[group_col] == 0, metric_col].dropna()
    churned = student_df.loc[student_df[group_col] == 1, metric_col].dropna()

    result = {
        "지표": metric_col,
        "유지군_평균": retained.mean() if len(retained) > 0 else np.nan,
        "퇴원군_평균": churned.mean() if len(churned) > 0 else np.nan,
        "유지군_중앙값": retained.median() if len(retained) > 0 else np.nan,
        "퇴원군_중앙값": churned.median() if len(churned) > 0 else np.nan,
        "유지군_표준편차": retained.std() if len(retained) > 1 else np.nan,
        "퇴원군_표준편차": churned.std() if len(churned) > 1 else np.nan,
        "유지군_n": len(retained),
        "퇴원군_n": len(churned),
    }

    # Mean difference
    result["평균차"] = (result["퇴원군_평균"] or 0) - (result["유지군_평균"] or 0)
    result["중앙값차"] = (result["퇴원군_중앙값"] or 0) - (result["유지군_중앙값"] or 0)

    # Cohen's d
    if len(retained) > 1 and len(churned) > 1:
        pooled_std = np.sqrt(
            ((len(retained) - 1) * retained.std() ** 2 + (len(churned) - 1) * churned.std() ** 2)
            / (len(retained) + len(churned) - 2)
        )
        result["효과크기_d"] = (result["퇴원군_평균"] - result["유지군_평균"]) / pooled_std if pooled_std > 0 else np.nan
    else:
        result["효과크기_d"] = np.nan

    # Mann-Whitney U test (non-parametric)
    if len(retained) >= 3 and len(churned) >= 3:
        try:
            stat, pval = sp_stats.mannwhitneyu(retained, churned, alternative="two-sided")
            result["p_value"] = pval
        except Exception:
            result["p_value"] = np.nan
    else:
        result["p_value"] = np.nan

    # Interpretation
    if pd.notna(result["p_value"]) and result["p_value"] < 0.05:
        direction = "퇴원군이 낮음" if result["평균차"] < 0 else "퇴원군이 높음"
        result["해석"] = f"통계적으로 유의한 차이 관찰 (p={result['p_value']:.4f}), {direction}"
    elif pd.notna(result["p_value"]):
        result["해석"] = f"통계적으로 유의하지 않음 (p={result['p_value']:.4f})"
    else:
        result["해석"] = "표본 부족으로 검정 불가"

    return result


def build_comparison_table(
    student_df: pd.DataFrame,
    metrics: List[str],
    suffix: str = "",
) -> pd.DataFrame:
    """Build a comparison table for multiple metrics."""
    rows = []
    for m in metrics:
        col_name = f"{m}{suffix}" if suffix else m
        if col_name in student_df.columns:
            rows.append(group_comparison(student_df, col_name))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].round(4)
    return df


# ── Point-biserial correlation ───────────────────────────────────────────────

def point_biserial_table(
    student_df: pd.DataFrame,
    metrics: List[str],
    target: str = "target_churn_sep",
) -> pd.DataFrame:
    from scipy.stats import pointbiserialr

    rows = []
    for m in metrics:
        if m not in student_df.columns:
            continue
        valid = student_df[[m, target]].dropna()
        if len(valid) < 5:
            rows.append({"지표": m, "상관계수": np.nan, "p_value": np.nan, "유효표본수": len(valid), "해석": "표본 부족"})
            continue
        try:
            r, p = pointbiserialr(valid[target], valid[m])
            direction = "양의 상관 (지표 높을수록 퇴원 경향)" if r > 0 else "음의 상관 (지표 낮을수록 퇴원 경향)"
            sig = f"유의 (p={p:.4f})" if p < 0.05 else f"비유의 (p={p:.4f})"
            rows.append({"지표": m, "상관계수": round(r, 4), "p_value": round(p, 4), "유효표본수": len(valid), "해석": f"{direction}, {sig}"})
        except Exception:
            rows.append({"지표": m, "상관계수": np.nan, "p_value": np.nan, "유효표본수": len(valid), "해석": "계산 오류"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Quantile × Tenure cross table ────────────────────────────────────────────

def quantile_tenure_crosstab(
    student_df: pd.DataFrame,
    metric_col: str,
    n_quantiles: int = 5,
    target: str = "target_churn_sep",
    tenure_col: str = "재원기간구간",
) -> pd.DataFrame:
    """Build quantile × tenure cross-table with n, churn_n, churn_rate."""
    df = student_df[[metric_col, target, tenure_col]].dropna(subset=[metric_col]).copy()
    if df.empty:
        return pd.DataFrame()

    try:
        df["분위"] = pd.qcut(df[metric_col], q=n_quantiles, labels=[f"분위{i+1}" for i in range(n_quantiles)], duplicates="drop")
    except ValueError:
        # If not enough unique values for qcut
        df["분위"] = pd.cut(df[metric_col], bins=n_quantiles, labels=[f"분위{i+1}" for i in range(n_quantiles)], duplicates="drop")

    rows = []
    for q_label in df["분위"].cat.categories if hasattr(df["분위"], "cat") else df["분위"].unique():
        q_data = df[df["분위"] == q_label]
        row = {"분위": q_label}
        for tenure_label in TENURE_LABELS_INTERNAL + ["전체"]:
            if tenure_label == "전체":
                subset = q_data
            else:
                subset = q_data[q_data[tenure_col] == tenure_label]
            n = len(subset)
            churn_n = int(subset[target].sum()) if n > 0 else 0
            rate = round(churn_n / n * 100, 1) if n > 0 else 0
            row[f"{tenure_label}_n"] = n
            row[f"{tenure_label}_퇴원"] = churn_n
            row[f"{tenure_label}_퇴원율"] = rate
        rows.append(row)

    # Add total row
    total_row = {"분위": "전체"}
    for tenure_label in TENURE_LABELS_INTERNAL + ["전체"]:
        if tenure_label == "전체":
            subset = df
        else:
            subset = df[df[tenure_col] == tenure_label]
        n = len(subset)
        churn_n = int(subset[target].sum()) if n > 0 else 0
        rate = round(churn_n / n * 100, 1) if n > 0 else 0
        total_row[f"{tenure_label}_n"] = n
        total_row[f"{tenure_label}_퇴원"] = churn_n
        total_row[f"{tenure_label}_퇴원율"] = rate
    rows.append(total_row)

    return pd.DataFrame(rows)


# ── Univariate logistic regression ───────────────────────────────────────────

def univariate_logistic(
    student_df: pd.DataFrame,
    metric_col: str,
    target: str = "target_churn_sep",
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    valid = student_df[[metric_col, target]].dropna()
    if len(valid) < 10 or valid[target].nunique() < 2:
        return {"지표": metric_col, "계수": np.nan, "오즈비": np.nan, "p_value": np.nan, "AUC": np.nan, "표본수": len(valid), "해석": "표본 또는 클래스 부족"}

    X = valid[[metric_col]].values
    y = valid[target].values.astype(int)

    try:
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(X, y)
        coef = model.coef_[0][0]
        odds_ratio = np.exp(coef)
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)

        # Approximate p-value via Wald test
        from scipy.stats import norm
        pred_probs = model.predict_proba(X)[:, 1]
        W = np.diag(pred_probs * (1 - pred_probs))
        try:
            cov = np.linalg.inv(X.T @ W @ X)
            se = np.sqrt(cov[0, 0])
            z = coef / se
            p_val = 2 * (1 - norm.cdf(abs(z)))
        except Exception:
            p_val = np.nan

        direction = "지표 높을수록 퇴원 경향" if coef > 0 else "지표 낮을수록 퇴원 경향"
        return {
            "지표": metric_col,
            "계수": round(coef, 4),
            "오즈비": round(odds_ratio, 4),
            "p_value": round(p_val, 4) if pd.notna(p_val) else np.nan,
            "AUC": round(auc, 4),
            "표본수": len(valid),
            "방향": direction,
            "해석": f"OR={odds_ratio:.2f}, AUC={auc:.3f}, {direction}",
        }
    except Exception as e:
        return {"지표": metric_col, "계수": np.nan, "오즈비": np.nan, "p_value": np.nan, "AUC": np.nan, "표본수": len(valid), "해석": f"계산 오류: {e}"}


def univariate_logistic_table(
    student_df: pd.DataFrame,
    metrics: List[str],
    target: str = "target_churn_sep",
) -> pd.DataFrame:
    rows = [univariate_logistic(student_df, m, target) for m in metrics if m in student_df.columns]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Multivariate logistic regression ─────────────────────────────────────────

def multivariate_logistic(
    student_df: pd.DataFrame,
    feature_cols: List[str],
    target: str = "target_churn_sep",
    model_name: str = "",
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    available = [c for c in feature_cols if c in student_df.columns]
    valid = student_df[available + [target]].dropna()
    if len(valid) < 20 or valid[target].nunique() < 2:
        return {"모델": model_name, "AUC": np.nan, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "유효표본수": len(valid), "해석": "표본 부족", "coefs": {}}

    X = valid[available].values
    y = valid[target].values.astype(int)

    try:
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        coefs = {}
        for i, col in enumerate(available):
            coefs[col] = {
                "계수": round(model.coef_[0][i], 4),
                "오즈비": round(np.exp(model.coef_[0][i]), 4),
            }

        return {
            "모델": model_name,
            "AUC": round(auc, 4),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "유효표본수": len(valid),
            "해석": f"AUC={auc:.3f}, 정밀도={prec:.3f}, 재현율={rec:.3f}",
            "coefs": coefs,
        }
    except Exception as e:
        return {"모델": model_name, "AUC": np.nan, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "유효표본수": len(valid), "해석": f"오류: {e}", "coefs": {}}


def run_all_multivariate_models(student_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run all defined multivariate models."""
    base_metrics_8 = ["TPI_8월"]
    metrics_8 = [f"{m}_8월" for m in ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR"]]
    metrics_3 = [f"{m}_3월" for m in ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR"]]
    metrics_delta = [f"{m}_변화량" for m in ["P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR"]]

    models = []
    # Model A: TPI only
    models.append(multivariate_logistic(student_df, ["TPI_8월"], model_name="Model A: TPI"))
    # Model B: TPI + tenure
    models.append(multivariate_logistic(student_df, ["TPI_8월", "tenure_aug_fixed"], model_name="Model B: TPI + 재원기간"))
    # Model C: TPI + tenure + exam type dummy
    if "시험유형" in student_df.columns:
        df_tmp = student_df.copy()
        df_tmp["시험유형_MT"] = (df_tmp["시험유형"].str.contains("MT", na=False)).astype(int)
        models.append(multivariate_logistic(df_tmp, ["TPI_8월", "tenure_aug_fixed", "시험유형_MT"], model_name="Model C: TPI + 재원기간 + 시험유형"))
    # Model D: All individual metrics + tenure
    models.append(multivariate_logistic(student_df, metrics_8 + ["tenure_aug_fixed"], model_name="Model D: 전체 8월 지표 + 재원기간"))
    # Model: 3월 only
    models.append(multivariate_logistic(student_df, metrics_3, model_name="3월 지표 모델"))
    # Model: 8월 only
    models.append(multivariate_logistic(student_df, metrics_8, model_name="8월 지표 모델"))
    # Model: 변화량 only
    models.append(multivariate_logistic(student_df, metrics_delta, model_name="3→8 변화량 모델"))
    # Model: 8월 + 변화량
    models.append(multivariate_logistic(student_df, metrics_8 + metrics_delta, model_name="8월 수준 + 변화량 결합 모델"))

    return models


# ── Correlation matrix ───────────────────────────────────────────────────────

def correlation_matrix(
    student_df: pd.DataFrame,
    metric_cols: List[str],
) -> pd.DataFrame:
    available = [c for c in metric_cols if c in student_df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return student_df[available].corr(method="pearson").round(4)


def correlation_pairs_table(
    student_df: pd.DataFrame,
    metric_cols: List[str],
) -> pd.DataFrame:
    from scipy.stats import pearsonr
    available = [c for c in metric_cols if c in student_df.columns]
    rows = []
    for i, c1 in enumerate(available):
        for c2 in available[i + 1:]:
            valid = student_df[[c1, c2]].dropna()
            if len(valid) < 5:
                rows.append({"변수1": c1, "변수2": c2, "상관계수": np.nan, "p_value": np.nan, "유효표본수": len(valid), "해석": "표본 부족"})
                continue
            try:
                r, p = pearsonr(valid[c1], valid[c2])
                strength = "강한" if abs(r) > 0.5 else ("중간" if abs(r) > 0.3 else "약한")
                rows.append({"변수1": c1, "변수2": c2, "상관계수": round(r, 4), "p_value": round(p, 4), "유효표본수": len(valid), "해석": f"{strength} 상관 (r={r:.3f})"})
            except Exception:
                rows.append({"변수1": c1, "변수2": c2, "상관계수": np.nan, "p_value": np.nan, "유효표본수": len(valid), "해석": "계산 오류"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Survival analysis ────────────────────────────────────────────────────────

def run_survival_analysis(
    student_df: pd.DataFrame,
    duration_col: str = "tenure_aug_fixed",
    event_col: str = "target_churn_sep",
) -> Dict[str, Any]:
    """Run Kaplan-Meier and Cox PH if lifelines available."""
    result = {"km_data": None, "cox_summary": None, "cox_table": None, "error": None}

    valid = student_df[[duration_col, event_col]].dropna()
    valid = valid[valid[duration_col] > 0]
    if len(valid) < 10 or valid[event_col].nunique() < 2:
        result["error"] = "생존분석에 충분한 표본이 없습니다."
        return result

    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter

        # KM by TPI group
        km_data = {}
        if "TPI_8월" in student_df.columns:
            df_km = student_df[[duration_col, event_col, "TPI_8월"]].dropna()
            df_km = df_km[df_km[duration_col] > 0]
            if len(df_km) >= 10:
                try:
                    df_km["TPI그룹"] = pd.qcut(df_km["TPI_8월"], q=3, labels=["하위", "중위", "상위"], duplicates="drop")
                except ValueError:
                    df_km["TPI그룹"] = pd.cut(df_km["TPI_8월"], bins=3, labels=["하위", "중위", "상위"])

                for grp_name in ["하위", "중위", "상위"]:
                    sub = df_km[df_km["TPI그룹"] == grp_name]
                    if len(sub) >= 3:
                        kmf = KaplanMeierFitter()
                        kmf.fit(sub[duration_col], event_observed=sub[event_col], label=grp_name)
                        km_data[grp_name] = kmf.survival_function_

        result["km_data"] = km_data

        # Cox PH
        cox_features = [duration_col, event_col]
        potential = ["TPI_8월", "P-Score_8월", "CI_8월", "QR_8월", "B.CV_8월"]
        cox_cols = [c for c in potential if c in student_df.columns]
        df_cox = student_df[cox_cols + [duration_col, event_col]].dropna()
        df_cox = df_cox[df_cox[duration_col] > 0]

        if len(df_cox) >= 20 and len(cox_cols) > 0:
            cph = CoxPHFitter()
            cph.fit(df_cox, duration_col=duration_col, event_col=event_col)
            summary = cph.summary
            result["cox_summary"] = summary

            cox_rows = []
            for var in summary.index:
                cox_rows.append({
                    "변수": var,
                    "계수": round(summary.loc[var, "coef"], 4),
                    "위험비(HR)": round(summary.loc[var, "exp(coef)"], 4),
                    "p_value": round(summary.loc[var, "p"], 4),
                    "방향": "위험 증가" if summary.loc[var, "coef"] > 0 else "위험 감소",
                    "해석": f"HR={summary.loc[var, 'exp(coef)']:.3f}",
                })
            result["cox_table"] = pd.DataFrame(cox_rows)
    except ImportError:
        result["error"] = "lifelines 패키지가 설치되어 있지 않습니다."
    except Exception as e:
        result["error"] = f"생존분석 오류: {e}"

    return result


# ── Risk scoring ─────────────────────────────────────────────────────────────

def compute_risk_scores(student_df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk scores for each cohort student."""
    df = student_df.copy()
    df["위험점수"] = 0
    df["위험사유"] = ""

    def add_risk(mask, points, reason):
        df.loc[mask, "위험점수"] += points
        df.loc[mask, "위험사유"] += reason + "; "

    # TPI 8월 하위 20%
    if "TPI_8월" in df.columns:
        threshold = df["TPI_8월"].quantile(0.2)
        if pd.notna(threshold):
            add_risk(df["TPI_8월"] <= threshold, 3, "8월 TPI 하위20%")

    # CI 8월 하위 20%
    if "CI_8월" in df.columns:
        threshold = df["CI_8월"].quantile(0.2)
        if pd.notna(threshold):
            add_risk(df["CI_8월"] <= threshold, 2, "8월 CI 하위20%")

    # QR 8월 하위 20%
    if "QR_8월" in df.columns:
        threshold = df["QR_8월"].quantile(0.2)
        if pd.notna(threshold):
            add_risk(df["QR_8월"] <= threshold, 2, "8월 QR 하위20%")

    # BCV 8월 하위 20%
    if "B.CV_8월" in df.columns:
        threshold = df["B.CV_8월"].quantile(0.2)
        if pd.notna(threshold):
            add_risk(df["B.CV_8월"] <= threshold, 1, "8월 BCV 하위20%")

    # TPI 3→8 하락
    if "TPI_변화량" in df.columns:
        add_risk(df["TPI_변화량"] < 0, 2, "TPI 3→8 하락")

    # CI 3→8 하락
    if "CI_변화량" in df.columns:
        add_risk(df["CI_변화량"] < 0, 1, "CI 3→8 하락")

    # QR 3→8 하락
    if "QR_변화량" in df.columns:
        add_risk(df["QR_변화량"] < 0, 1, "QR 3→8 하락")

    # Short tenure + low performance
    if "tenure_aug_fixed" in df.columns and "TPI_8월" in df.columns:
        short_tenure = df["tenure_aug_fixed"] <= 6
        tpi_threshold = df["TPI_8월"].quantile(0.3) if pd.notna(df["TPI_8월"].quantile(0.3)) else 50
        low_perf = df["TPI_8월"] <= tpi_threshold
        add_risk(short_tenure & low_perf, 1, "짧은 재원기간+저성과")

    # Risk level
    df["위험도"] = "Low"
    df.loc[df["위험점수"] >= 4, "위험도"] = "Medium"
    df.loc[df["위험점수"] >= 7, "위험도"] = "High"

    # Clean up reason string
    df["위험사유"] = df["위험사유"].str.rstrip("; ")

    return df


# ── Monthly flow analysis ────────────────────────────────────────────────────

def build_monthly_flow(
    cohort_df: pd.DataFrame,
    metric: str = "TPI",
) -> pd.DataFrame:
    """Build monthly average for a metric, split by churn group."""
    if metric not in cohort_df.columns:
        return pd.DataFrame()

    df = cohort_df[
        (cohort_df["연도"].astype(int) == 2025)
        & (cohort_df["월"].astype(int) >= 3)
        & (cohort_df["월"].astype(int) <= 8)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    rows = []
    for month in range(3, 9):
        month_data = df[df["월"].astype(int) == month]
        if month_data.empty:
            continue

        all_vals = month_data[metric].dropna()
        retained = month_data.loc[month_data["target_churn_sep"] == 0, metric].dropna()
        churned = month_data.loc[month_data["target_churn_sep"] == 1, metric].dropna()

        rows.append({
            "월": month,
            "전체_평균": round(all_vals.mean(), 2) if len(all_vals) > 0 else np.nan,
            "유지군_평균": round(retained.mean(), 2) if len(retained) > 0 else np.nan,
            "퇴원군_평균": round(churned.mean(), 2) if len(churned) > 0 else np.nan,
            "전체_n": len(all_vals),
            "유지군_n": len(retained),
            "퇴원군_n": len(churned),
        })

    return pd.DataFrame(rows)


# ── Campus breakdown ─────────────────────────────────────────────────────────

def campus_summary(student_df: pd.DataFrame) -> pd.DataFrame:
    """Campus-level summary."""
    if "캠퍼스" not in student_df.columns:
        return pd.DataFrame()

    rows = []
    for campus, grp in student_df.groupby("캠퍼스"):
        n = len(grp)
        churned = int(grp["target_churn_sep"].sum())
        retained = n - churned
        tpi_mean = grp["TPI_8월"].mean() if "TPI_8월" in grp.columns else np.nan
        risk_high = int((grp["위험도"] == "High").sum()) if "위험도" in grp.columns else 0
        risk_medium = int((grp["위험도"] == "Medium").sum()) if "위험도" in grp.columns else 0

        # Bottom 20% ratio
        if "TPI_8월" in grp.columns and "TPI_8월" in student_df.columns:
            threshold = student_df["TPI_8월"].quantile(0.2)
            bottom_pct = (grp["TPI_8월"] <= threshold).mean() * 100 if pd.notna(threshold) else np.nan
        else:
            bottom_pct = np.nan

        rows.append({
            "캠퍼스": campus,
            "학생수": n,
            "유지": retained,
            "퇴원": churned,
            "퇴원율": round(churned / n * 100, 1) if n > 0 else 0,
            "평균TPI": round(tpi_mean, 2) if pd.notna(tpi_mean) else np.nan,
            "위험(High)": risk_high,
            "위험(Medium)": risk_medium,
            "하위20%비중": round(bottom_pct, 1) if pd.notna(bottom_pct) else np.nan,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("퇴원율", ascending=False).reset_index(drop=True)
    return result


# ── Integrated statistics summary ────────────────────────────────────────────

def build_integrated_summary(
    pb_table: pd.DataFrame,
    uni_table: pd.DataFrame,
    model_results: List[Dict],
    comparison_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single integrated summary table."""
    rows = []

    # Top correlations
    if not pb_table.empty:
        top_corr = pb_table.nlargest(5, "상관계수", keep="first") if "상관계수" in pb_table.columns else pb_table.head(5)
        for _, r in top_corr.iterrows():
            rows.append({
                "분석영역": "상관분석",
                "지표/모델": r.get("지표", ""),
                "핵심수치": f"r={r.get('상관계수', '')}" if pd.notna(r.get("상관계수")) else "",
                "표본수": r.get("유효표본수", ""),
                "방향성요약": r.get("해석", ""),
                "실무해석": "",
            })

    # Top univariate
    if not uni_table.empty and "AUC" in uni_table.columns:
        top_uni = uni_table.nlargest(5, "AUC", keep="first")
        for _, r in top_uni.iterrows():
            rows.append({
                "분석영역": "단변량 회귀",
                "지표/모델": r.get("지표", ""),
                "핵심수치": f"AUC={r.get('AUC', '')}, OR={r.get('오즈비', '')}",
                "표본수": r.get("표본수", ""),
                "방향성요약": r.get("방향", ""),
                "실무해석": r.get("해석", ""),
            })

    # Multivariate models
    for m in model_results:
        if m.get("AUC") and pd.notna(m.get("AUC")):
            rows.append({
                "분석영역": "다변량 회귀",
                "지표/모델": m.get("모델", ""),
                "핵심수치": f"AUC={m.get('AUC', '')}, F1={m.get('F1', '')}",
                "표본수": m.get("유효표본수", ""),
                "방향성요약": m.get("해석", ""),
                "실무해석": "",
            })

    # Top effect size differences
    if not comparison_table.empty and "효과크기_d" in comparison_table.columns:
        top_effect = comparison_table.reindex(comparison_table["효과크기_d"].abs().sort_values(ascending=False).index).head(5)
        for _, r in top_effect.iterrows():
            rows.append({
                "분석영역": "효과크기",
                "지표/모델": r.get("지표", ""),
                "핵심수치": f"d={r.get('효과크기_d', '')}",
                "표본수": f"유지={r.get('유지군_n', '')}, 퇴원={r.get('퇴원군_n', '')}",
                "방향성요약": r.get("해석", ""),
                "실무해석": "",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Download helpers ─────────────────────────────────────────────────────────

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def build_zip_package(file_dict: Dict[str, bytes]) -> bytes:
    """Create a ZIP file from a dict of {filename: csv_bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in file_dict.items():
            zf.writestr(fname, data)
    return buf.getvalue()

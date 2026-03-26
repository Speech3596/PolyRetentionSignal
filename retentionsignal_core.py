from __future__ import annotations

import ast
import io
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

MONTH_MAP = {
    "jan": 1, "january": 1, "1": 1, "1월": 1,
    "feb": 2, "february": 2, "2": 2, "2월": 2,
    "mar": 3, "march": 3, "3": 3, "3월": 3,
    "apr": 4, "april": 4, "4": 4, "4월": 4,
    "may": 5, "5": 5, "5월": 5,
    "jun": 6, "june": 6, "6": 6, "6월": 6,
    "jul": 7, "july": 7, "7": 7, "7월": 7,
    "aug": 8, "august": 8, "8": 8, "8월": 8,
    "sep": 9, "september": 9, "9": 9, "9월": 9,
    "oct": 10, "october": 10, "10": 10, "10월": 10,
    "nov": 11, "november": 11, "11": 11, "11월": 11,
    "dec": 12, "december": 12, "12": 12, "12월": 12,
}

METRIC_ORDER = ["TPI", "P-Score", "T-Score", "B.CV", "CI", "QR", "C.T-Score", "C.CV", "C.QR"]
ALIAS_TO_COLUMN = {
    "P": "P-Score",
    "T": "T-Score",
    "BCV": "B.CV",
    "CI": "CI",
    "QR": "QR",
    "CT": "C.T-Score",
    "CCV": "C.CV",
    "CQR": "C.QR",
    "CV": "CV",
}

REQUIRED_EXAM_COLUMNS = [
    "curriculum", "campus_type", "campus", "class_name", "student_code", "student_name",
    "exam_type", "year", "semester", "month", "subject", "item_no", "correct_answer", "student_answer",
]


def month_to_num(value) -> Optional[int]:
    if pd.isna(value):
        return None
    return MONTH_MAP.get(str(value).strip().lower())


def clip_0_100(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0, upper=100)


def inverse_cv_score(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(vals) <= 1:
        return 100.0
    mean = vals.mean()
    if mean == 0 or np.isnan(mean):
        return 0.0
    raw_cv = (vals.std(ddof=0) / mean) * 100
    return float(np.clip(100 - raw_cv, 0, 100))


def tscore_from_series(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    mean = vals.mean()
    std = vals.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.full(len(vals), 50.0), index=series.index)
    z = (vals - mean) / std
    return (50 + 10 * z).clip(lower=0, upper=100)


def percentile_rank(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return (vals.rank(method="average", pct=True) * 100).astype(float)


def normalize_exam_type(v) -> str:
    s = str(v).strip().upper()
    if "MT" in s:
        return "MT"
    if "LT" in s:
        return "LT"
    return s


def detect_file_kind(name: str, expected_ext: str | None = None) -> str:
    s = Path(name).name.lower()
    ext = Path(s).suffix.lower()
    if expected_ext is not None and ext != expected_ext.lower():
        return "unknown"

    exam_markers = ["mt", "lt", "문항결과", "시험", "score", "exam"]
    student_markers = ["student", "학생", "master", "roster"]

    if ext == ".xlsx" and any(marker in s for marker in exam_markers):
        return "exam"
    if ext == ".csv" and any(marker in s for marker in student_markers):
        return "student"
    return "unknown"


def parse_exam_filename(name: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    year_m = re.search(r"(20\d{2})년", name)
    month_m = re.search(r"(\d{1,2})월", name)
    exam_m = re.search(r"(?:^|[_\-\s])(MT|LT)(?:$|[_\-\s])", name, re.IGNORECASE)
    year = int(year_m.group(1)) if year_m else None
    month = int(month_m.group(1)) if month_m else None
    exam_type = exam_m.group(1).upper() if exam_m else None
    return year, month, exam_type


def _read_excel_bytes(file_obj):
    if hasattr(file_obj, "getvalue"):
        return io.BytesIO(file_obj.getvalue())
    return file_obj


def read_single_exam(file_obj) -> pd.DataFrame:
    fname = getattr(file_obj, "name", None) or str(file_obj)
    if detect_file_kind(fname, expected_ext=".xlsx") != "exam":
        raise ValueError(f"시험 데이터 파일명이 아님: {Path(fname).name}")

    # Reset file pointer in case it was previously read
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    xls = pd.ExcelFile(_read_excel_bytes(file_obj))
    frames = []
    for sheet in xls.sheet_names:
        raw = xls.parse(sheet_name=sheet, header=None)
        if raw.shape[0] < 4:
            continue
        header = raw.iloc[2].tolist()
        df = raw.iloc[3:].copy()
        df.columns = header
        df = df.rename(columns={
            "교육과정": "curriculum",
            "운영구분": "campus_type",
            "캠퍼스": "campus",
            "학급": "class_name",
            "학번": "student_code",
            "이름": "student_name",
            "구분": "exam_type",
            "Year": "year",
            "Semester": "semester",
            "Month": "month",
            "시험과목": "subject",
            "문항 순번": "item_no",
            "문항정답": "correct_answer",
            "학생선택": "student_answer",
        })
        missing = [c for c in REQUIRED_EXAM_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{fname} / {sheet} 필수 컬럼 누락: {missing}")
        df = df[REQUIRED_EXAM_COLUMNS].copy()
        df["student_code"] = pd.to_numeric(df["student_code"], errors="coerce").astype("Int64")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["item_no"] = pd.to_numeric(df["item_no"], errors="coerce").astype("Int64")
        df["month_num"] = df["month"].map(month_to_num)
        y0, m0, e0 = parse_exam_filename(Path(fname).name)
        if df["year"].isna().all() and y0:
            df["year"] = y0
        if df["month_num"].isna().all() and m0:
            df["month_num"] = m0
        df["exam_type"] = df["exam_type"].fillna(e0).map(normalize_exam_type)
        df["is_correct"] = (
            df["student_answer"].astype(str).str.strip().str.lower()
            == df["correct_answer"].astype(str).str.strip().str.lower()
        ).astype(int)
        frames.append(df)
    if not frames:
        raise ValueError(f"유효한 시트가 없음: {fname}")
    return pd.concat(frames, ignore_index=True)


def read_student_info(file_obj) -> pd.DataFrame:
    fname = getattr(file_obj, "name", None) or str(file_obj)
    if detect_file_kind(fname, expected_ext=".csv") != "student":
        raise ValueError(f"학생 데이터 파일명이 아님: {Path(fname).name}")
    # Reset file pointer in case it was previously read
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    df = pd.read_csv(file_obj)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 6:
        raise ValueError("학생 데이터 컬럼 수가 부족하다.")

    right_second = df.columns[-2]
    right_last = df.columns[-1]
    rename_map = {
        df.columns[0]: "master_campus_type",
        df.columns[1]: "master_campus",
        df.columns[2]: "student_code",
        df.columns[3]: "master_student_name",
        right_second: "enrollment_months",
        right_last: "is_enrolled",
    }
    out = df.rename(columns=rename_map).copy()
    out = out[[
        "master_campus_type",
        "master_campus",
        "student_code",
        "master_student_name",
        "enrollment_months",
        "is_enrolled",
    ]]
    out["student_code"] = pd.to_numeric(out["student_code"], errors="coerce").astype("Int64")
    out["enrollment_months"] = pd.to_numeric(out["enrollment_months"], errors="coerce")
    out["is_enrolled"] = pd.to_numeric(out["is_enrolled"], errors="coerce").fillna(0).astype(int)
    return out.drop_duplicates(subset=["student_code"])


def build_item_stats(raw: pd.DataFrame) -> pd.DataFrame:
    grp = ["exam_type", "year", "month_num", "subject", "item_no"]
    item = raw.groupby(grp, dropna=False).agg(응시자수=("student_code", "nunique"), 정답자수=("is_correct", "sum")).reset_index()
    item["문항 정답률"] = item["정답자수"] / item["응시자수"] * 100
    return item


def build_student_summary(raw: pd.DataFrame, student_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    grp_exam = ["exam_type", "year", "month_num"]
    grp_student = grp_exam + ["campus", "campus_type", "student_code", "student_name"]
    item = build_item_stats(raw)

    subj = raw.groupby(grp_student + ["subject"], dropna=False).agg(correct=("is_correct", "sum"), total=("is_correct", "count")).reset_index()
    subj["subject_score"] = subj["correct"] / subj["total"] * 100
    subj_wide = subj.pivot_table(index=grp_student, columns="subject", values="subject_score", aggfunc="first").reset_index()
    subj_wide.columns.name = None

    overall = raw.groupby(grp_student, dropna=False).agg(total_correct=("is_correct", "sum"), total_items=("is_correct", "count")).reset_index()
    overall["P-Score"] = overall["total_correct"] / overall["total_items"] * 100

    score = overall.merge(subj_wide, on=grp_student, how="left")
    subject_cols = [c for c in score.columns if c not in grp_student + ["total_correct", "total_items", "P-Score"]]
    score["B.CV"] = score[subject_cols].apply(inverse_cv_score, axis=1) if subject_cols else 100.0
    exam_cv = score.groupby(grp_exam, dropna=False)["P-Score"].apply(inverse_cv_score).rename("CV").reset_index()
    score = score.merge(exam_cv, on=grp_exam, how="left")
    score["T-Score"] = score.groupby(grp_exam, dropna=False)["P-Score"].transform(tscore_from_series)
    score["QR"] = score.groupby(grp_exam, dropna=False)["P-Score"].transform(percentile_rank)

    campus_grp = grp_exam + ["campus"]
    campus_cv = score.groupby(campus_grp, dropna=False)["P-Score"].apply(inverse_cv_score).rename("C.CV").reset_index()
    score = score.merge(campus_cv, on=campus_grp, how="left")
    score["C.T-Score"] = score.groupby(campus_grp, dropna=False)["P-Score"].transform(tscore_from_series)
    score["C.QR"] = score.groupby(campus_grp, dropna=False)["P-Score"].transform(percentile_rank)

    item_exam = item.groupby(grp_exam, dropna=False).agg(exam_mean_rate=("문항 정답률", "mean")).reset_index()
    student_item = raw.merge(item[grp_exam + ["subject", "item_no", "문항 정답률"]], on=grp_exam + ["subject", "item_no"], how="left")
    actual = student_item[student_item["is_correct"] == 1].groupby(grp_student, dropna=False).agg(actual_sum=("문항 정답률", "sum"), k=("is_correct", "sum")).reset_index()
    actual = overall[grp_student + ["total_correct"]].merge(actual, on=grp_student, how="left")
    actual["actual_sum"] = actual["actual_sum"].fillna(0)
    actual["k"] = actual["k"].fillna(0)

    ideal_parts = []
    for keys, sub in item.groupby(grp_exam, dropna=False):
        k_max = int(sub.shape[0])
        rates = np.sort(sub["문항 정답률"].astype(float).values)[::-1]
        csum = np.cumsum(rates)
        tmp = pd.DataFrame({"k": np.arange(0, k_max + 1, dtype=int), "ideal_sum": np.concatenate([[0.0], csum])})
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(grp_exam, keys):
            tmp[col] = val
        ideal_parts.append(tmp)
    ideal = pd.concat(ideal_parts, ignore_index=True)

    ci = actual.merge(item_exam, on=grp_exam, how="left").merge(ideal, on=grp_exam + ["k"], how="left")
    ci["baseline"] = ci["exam_mean_rate"] * ci["k"]
    ci["actual_excess"] = ci["actual_sum"] - ci["baseline"]
    ci["ideal_excess"] = ci["ideal_sum"] - ci["baseline"]
    ci["CI"] = np.where(ci["ideal_excess"] <= 0, 0, (ci["actual_excess"] / ci["ideal_excess"]) * 100)
    ci["CI"] = ci["CI"].clip(lower=0, upper=100)
    score = score.merge(ci[grp_student + ["CI"]], on=grp_student, how="left")

    if student_info is not None and not student_info.empty:
        score = score.merge(student_info, on="student_code", how="left")
        score["캠퍼스"] = score["master_campus"].fillna(score["campus"])
        score["학생명"] = score["master_student_name"].fillna(score["student_name"])
        score["재원기간"] = score["enrollment_months"]
        score["재원상태"] = np.where(score["is_enrolled"].fillna(0).astype(int) == 1, "재원", "퇴원")
    else:
        score["캠퍼스"] = score["campus"]
        score["학생명"] = score["student_name"]
        score["재원기간"] = np.nan
        score["재원상태"] = np.nan

    score["학생코드"] = score["student_code"]
    score["월"] = score["month_num"]
    score["시험유형"] = score["exam_type"]
    score["연도"] = score["year"]

    ordered = ["캠퍼스", "학생명", "학생코드", "재원기간", "재원상태", "시험유형", "연도", "월", "P-Score"]
    subject_cols_sorted = sorted(subject_cols)
    ordered += subject_cols_sorted + ["CV", "B.CV", "T-Score", "CI", "QR", "C.T-Score", "C.CV", "C.QR"]

    for c in ["P-Score", "CV", "B.CV", "T-Score", "CI", "QR", "C.T-Score", "C.CV", "C.QR"] + subject_cols_sorted:
        if c in score.columns:
            score[c] = clip_0_100(score[c]).round(2)

    return score[ordered].sort_values(["학생명", "학생코드", "시험유형", "월"]).reset_index(drop=True)


class SafeFormulaEvaluator(ast.NodeVisitor):
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, ast.Name, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)

    def __init__(self, variables: Dict[str, float]):
        self.variables = variables

    def visit(self, node):
        if not isinstance(node, self.allowed_nodes):
            raise ValueError("허용되지 않은 수식 요소가 포함되어 있습니다.")
        return super().visit(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in self.variables:
            raise ValueError(f"알 수 없는 변수: {node.id}")
        return float(self.variables[node.id])

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("숫자만 사용할 수 있습니다.")

    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("허용되지 않은 단항 연산자")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return 0.0 if right == 0 else left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("허용되지 않은 연산자")


def apply_tpi_formula(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    out = df.copy()
    aliases = {}
    for alias, col in ALIAS_TO_COLUMN.items():
        aliases[alias] = out[col].fillna(0).astype(float) if col in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    tree = ast.parse(formula, mode="eval")
    values = []
    for idx in out.index:
        vars_i = {k: float(v.loc[idx]) for k, v in aliases.items()}
        values.append(SafeFormulaEvaluator(vars_i).visit(tree))
    out["TPI"] = pd.Series(values, index=out.index).clip(lower=0, upper=100).round(2)
    return out


def make_default_formula(enabled_weights: Dict[str, float]) -> str:
    parts = []
    total = 0.0
    for alias, w in enabled_weights.items():
        if w and w > 0:
            parts.append(f"({alias}*{w})")
            total += w
    return f"({' + '.join(parts)}) / {total}" if parts else "0"


def period_sort_key(label: str) -> Tuple[int, int]:
    m = re.match(r"(MT|LT)-(\d+)월", label)
    if not m:
        return (99, 99)
    return (0 if m.group(1) == "MT" else 1, int(m.group(2)))


def build_tpi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["캠퍼스", "학생명", "학생코드", "재원기간", "지표"])
    work = df.copy()
    work["기간"] = work["시험유형"].astype(str) + "-" + work["월"].astype(int).astype(str) + "월"
    metrics_map = {m: m for m in METRIC_ORDER}
    long_parts = []
    id_cols = ["캠퍼스", "학생명", "학생코드", "재원기간", "기간"]
    for metric_name, col in metrics_map.items():
        tmp = work[id_cols + [col]].copy().rename(columns={col: "값"})
        tmp["지표"] = metric_name
        long_parts.append(tmp)
    long_df = pd.concat(long_parts, ignore_index=True)
    pivot = long_df.pivot_table(index=["캠퍼스", "학생명", "학생코드", "재원기간", "지표"], columns="기간", values="값", aggfunc="first").reset_index()
    period_cols = sorted([c for c in pivot.columns if c not in ["캠퍼스", "학생명", "학생코드", "재원기간", "지표"]], key=period_sort_key)
    pivot = pivot[["캠퍼스", "학생명", "학생코드", "재원기간", "지표"] + period_cols].copy()
    metric_map = {m: i for i, m in enumerate(METRIC_ORDER)}
    pivot["_metric_order"] = pivot["지표"].map(metric_map).fillna(999)
    pivot = pivot.sort_values(["학생명", "학생코드", "_metric_order"]).drop(columns=["_metric_order"]).reset_index(drop=True)
    for c in period_cols:
        pivot[c] = pd.to_numeric(pivot[c], errors="coerce").round(2)
    return pivot


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()

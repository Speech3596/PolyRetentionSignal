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

METRIC_ORDER = ["TPI", "P-Score", "T-Score", "B.CV", "CI", "QR", "CV", "C.CV"]
ALIAS_TO_COLUMN = {
    "P": "P-Score",
    "T": "T-Score",
    "BCV": "B.CV",
    "CI": "CI",
    "QR": "QR",
    "CV": "CV",
    "CCV": "C.CV",
}
# T-Subject aliases are dynamic and added at runtime (e.g., "T-Eng": "T-Eng")

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
    # Fallback: accept any .xlsx as exam, any .csv as student
    if ext == ".xlsx":
        return "exam"
    if ext == ".csv":
        return "student"
    return "unknown"


def parse_exam_filename(name: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    # Year: try "2025년" first, then bare "2025"
    year_m = re.search(r"(20\d{2})년", name) or re.search(r"(20\d{2})", name)
    # Month: try "3월" first, then month names, then contextual bare digits
    month_m = re.search(r"(\d{1,2})월", name)
    if not month_m:
        # Try English month names
        for eng, num in [("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
                         ("may", 5), ("jun", 6), ("jul", 7), ("aug", 8),
                         ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12)]:
            if eng in name.lower():
                month_m = type("M", (), {"group": lambda self, _=num: str(_)})()
                break
    # Exam type: MT or LT anywhere (bounded by non-alpha or start/end)
    exam_m = re.search(r"(?:^|[^a-zA-Z])(MT|LT)(?:$|[^a-zA-Z])", name, re.IGNORECASE)
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
        # Build flexible rename map: strip whitespace from actual headers
        df.columns = [str(c).strip() if pd.notna(c) else c for c in df.columns]
        _EXAM_COL_ALIASES = {
            "curriculum": ["교육과정", "curriculum"],
            "campus_type": ["운영구분", "campus_type", "운영_구분"],
            "campus": ["캠퍼스", "campus", "센터"],
            "class_name": ["학급", "class_name", "class", "반"],
            "student_code": ["학번", "student_code", "학생코드"],
            "student_name": ["이름", "student_name", "학생명", "학생이름"],
            "exam_type": ["구분", "exam_type", "시험구분", "시험유형"],
            "year": ["Year", "year", "연도"],
            "semester": ["Semester", "semester", "학기"],
            "month": ["Month", "month", "월"],
            "subject": ["시험과목", "subject", "과목"],
            "item_no": ["문항 순번", "문항순번", "item_no", "문항번호"],
            "correct_answer": ["문항정답", "correct_answer", "정답"],
            "student_answer": ["학생선택", "student_answer", "학생답", "학생응답"],
        }
        _col_lower = {str(c).lower(): c for c in df.columns if pd.notna(c)}
        _rename = {}
        for internal, aliases in _EXAM_COL_ALIASES.items():
            for alias in aliases:
                if alias in df.columns:
                    _rename[alias] = internal
                    break
                elif alias.lower() in _col_lower:
                    _rename[_col_lower[alias.lower()]] = internal
                    break
        df = df.rename(columns=_rename)
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


def _read_csv_auto_encoding(file_obj) -> pd.DataFrame:
    """Read CSV with automatic encoding detection (utf-8 → utf-8-sig → cp949 → euc-kr)."""
    raw_bytes = None
    if hasattr(file_obj, "getvalue"):
        raw_bytes = file_obj.getvalue()
    elif hasattr(file_obj, "read"):
        raw_bytes = file_obj.read()
    else:
        # file path string
        with open(file_obj, "rb") as f:
            raw_bytes = f.read()

    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: ignore errors
    return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", errors="replace")


# Known column name mappings (header name → internal name)
_STUDENT_COL_ALIASES = {
    "master_campus_type": ["campus_type", "운영구분", "운영_구분", "type"],
    "master_campus": ["campus_name", "campus", "캠퍼스", "센터"],
    "student_code": ["student_code", "학번", "학생코드", "code", "student_id"],
    "master_student_name": ["student_name", "이름", "학생명", "name", "학생이름"],
    "enrollment_months": ["enrollment_months", "재원기간", "재원개월", "months", "tenure"],
    "is_enrolled": ["is_enrolled_sept", "is_enrolled", "재원여부", "재원상태", "enrolled"],
}


def read_student_info(file_obj) -> pd.DataFrame:
    fname = getattr(file_obj, "name", None) or str(file_obj)
    if detect_file_kind(fname, expected_ext=".csv") != "student":
        raise ValueError(f"학생 데이터 파일명이 아님: {Path(fname).name}")
    # Reset file pointer in case it was previously read
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    df = _read_csv_auto_encoding(file_obj)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 6:
        raise ValueError("학생 데이터 컬럼 수가 부족하다.")

    # Try header-aware column matching first
    col_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    matched = set()
    for internal_name, aliases in _STUDENT_COL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in col_lower:
                orig_col = col_lower[alias.lower()]
                if orig_col not in matched:
                    rename_map[orig_col] = internal_name
                    matched.add(orig_col)
                    break

    if len(rename_map) >= 4:
        # Header-aware matching succeeded
        out = df.rename(columns=rename_map).copy()
    else:
        # Fallback: positional mapping (legacy behavior)
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

    required = ["master_campus_type", "master_campus", "student_code",
                "master_student_name", "enrollment_months", "is_enrolled"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"학생 데이터 필수 컬럼 매핑 실패: {missing}")

    out = out[required].copy()
    out["student_code"] = pd.to_numeric(out["student_code"], errors="coerce").astype("Int64")
    out["enrollment_months"] = pd.to_numeric(out["enrollment_months"], errors="coerce")
    out["is_enrolled"] = pd.to_numeric(out["is_enrolled"], errors="coerce").fillna(0).astype(int)
    return out.drop_duplicates(subset=["student_code"])


def build_item_stats(raw: pd.DataFrame) -> pd.DataFrame:
    grp = ["exam_type", "year", "month_num", "subject", "item_no"]
    item = raw.groupby(grp, dropna=False).agg(응시자수=("student_code", "nunique"), 정답자수=("is_correct", "sum")).reset_index()
    item["문항 정답률"] = item["정답자수"] / item["응시자수"] * 100
    return item


_SUBJECT_ABBREV = {
    "english": "Eng",
    "speech building": "S.B",
    "eng. foundations": "Eng.F",
    "foundations": "Fnd",
    "math": "Math",
    "science": "Sci",
    "social studies": "S.S",
    "reading": "Rdg",
    "writing": "Wrt",
    "grammar": "Grm",
    "vocabulary": "Voc",
    "listening": "Lst",
    "speaking": "Spk",
}


def _make_t_subject_name(subject: str) -> str:
    """Create abbreviated T-Subject metric name, e.g., 'English' -> 'T-Eng'.
    Column names may contain dots/spaces but formula aliases must be valid Python identifiers.
    """
    s = subject.strip()
    key = s.lower()
    if key in _SUBJECT_ABBREV:
        return f"T-{_SUBJECT_ABBREV[key]}"
    # Auto-abbreviate: take first letters of each word, max 5 chars
    parts = s.replace(".", " ").split()
    if len(parts) == 1:
        abbr = s[:4]
    else:
        abbr = "".join(p[0].upper() for p in parts[:3])
    return f"T-{abbr}"


def get_t_subject_cols(df: pd.DataFrame) -> list:
    """Return list of T-Subject column names present in the dataframe."""
    return [c for c in df.columns if c.startswith("T-") and c != "T-Score"]


def _col_to_formula_alias(col: str) -> str:
    """Convert column name to a valid Python identifier for formula evaluation.
    E.g., 'T-Eng' -> 'TSEng', 'T-S.B' -> 'TSSB', 'T-Eng.F' -> 'TSEngF'
    """
    import re as _re
    # Replace non-alphanumeric with nothing, prefix 'TS' for T-Subject
    clean = col.replace("T-", "TS")
    clean = _re.sub(r'[^a-zA-Z0-9]', '', clean)
    return clean


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

    # T-Subject: per-subject T-Score (e.g., T-Eng, T-S.B, T-Eng.F)
    # Students who did NOT take a particular subject keep NaN (not 50)
    t_subject_cols = []
    for subj_col in subject_cols:
        if subj_col in score.columns:
            t_col = _make_t_subject_name(subj_col)
            # Compute T-Score only for non-NaN values; preserve NaN for missing subjects
            has_value = score[subj_col].notna()
            score[t_col] = np.nan
            if has_value.any():
                score.loc[has_value, t_col] = score.loc[has_value].groupby(
                    grp_exam, dropna=False
                )[subj_col].transform(tscore_from_series)
            t_subject_cols.append(t_col)

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
    t_subject_cols_sorted = sorted(t_subject_cols)
    ordered += subject_cols_sorted + ["CV", "B.CV", "T-Score", "CI", "QR", "C.CV"] + t_subject_cols_sorted

    for c in ["P-Score", "CV", "B.CV", "T-Score", "CI", "QR", "C.CV"] + subject_cols_sorted + t_subject_cols_sorted:
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

    # T-Subject aliases: keep NaN tracking for redistribution
    t_subj_alias_map = {}  # alias -> column name
    for col in get_t_subject_cols(out):
        alias_key = _col_to_formula_alias(col)
        aliases[alias_key] = out[col].fillna(0).astype(float)
        t_subj_alias_map[alias_key] = col

    tree = ast.parse(formula, mode="eval")

    # Parse weights from formula to enable T-Subject redistribution
    t_subj_weights = _extract_t_subject_weights(formula, set(t_subj_alias_map.keys()))

    values = []
    for idx in out.index:
        vars_i = {k: float(v.loc[idx]) for k, v in aliases.items()}

        # T-Subject redistribution: if some T-Subject cols are NaN for this row,
        # redistribute their weight to present T-Subjects proportionally
        if t_subj_weights:
            present_aliases = []
            missing_aliases = []
            for ts_alias, ts_col in t_subj_alias_map.items():
                if ts_alias in t_subj_weights:
                    if pd.notna(out.at[idx, ts_col]):
                        present_aliases.append(ts_alias)
                    else:
                        missing_aliases.append(ts_alias)

            if missing_aliases and present_aliases:
                # Redistribute missing weight to present T-Subjects
                missing_total = sum(t_subj_weights[a] for a in missing_aliases)
                present_total = sum(t_subj_weights[a] for a in present_aliases)
                if present_total > 0:
                    boost_factor = (present_total + missing_total) / present_total
                    for a in present_aliases:
                        vars_i[a] = vars_i[a] * boost_factor
                # Zero out missing so they don't contribute
                for a in missing_aliases:
                    vars_i[a] = 0.0

        values.append(SafeFormulaEvaluator(vars_i).visit(tree))

    out["TPI"] = pd.Series(values, index=out.index).clip(lower=0, upper=100).round(2)
    return out


def _extract_t_subject_weights(formula: str, t_subj_aliases: set) -> Dict[str, float]:
    """Extract T-Subject weights from formula string like '(TSEng*2.0)'.
    Returns {alias: weight} for T-Subject aliases found in the formula.
    """
    import re
    weights = {}
    for alias in t_subj_aliases:
        # Match patterns like (TSEng*2.0) or TSEng*2.0
        pattern = rf'{re.escape(alias)}\s*\*\s*([0-9.]+)'
        m = re.search(pattern, formula)
        if m:
            weights[alias] = float(m.group(1))
    return weights


def make_default_formula(enabled_weights: Dict[str, float]) -> str:
    parts = []
    total = 0.0
    for alias, w in enabled_weights.items():
        if w and w > 0:
            # T-Subject aliases use underscore in formulas (T_Eng, T_S.B)
            parts.append(f"({alias}*{w})")
            total += w
    return f"({' + '.join(parts)}) / {total}" if parts else "0"


def compute_default_tpi_weights(t_subject_cols: list) -> Dict[str, float]:
    """Compute default TPI weights based on available T-Subject metrics.
    Default: T 40%, BCV 30%, T-subjects share 10% equally (split by count).
    If 4 subjects: each T-subject = 2.5%, T adjusts to 30%.
    Remaining metrics (P, CI, CV, CCV) default to 0 (unchecked).
    """
    n_subj = len(t_subject_cols)
    if n_subj > 0:
        t_subj_total = 10.0
        t_subj_each = round(t_subj_total / n_subj, 2)
        # If subjects present, T adjusts: T=40% - 10% = 30% (total T-family = 40%)
        t_weight = 30.0
    else:
        t_weight = 40.0
        t_subj_each = 0.0

    weights = {
        "P": 0.0,
        "T": t_weight,
        "BCV": 30.0,
        "CI": 0.0,
        "QR": 0.0,
        "CV": 0.0,
        "CCV": 0.0,
    }
    # Add T-Subject weights
    for col in t_subject_cols:
        alias = _col_to_formula_alias(col)
        weights[alias] = t_subj_each

    return weights


def period_sort_key(label: str) -> Tuple[int, int]:
    m = re.match(r"(MT|LT)-(\d+)월", label)
    if not m:
        return (99, 99)
    return (0 if m.group(1) == "MT" else 1, int(m.group(2)))


def get_all_metrics(df: pd.DataFrame) -> list:
    """Return METRIC_ORDER + any T-Subject columns found in df."""
    t_subj = get_t_subject_cols(df)
    return METRIC_ORDER + t_subj


def build_tpi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["캠퍼스", "학생명", "학생코드", "재원기간", "지표"])
    work = df.copy()
    work["기간"] = work["시험유형"].astype(str) + "-" + work["월"].astype(int).astype(str) + "월"
    all_metrics = get_all_metrics(df)
    metrics_map = {m: m for m in all_metrics}
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
    metric_map = {m: i for i, m in enumerate(all_metrics)}
    pivot["_metric_order"] = pivot["지표"].map(metric_map).fillna(999)
    pivot = pivot.sort_values(["학생명", "학생코드", "_metric_order"]).drop(columns=["_metric_order"]).reset_index(drop=True)
    for c in period_cols:
        pivot[c] = pd.to_numeric(pivot[c], errors="coerce").round(2)
    return pivot


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()

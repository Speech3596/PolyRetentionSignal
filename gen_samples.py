import pandas as pd
import numpy as np
import os

# 1. Student Sample (CSV)
# Header: campus_type, campus, student_code, student_name, ... , enrollment_months, is_enrolled
students = pd.DataFrame({
    "운영구분": ["A", "A", "B"],
    "캠퍼스": ["서울", "서울", "부산"],
    "학번": [1001, 1002, 1003],
    "이름": ["김철수", "이영희", "박지민"],
    "기타1": ["X", "Y", "Z"],
    "기타2": [1, 2, 3],
    "재원기간": [12, 6, 24],
    "재원여부": [1, 1, 0]
})
students.to_csv("sample_students.csv", index=False, encoding="utf-8-sig")

# 2. Exam Sample (XLSX)
# Sheets: any name
# Header starts at row 3 (0-indexed 2)
# Columns: 교육과정, 운영구분, 캠퍼스, 학급, 학번, 이름, 구분, Year, Semester, Month, 시험과목, 문항 순번, 문항정답, 학생선택
data = []
for code, name in [(1001, "김철수"), (1002, "이영희"), (1003, "박지민")]:
    for item in range(1, 6):
        data.append({
            "교육과정": "Regular",
            "운영구분": "A" if code < 1003 else "B",
            "캠퍼스": "서울" if code < 1003 else "부산",
            "학급": "Class1",
            "학번": code,
            "이름": name,
            "구분": "MT",
            "Year": 2024,
            "Semester": 1,
            "Month": "3월",
            "시험과목": "Math" if item <= 3 else "English",
            "문항 순번": item,
            "문항정답": "1",
            "학생선택": "1" if np.random.rand() > 0.3 else "2"
        })

df_exam = pd.DataFrame(data)

# Create Excel with header at row 3
with pd.ExcelWriter("sample_exams.xlsx", engine="openpyxl") as writer:
    # Empty rows for padding (header at row 3)
    dummy = pd.DataFrame([[""]*14]*2)
    dummy.to_excel(writer, index=False, header=False, startrow=0)
    df_exam.to_excel(writer, index=False, startrow=2)

print("Sample files created: sample_students.csv, sample_exams.xlsx")

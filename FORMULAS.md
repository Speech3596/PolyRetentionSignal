# PolyRetentionSignal 산식

- P-Score = 총정답수 / 총문항수 × 100
- 과목별 총점 = 과목정답수 / 과목문항수 × 100
- CV = 같은 시험유형·연도·월 전체 학생의 P-Score 변동계수를 100 기준 역방향 환산한 값
  - raw CV = 표준편차 / 평균 × 100
  - 표시 CV = max(0, 100 - raw CV)
- B.CV = 학생별 과목 총점 간 변동계수를 100 기준 역방향 환산한 값
  - raw B.CV = 과목별 총점의 표준편차 / 과목별 총점의 평균 × 100
  - 표시 B.CV = max(0, 100 - raw B.CV)
- T-Score = 50 + 10 × ((개인 P-Score - 전체 평균 P-Score) / 전체 표준편차), 0~100 clip
- CI
  1. 한 시험의 모든 문항 정답률 계산
  2. 문항 정답률을 내림차순 정렬
  3. 학생이 맞힌 문항 수를 k라 둠
  4. 학생이 실제로 맞힌 문항 정답률 합 = actual_sum
  5. 기준선 = 전체 문항 정답률 평균 × k
  6. 이상적 top-k 문항 정답률 합 = ideal_sum
  7. actual_excess = actual_sum - 기준선
  8. ideal_excess = ideal_sum - 기준선
  9. CI = actual_excess / ideal_excess × 100
  10. 분모가 0 이하이면 0 처리
  11. 결과는 0~100 clip
- QR = 같은 시험유형·연도·월 전체 학생 기준 P-Score 백분위 위치값(0~100)
- C.T-Score = 같은 캠퍼스 내 P-Score 기준 T-Score, 0~100 clip
- C.CV = 같은 시험유형·연도·월·캠퍼스 내 학생들의 P-Score 변동계수를 100 기준 역방향 환산한 값
  - raw C.CV = 표준편차 / 평균 × 100
  - 표시 C.CV = max(0, 100 - raw CV)
- C.QR = 같은 시험유형·연도·월·캠퍼스 기준 P-Score 백분위 위치값(0~100)

## TPI
- 사용자가 지표 채택 여부 선택
- 사용자가 비율 입력
- 사용자가 사칙연산 및 괄호를 사용해 수식을 직접 작성
- 사용 가능한 별칭
  - P = P-Score
  - T = T-Score
  - BCV = B.CV
  - CI = CI
  - QR = QR
  - CT = C.T-Score
  - CCV = C.CV
  - CQR = C.QR
  - CV = CV

## TPI 출력 구조
- A열: 캠퍼스
- B열: 학생명
- C열: 학생코드
- D열: 재원기간
- E열: 지표명
- F열 이후: 선택한 기간별 값
- 기간 컬럼명 예시: MT-3월, MT-4월, LT-5월
- 한 학생당 아래 지표 9행이 세로로 누적
  - TPI
  - P-Score
  - T-Score
  - B.CV
  - CI
  - QR
  - C.T-Score
  - C.CV
  - C.QR

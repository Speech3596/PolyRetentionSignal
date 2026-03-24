# PolyRetentionSignal

## 패키지 구성
- `app.py` : Streamlit UI 진입점
- `retentionsignal_core.py` : 로딩/정리/산식/매트릭스 생성 코어
- `requirements.txt` : 의존성
- `FORMULAS.md` : 산식 정의
- `run.bat` : 로컬 원클릭 실행 배치

## 로컬 실행
1. 같은 폴더에 패키지 파일 5개를 둔다.
2. `run.bat` 더블클릭
3. 브라우저에서 Streamlit 실행 화면 확인

### 오프라인 실행 조건
- 최초 1회 의존성 설치 전에는 인터넷이 필요할 수 있다.
- 한 번 설치가 끝나 `.venv`가 만들어진 뒤에는 동일 PC에서 오프라인 재실행 가능하다.

## GitHub → Streamlit Cloud 배포
- 저장소 루트에 아래 파일 그대로 업로드
  - `app.py`
  - `retentionsignal_core.py`
  - `requirements.txt`
  - `README.md`
  - `FORMULAS.md`
- Streamlit Cloud에서 Main file path를 `app.py`로 지정
- 배포 후 웹에서 시험 `.xlsx` 여러 개와 학생 `.csv` 1개 업로드하여 사용

## 입력 규칙
- 시험 데이터: `.xlsx` 여러 개
- 학생 데이터: `.csv` 1개
- 학생 데이터 규칙
  - 우측 끝에서 두번째 열 = 재원기간
  - 마지막 열 = 재원 여부 (`1` 재원 / `0` 퇴원)

## 출력
- 학생 성적표 CSV
- 문항 정답률 CSV
- TPI 매트릭스 CSV

## 주의
- 샘플 데이터는 포함하지 않음
- 시험 파일과 학생 파일은 업로드 구멍이 분리되어 있음
- 파일명은 시험/학생 성격이 드러나도록 저장하는 것이 안전함

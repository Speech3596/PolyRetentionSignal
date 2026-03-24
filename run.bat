@echo off
setlocal
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip --version >nul 2>&1 || goto :pip_error

if not exist ".venv\.deps_installed" (
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt || goto :install_error
    echo ok> ".venv\.deps_installed"
)

python -m streamlit run app.py
exit /b 0

:pip_error
echo Python venv 실행 실패
pause
exit /b 1

:install_error
echo 패키지 설치 실패. 최초 1회는 인터넷 연결이 필요할 수 있음.
pause
exit /b 1
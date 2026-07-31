@echo off
setlocal
cd /d "%~dp0StrokeGuard_App"
echo StrokeGuard - Clinical Decision Support System
echo Product design and application engineering by Estiuk Arafat Arnob

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 and try again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating an isolated preview environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create the preview environment.
    pause
    exit /b 1
  )
)

set "VENV_PYTHON=.venv\Scripts\python.exe"

"%VENV_PYTHON%" -c "import streamlit, sklearn, xgboost, lightgbm, catboost, plotly, reportlab" >nul 2>&1
if errorlevel 1 (
  echo Installing StrokeGuard dependencies. This may take several minutes...
  "%VENV_PYTHON%" -m pip install --upgrade pip
  "%VENV_PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

"%VENV_PYTHON%" preview.py
endlocal

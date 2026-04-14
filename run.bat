@echo off
title ARIA — Disaster Command Center

:: Always run from the folder this script is in
cd /d "%~dp0"

echo.
echo  =================================================
echo    ARIA — Agentic Disaster Response AI
echo  =================================================
echo.

:: Check .env exists
if not exist ".env" (
    echo  ERROR: .env file not found!
    echo  Copy .env.example to .env and add your GROQ_API_KEY
    echo  Get a free key at: https://console.groq.com
    pause
    exit /b 1
)

:: Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo  Activating virtual environment...
    call "venv\Scripts\activate.bat"
) else (
    echo  WARNING: venv not found. Creating it now...
    python -m venv venv
    call "venv\Scripts\activate.bat"
    pip install -r requirements.txt
)

:: Quick dependency check
python -c "import langgraph, langchain_groq, streamlit" 2>nul
if errorlevel 1 (
    echo  Installing missing packages...
    pip install -r requirements.txt
)

echo  Starting ARIA dashboard...
echo  Open browser at: http://localhost:8501
echo.
streamlit run app.py

pause

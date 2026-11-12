@echo off
REM ================================================================
REM run.bat - Windows one-click launcher for Disaster Response AI
REM ================================================================
REM Usage:
REM   run.bat            - Start both API + Next.js frontend (no Docker)
REM   run.bat docker     - Start with Docker Compose
REM   run.bat api        - Start FastAPI only
REM   run.bat frontend   - Start Next.js frontend only
REM ================================================================

SET ROOT=%~dp0
SET VENV=%ROOT%venv

IF "%1"=="docker" GOTO docker
IF "%1"=="api" GOTO api
IF "%1"=="frontend" GOTO frontend

REM Default: start both
:both
echo [Disaster Response AI] Starting API + Frontend...
CALL :check_venv
START "FastAPI Backend" cmd /k "%VENV%\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000 --reload"
TIMEOUT /T 3 /NOBREAK >NUL
START "Next.js Frontend" cmd /k "cd /d %ROOT%disaster-frontend && npm run dev"
echo [OK] FastAPI running on http://localhost:8000
echo [OK] Next.js running on http://localhost:3000
echo [OK] FastAPI docs: http://localhost:8000/docs
GOTO end

:api
echo [Disaster Response AI] Starting FastAPI only...
CALL :check_venv
"%VENV%\Scripts\uvicorn.exe" api.main:app --host 0.0.0.0 --port 8000 --reload
GOTO end

:frontend
echo [Disaster Response AI] Starting Next.js frontend only...
cd /d "%ROOT%disaster-frontend"
npm run dev
GOTO end

:docker
echo [Disaster Response AI] Starting with Docker Compose...
docker compose up --build
GOTO end

:check_venv
IF NOT EXIST "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at %VENV%
    echo [FIX] Run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    EXIT /B 1
)
CALL "%VENV%\Scripts\activate.bat"
EXIT /B 0

:end

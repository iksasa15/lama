@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting FastAPI server (port 8010)...
start "Detection API" cmd /k "cd /d "%~dp0server" && .venv\Scripts\activate && uvicorn main:app --reload --host 127.0.0.1 --port 8010"

timeout /t 2 /nobreak >nul

echo Starting React (Vite)...
start "Detection UI" cmd /k "cd /d "%~dp0" && npm run dev"

echo.
echo Both started in separate windows.
echo UI: http://localhost:5173
echo API: http://127.0.0.1:8010
echo.
pause

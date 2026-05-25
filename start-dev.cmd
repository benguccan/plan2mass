@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%backend"
set "FRONTEND_DIR=%ROOT%frontend"

set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%BACKEND_DIR%\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Backend Python bulunamadi.
  echo Beklenen konumlar:
  echo   %BACKEND_DIR%\.venv\Scripts\python.exe
  echo   %BACKEND_DIR%\venv\Scripts\python.exe
  exit /b 1
)

where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo npm.cmd bulunamadi. Node.js kurulu oldugundan emin olun.
  exit /b 1
)

echo Backend baslatiliyor: http://127.0.0.1:8000
start "Plan2Mass Backend" cmd /k "cd /d "%BACKEND_DIR%" && "%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo Frontend baslatiliyor: http://127.0.0.1:5173
start "Plan2Mass Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && npm.cmd run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Site acildiktan sonra su adresi ziyaret edin:
echo   http://127.0.0.1:5173

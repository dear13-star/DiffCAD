@echo off
chcp 65001 >nul
echo ==========================================
echo   DIffCAD Agent - Public Deploy Script
echo ==========================================
echo.

cd /d "%~dp0"

:: Kill any existing backend
echo [1/3] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":5000" ^| find "LISTENING"') do (
    echo   Killing PID %%a on port 5000...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: Start backend
echo [2/3] Starting DIffCAD Agent backend...
start "DIffCAD-Backend" /MIN python backend_api.py
echo   Waiting for backend to be ready...

:wait
timeout /t 3 /nobreak >nul
curl -s -o nul http://127.0.0.1:5000/api/status 2>nul
if errorlevel 1 goto wait
echo   Backend is ready!

:: Start tunnel
echo [3/3] Starting Cloudflare tunnel (no interstitial page)...
echo.
echo   Look for the URL below starting with https:// and ending in trycloudflare.com
echo.

cloudflared.exe tunnel --url http://localhost:5000 2>&1 | findstr "trycloudflare.com"

echo.
echo ==========================================
echo   Tunnel closed. Backend may still be running.
echo ==========================================
pause

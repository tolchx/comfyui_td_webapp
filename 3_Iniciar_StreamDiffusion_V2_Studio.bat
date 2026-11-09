@echo off
chcp 65001 >nul
title Modulo 2: StreamDiffusionV2 + ComfyUI_StreamDiffusion + ScreenDiffusion Studio
cd /d "%~dp0"

echo ====================================================================
echo   MODULO 2: STREAMDIFFUSION V2 + SCREENDIFFUSION STUDIO (PUERTO 8081)
echo ====================================================================
echo   - Conectado a ComfyUI (http://127.0.0.1:8188) y TouchDesigner
echo   - WebApp Fusion en http://127.0.0.1:8081
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

REM Abrir la WebApp de StreamDiffusionV2 Studio en el navegador
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8081"

%PYTHON_CMD% "streamdiffusion_studio\server_stream_studio.py" --port 8081

pause

@echo off
chcp 65001 >nul
title Módulo 4: AI Video Studio (T2V & I2V)
cd /d "%~dp0.."

echo ====================================================================
echo   MÓDULO 4: AI VIDEO STUDIO (PUERTO 8083)
echo ====================================================================
echo   - Modos: Text-to-Video (T2V) e Image-to-Video (I2V)
echo   - Motores: Wan 2.2 / 2.1 14B 4-steps, LTX-Video, Motion Synth 3D
echo   - WebApp Video en http://127.0.0.1:8083
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8083"
%PYTHON_CMD% "video_studio\server_video.py" --port 8083

pause

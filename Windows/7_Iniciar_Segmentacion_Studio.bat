@echo off
chcp 65001 >nul
title Módulo 5: AI Segmentation & Inpainting Studio (SAM 2 & Alpha Matting)
cd /d "%~dp0.."

echo ====================================================================
echo   MÓDULO 5: SEGMENTACIÓN ALFA & INPAINTING (PUERTO 8084)
echo ====================================================================
echo   - Segmentación interactiva por puntos (SAM 2) y 1-Click Auto
echo   - Matting sub-píxel transparente y reconstrucción de fondo (Inpainting)
echo   - WebApp Segmentación en http://127.0.0.1:8084
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8084"
%PYTHON_CMD% "segmentation_studio\server_segmentation.py" --port 8084

pause

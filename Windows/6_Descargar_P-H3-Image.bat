@echo off
chcp 65001 >nul
title Descargador Oficial P-H3-Image INT8 ConvRot (8MP / 8 Pasos / 6GB VRAM)
cd /d "%~dp0.."

echo ====================================================================
echo   DESCARGADOR OFICIAL P-H3-IMAGE (Pruna AI / MiniMax-H3)
echo   8 Megapixels • 8 Pasos • 4MP Edición • INT8 ConvRot • 6GB+ VRAM
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

%PYTHON_CMD% "bridge\download_ph3_image.py"

echo.
pause

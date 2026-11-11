@echo off
chcp 65001 >nul
title Descargador Oficial MiniMax-H3-Image Nativo ComfyUI (4MP-16MP / 6 Pasos)
cd /d "%~dp0.."

echo ====================================================================
echo   DESCARGADOR OFICIAL MINIMAX-H3-IMAGE (Nativo ComfyUI)
echo   4MP-16MP • 6 Pasos • Multi-Ref Edit • INT8 ConvRot • 6GB+ VRAM
echo   Referencia: https://blog.comfy.org/p/minimax-h3-image-is-now-available
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

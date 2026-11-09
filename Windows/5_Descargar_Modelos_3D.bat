@echo off
chcp 65001 > nul
title Descargador de Modelos 3D (Hunyuan 3D v2.1 y TRELLIS.2)
cd /d "%~dp0.."

echo =======================================================================
echo    MÓDULO 3: DESCARGADOR DE MODELOS 3D NEURONALES (COMFYUI)
echo =======================================================================
echo.
echo Descarga los checkpoints oficiales de Tencent Hunyuan 3D y Microsoft
echo TRELLIS.2 directamente a tu carpeta de ComfyUI con soporte de reanudación.
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

%PYTHON_CMD% "trellis3d_studio\download_3d_models.py"

echo.
pause

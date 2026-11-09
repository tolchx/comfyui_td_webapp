@echo off
chcp 65001 > nul
title Descargador de Modelos 3D (Hunyuan 3D v2.1 y TRELLIS.2)

echo =======================================================================
echo    MODULO 3: DESCARGADOR DE MODELOS 3D NEURONALES (COMFYUI)
echo =======================================================================
echo.
echo Descarga los checkpoints oficiales de Tencent Hunyuan 3D y Microsoft
echo TRELLIS.2 directamente a tu carpeta de ComfyUI con soporte de reanudacion.
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] No se encontro Python en el PATH del sistema.
    pause
    exit /b 1
)

python "%~dp0trellis3d_studio\download_3d_models.py"

echo.
pause

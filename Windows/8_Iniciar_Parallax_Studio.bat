@echo off
chcp 65001 >nul
title Módulo 6: 2.5D Parallax & Depth Studio para TouchDesigner
cd /d "%~dp0.."

echo ====================================================================
echo   MÓDULO 6: 2.5D PARALLAX & DEPTH STUDIO (PUERTO 8085)
echo ====================================================================
echo   - Estimación de mapas de profundidad y descomposición multi-plano
echo   - Malla 3D deformada (.OBJ y .GLB) con textura proyectada
echo   - WebApp Parallax en http://127.0.0.1:8085
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8085"
%PYTHON_CMD% "parallax25d_studio\server_parallax.py" --port 8085

pause

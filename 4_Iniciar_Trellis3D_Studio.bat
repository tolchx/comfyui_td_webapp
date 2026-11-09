@echo off
chcp 65001 >nul
title Modulo 3: TRELLIS v2 + Hunyuan 3D Local Studio (Imagen a Objeto 3D)
cd /d "%~dp0"

echo ====================================================================
echo   MODULO 3: TRELLIS v2 + HUNYUAN 3D LOCAL STUDIO (PUERTO 8082)
echo ====================================================================
echo   - Conectado a ComfyUI (http://127.0.0.1:8188), Modulo 1 y TouchDesigner
echo   - WebApp 3D en http://127.0.0.1:8082
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

REM Abrir la WebApp de Trellis 3D Studio en el navegador
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8082"

%PYTHON_CMD% "trellis3d_studio\server_trellis3d.py" --port 8082

pause

@echo off
chcp 65001 >nul
title Servidor Puente TouchDesigner ^<--^> ComfyUI + WebApp
cd /d "%~dp0"

echo ====================================================================
echo   INICIANDO PUENTE TOUCHDESIGNER ^<--^> COMFYUI + WEBAPP
echo ====================================================================
echo   - Iniciando ComfyUI en segundo plano (si esta instalado)
echo   - Levantando Servidor Puente y WebApp en http://127.0.0.1:8080
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

REM Abrir la WebApp en el navegador luego de 2 segundos
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8080"

REM Ejecutar el servidor puente con inicio automatico de ComfyUI en segundo plano
%PYTHON_CMD% "bridge\server.py" --autostart-comfy

pause

@echo off
chcp 65001 >nul
title Curso IA Generativa — Instalador de Modelos y ComfyUI
cd /d "%~dp0.."

echo ====================================================================
echo   CURSO INTELIGENCIA ARTIFICIAL: TOUCHDESIGNER ^<--^> COMFYUI
echo   Asistente de Detección de Hardware e Instalación de Modelos
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

%PYTHON_CMD% "bridge\installer.py"

echo.
echo ====================================================================
echo   Configuración finalizada.
echo   Ahora podés hacer doble clic en: Windows\2_Iniciar_Sistema_TD_Comfy.bat
echo ====================================================================
pause

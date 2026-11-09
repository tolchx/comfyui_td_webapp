@echo off
chcp 65001 >nul
title TouchDiffusion — Lanzador de Proyecto TouchDesigner
cd /d "%~dp0"

echo ====================================================================
echo   ABRIENDO PROYECTO TOUCHDIFFUSION EN TOUCHDESIGNER
echo ====================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] El entorno .venv aun no esta instalado. Ejecutando instalador primero...
    call "1_Instalar_TouchDiffusion.bat"
)

set "PATH=%~dp0.venv\Lib\site-packages\torch\lib;%~dp0.venv\Lib\site-packages\tensorrt_libs;%PATH%"

set TD_EXE=
for /d %%D in ("C:\Program Files\Derivative\TouchDesigner*") do (
    if exist "%%D\bin\TouchDesigner.exe" (
        set TD_EXE="%%D\bin\TouchDesigner.exe"
    )
)

if defined TD_EXE (
    echo Iniciando TouchDesigner: %TD_EXE%
    start "" %TD_EXE% "%~dp0TD\TouchDiffusion_Clase_IA.toe"
) else (
    echo Abriendo "%~dp0TD\TouchDiffusion_Clase_IA.toe" con la aplicacion predeterminada...
    start "" "%~dp0TD\TouchDiffusion_Clase_IA.toe"
)

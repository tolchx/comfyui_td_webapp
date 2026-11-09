@echo off
chcp 65001 >nul
title TouchDiffusion — WebUI Compilador de Motores TensorRT
cd /d "%~dp0"

echo ====================================================================
echo   TOUCHDIFFUSION STANDALONE — COMPILADOR DE MOTORES TENSORRT
echo   Abriendo interfaz WebUI en http://127.0.0.1:7860
echo ====================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [!] El entorno .venv aun no esta instalado. Ejecutando instalador automatico...
    call "1_Instalar_TouchDiffusion.bat"
)

set "PATH=%~dp0.venv\Lib\site-packages\torch\lib;%~dp0.venv\Lib\site-packages\tensorrt_libs;%PATH%"

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:7860"
".venv\Scripts\python.exe" "StreamDiffusion\webui.py"

pause

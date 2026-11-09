@echo off
chcp 65001 >nul
title Curso IA Generativa — Instalador Standalone de TouchDiffusion
cd /d "%~dp0"

echo ====================================================================
echo   CURSO INTELIGENCIA ARTIFICIAL: TOUCHDIFFUSION STANDALONE
echo   Instalador Automatico (Python 3.11 + PyTorch CUDA + TensorRT)
echo ====================================================================
echo.

set BOOTSTRAP_PY=

REM 1. Si ya existe .venv con Python 3.11, usarlo
if exist ".venv\Scripts\python.exe" (
    set BOOTSTRAP_PY=".venv\Scripts\python.exe"
    goto :run_setup
)

REM 2. Buscar Python 3.11 integrado en TouchDesigner 2023/2025
for /d %%D in ("C:\Program Files\Derivative\TouchDesigner*") do (
    if exist "%%D\bin\python.exe" (
        set BOOTSTRAP_PY="%%D\bin\python.exe"
    )
)
if defined BOOTSTRAP_PY goto :run_setup

REM 3. Buscar Python del sistema
where python >nul 2>&1
if %errorlevel% equ 0 (
    set BOOTSTRAP_PY=python
    goto :run_setup
)

REM 4. Si no hay ningun Python en la PC, descargar Python 3.11.10 Portable con PowerShell
if not exist "python311_portable\python\python.exe" (
    echo [!] Descargando Python 3.11.10 Portable para iniciar el instalador...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "New-Item -ItemType Directory -Force -Path 'python311_portable' | Out-Null; " ^
      "Invoke-WebRequest -Uri 'https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-x86_64-pc-windows-msvc-install_only.tar.gz' -OutFile 'python311_portable\py311.tar.gz'; " ^
      "tar -xzf 'python311_portable\py311.tar.gz' -C 'python311_portable'; " ^
      "Remove-Item 'python311_portable\py311.tar.gz' -Force"
)
set BOOTSTRAP_PY="python311_portable\python\python.exe"

:run_setup
%BOOTSTRAP_PY% "setup_standalone.py"

echo.
pause

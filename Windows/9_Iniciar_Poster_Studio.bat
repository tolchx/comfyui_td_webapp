@echo off
chcp 65001 >nul
title Módulo 7: Poster Studio (MOSS-Image)
cd /d "%~dp0.."

echo ====================================================================
echo   MÓDULO 7: POSTER STUDIO - MOSS-Image (PUERTO 8086)
echo ====================================================================
echo   - Modo hibrido: arte con MOSS + tipografia real (Pillow)
echo   - Modo IA pura: el modelo dibuja tambien el texto
echo   - Requiere la instancia ComfyUI con MOSS en 127.0.0.1:8189
echo   - WebApp en http://127.0.0.1:8086
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

rem Chequeo rapido del motor MOSS (8189). El 8188 de la suite no sirve: MOSS pide ComfyUI >= 0.44.
%PYTHON_CMD% -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8189/system_stats',timeout=3); print('[OK] motor MOSS online')" 2>nul || echo [AVISO] El motor MOSS (8189) no responde. Iniciar la instancia MOSS antes de generar.

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8086"
%PYTHON_CMD% "poster_studio\server_poster.py" --port 8086

pause

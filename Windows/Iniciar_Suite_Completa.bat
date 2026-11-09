@echo off
chcp 65001 >nul
title Suite Completa TouchDesigner <-> ComfyUI AI (6 Módulos)
cd /d "%~dp0.."

echo ====================================================================
echo   INICIANDO SUITE COMPLETA TD <-> COMFYUI (6 MÓDULOS ACTIVOS)
echo ====================================================================
echo   🎨 Módulo 1 (Imágenes HQ):           http://127.0.0.1:8080
echo   ⚡ Módulo 2 (StreamDiffusion RT):    http://127.0.0.1:8081
echo   🧊 Módulo 3 (3D Studio):             http://127.0.0.1:8082
echo   🎬 Módulo 4 (Video Studio):          http://127.0.0.1:8083
echo   ✂️ Módulo 5 (Alfa & Inpaint):        http://127.0.0.1:8084
echo   🌌 Módulo 6 (Parallax 2.5D):         http://127.0.0.1:8085
echo   ⚙️ Servidor ComfyUI Base:            http://127.0.0.1:8188
echo ====================================================================
echo.

set PYTHON_CMD=python
if exist "ComfyUI_windows_portable\python_embeded\python.exe" (
    set PYTHON_CMD="ComfyUI_windows_portable\python_embeded\python.exe"
) else if exist "D:\ComfyUI\.venv\Scripts\python.exe" (
    set PYTHON_CMD="D:\ComfyUI\.venv\Scripts\python.exe"
)

REM Iniciar servidores en segundo plano
start "Módulo 1: Imagen HQ" %PYTHON_CMD% "bridge\server.py" --port 8080
start "Módulo 2: StreamDiffusion" %PYTHON_CMD% "streamdiffusion_studio\server_stream_studio.py" --port 8081
start "Módulo 3: 3D Studio" %PYTHON_CMD% "trellis3d_studio\server_trellis3d.py" --port 8082
start "Módulo 4: Video Studio" %PYTHON_CMD% "video_studio\server_video.py" --port 8083
start "Módulo 5: Segmentación" %PYTHON_CMD% "segmentation_studio\server_segmentation.py" --port 8084
start "Módulo 6: Parallax 2.5D" %PYTHON_CMD% "parallax25d_studio\server_parallax.py" --port 8085

timeout /t 3 /nobreak >nul
start http://127.0.0.1:8080

echo.
echo Todos los módulos han sido iniciados correctamente.
echo Podés navegar entre ellos desde la barra superior de cualquiera de las aplicaciones.
echo.
pause

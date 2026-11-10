#!/bin/bash
# Lanzador de la Suite Completa TD <-> ComfyUI AI para macOS
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  INICIANDO SUITE COMPLETA TD <--> COMFYUI (macOS)"
echo "===================================================================="
echo "  🎨 Módulo 1 (Imágenes HQ):           http://127.0.0.1:8080"
echo "  ⚡ Módulo 2 (StreamDiffusion RT):    http://127.0.0.1:8081"
echo "  🧊 Módulo 3 (3D Studio):             http://127.0.0.1:8082"
echo "  🎬 Módulo 4 (Video Studio):          http://127.0.0.1:8083"
echo "  ✂️ Módulo 5 (Alfa & Inpaint):        http://127.0.0.1:8084"
echo "  🌌 Módulo 6 (Parallax 2.5D):         http://127.0.0.1:8085"
echo "===================================================================="
echo ""

PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi

"$PYTHON_CMD" "bridge/server.py" --port 8080 &
"$PYTHON_CMD" "streamdiffusion_studio/server_stream_studio.py" --port 8081 &
"$PYTHON_CMD" "trellis3d_studio/server_trellis3d.py" --port 8082 &
"$PYTHON_CMD" "video_studio/server_video.py" --port 8083 &
"$PYTHON_CMD" "segmentation_studio/server_segmentation.py" --port 8084 &
"$PYTHON_CMD" "parallax25d_studio/server_parallax.py" --port 8085 &

sleep 3
open "http://127.0.0.1:8080"

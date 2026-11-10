#!/bin/bash
# Lanzador Módulo 4: Video Studio T2V & I2V para macOS
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  INICIANDO MÓDULO 4: AI VIDEO STUDIO (macOS)"
echo "  URL WebApp: http://127.0.0.1:8083"
echo "===================================================================="
PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi
(sleep 2 && open "http://127.0.0.1:8083") &
"$PYTHON_CMD" "video_studio/server_video.py" --port 8083

#!/bin/bash
# Lanzador Módulo 6: 2.5D Parallax & Depth Studio para macOS
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  INICIANDO MÓDULO 6: 2.5D PARALLAX & DEPTH STUDIO (macOS)"
echo "  URL WebApp: http://127.0.0.1:8085"
echo "===================================================================="
PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi
(sleep 2 && open "http://127.0.0.1:8085") &
"$PYTHON_CMD" "parallax25d_studio/server_parallax.py" --port 8085

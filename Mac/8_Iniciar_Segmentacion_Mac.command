#!/bin/bash
# Lanzador Módulo 5: Segmentación Alfa & Inpainting Studio para macOS
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  INICIANDO MÓDULO 5: SEGMENTACIÓN ALFA & INPAINTING (macOS)"
echo "  URL WebApp: http://127.0.0.1:8084"
echo "===================================================================="
PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi
(sleep 2 && open "http://127.0.0.1:8084") &
"$PYTHON_CMD" "segmentation_studio/server_segmentation.py" --port 8084

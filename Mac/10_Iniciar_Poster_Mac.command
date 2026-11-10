#!/bin/bash
# Lanzador Módulo 7: Poster Studio (MOSS-Image) para macOS
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  INICIANDO MÓDULO 7: POSTER STUDIO (macOS)"
echo "  URL WebApp: http://127.0.0.1:8086"
echo "===================================================================="
PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi
(sleep 2 && open "http://127.0.0.1:8086") &
"$PYTHON_CMD" "poster_studio/server_poster.py" --port 8086

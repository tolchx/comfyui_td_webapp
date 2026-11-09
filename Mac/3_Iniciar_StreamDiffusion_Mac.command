#!/bin/bash
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  MÓDULO 2: STREAMDIFFUSION V2 + SCREENDIFFUSION STUDIO (macOS)"
echo "  WebApp en http://127.0.0.1:8081"
echo "===================================================================="

PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
fi

(sleep 2 && open "http://127.0.0.1:8081") &
"$PYTHON_CMD" "streamdiffusion_studio/server_stream_studio.py" --port 8081

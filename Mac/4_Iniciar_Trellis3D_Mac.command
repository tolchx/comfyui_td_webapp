#!/bin/bash
cd "$(dirname "$0")/.."
echo "===================================================================="
echo "  MÓDULO 3: TRELLIS v2 + HUNYUAN 3D LOCAL STUDIO (macOS)"
echo "  WebApp en http://127.0.0.1:8082"
echo "===================================================================="

PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
fi

(sleep 2 && open "http://127.0.0.1:8082") &
"$PYTHON_CMD" "trellis3d_studio/server_trellis3d.py" --port 8082

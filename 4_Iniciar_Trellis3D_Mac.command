#!/bin/bash
cd "$(dirname "$0")"
echo "===================================================================="
echo "  MODULO 3: TRELLIS v2 + HUNYUAN 3D LOCAL STUDIO (macOS)"
echo "  WebApp en http://127.0.0.1:8082"
echo "===================================================================="

(sleep 2 && open "http://127.0.0.1:8082") &
python3 "trellis3d_studio/server_trellis3d.py" --port 8082

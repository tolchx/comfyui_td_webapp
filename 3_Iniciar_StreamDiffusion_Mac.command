#!/bin/bash
cd "$(dirname "$0")"
echo "===================================================================="
echo "  MODULO 2: STREAMDIFFUSION V2 + SCREENDIFFUSION STUDIO (macOS)"
echo "  WebApp en http://127.0.0.1:8081"
echo "===================================================================="

(sleep 2 && open "http://127.0.0.1:8081") &
python3 "streamdiffusion_studio/server_stream_studio.py" --port 8081

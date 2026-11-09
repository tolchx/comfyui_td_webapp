#!/bin/bash
# Lanzador del Servidor Puente + ComfyUI en Segundo Plano para macOS
cd "$(dirname "$0")/.."

echo "===================================================================="
echo "  INICIANDO PUENTE TOUCHDESIGNER <--> COMFYUI + WEBAPP (macOS)"
echo "===================================================================="
echo "  - Iniciando ComfyUI en segundo plano con soporte Metal (MPS)"
echo "  - Levantando Servidor Puente y WebApp en http://127.0.0.1:8080"
echo "===================================================================="
echo ""

PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi

# Asegurar Pillow en el entorno Python para el sintetizador procedural
"$PYTHON_CMD" -c "import PIL" 2>/dev/null || "$PYTHON_CMD" -m pip install pillow >/dev/null 2>&1

# Iniciar Módulo 2 y Módulo 3 en segundo plano
"$PYTHON_CMD" "streamdiffusion_studio/server_stream_studio.py" --port 8081 &
"$PYTHON_CMD" "trellis3d_studio/server_trellis3d.py" --port 8082 &

# Abrir la WebApp en el navegador predeterminado de macOS luego de 2 segundos
(sleep 2 && open "http://127.0.0.1:8080") &

# Ejecutar el servidor puente con inicio automático de ComfyUI
"$PYTHON_CMD" "bridge/server.py" --autostart-comfy

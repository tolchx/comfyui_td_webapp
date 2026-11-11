#!/bin/bash
# Descargador Oficial MiniMax-H3-Image Nativo ComfyUI para macOS
cd "$(dirname "$0")/.."

echo "===================================================================="
echo "  DESCARGADOR OFICIAL MINIMAX-H3-IMAGE (Nativo ComfyUI) - macOS"
echo "  4MP-16MP • 6 Pasos • Multi-Ref Edit • INT8 ConvRot • 6GB+ VRAM"
echo "  Referencia: https://blog.comfy.org/p/minimax-h3-image-is-now-available"
echo "===================================================================="
echo ""

PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi

"$PYTHON_CMD" "bridge/download_ph3_image.py"

echo ""
echo "Presioná Enter para cerrar..."
read -r

#!/bin/bash
# Descargador Oficial P-H3-Image INT8 ConvRot para macOS
cd "$(dirname "$0")/.."

echo "===================================================================="
echo "  DESCARGADOR OFICIAL P-H3-IMAGE (Pruna AI / MiniMax-H3) - macOS"
echo "  8 Megapixels • 8 Pasos • 4MP Edición • INT8 ConvRot • 6GB+ VRAM"
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

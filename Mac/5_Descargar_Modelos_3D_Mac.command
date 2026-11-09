#!/bin/bash
cd "$(dirname "$0")/.."
echo "======================================================================="
echo "   MÓDULO 3: DESCARGADOR DE MODELOS 3D NEURONALES (COMFYUI MAC)"
echo "======================================================================="

PYTHON_CMD="python3"
if [ -d "ComfyUI/.venv/bin" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
fi

"$PYTHON_CMD" "trellis3d_studio/download_3d_models.py"
read -p "Presiona ENTER para salir..."

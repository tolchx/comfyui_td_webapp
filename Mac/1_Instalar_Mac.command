#!/bin/bash
# Lanzador de Instalación y Configuración para macOS (Apple Silicon M1/M2/M3/M4 e Intel)
cd "$(dirname "$0")/.."

echo "===================================================================="
echo "  CURSO INTELIGENCIA ARTIFICIAL: TOUCHDESIGNER <--> COMFYUI (macOS)"
echo "  Asistente de Detección de Hardware Apple Metal (MPS) y Modelos"
echo "===================================================================="
echo ""

PYTHON_CMD="python3"
if [ -f "ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="ComfyUI/.venv/bin/python"
elif [ -f "$HOME/Documents/ComfyUI/.venv/bin/python" ]; then
    PYTHON_CMD="$HOME/Documents/ComfyUI/.venv/bin/python"
fi

"$PYTHON_CMD" "bridge/installer.py"

echo ""
echo "===================================================================="
echo "  Configuración finalizada en macOS."
echo "  Ahora podés hacer doble clic en: Mac/2_Iniciar_Sistema_Mac.command"
echo "===================================================================="
read -p "Presiona Enter para salir..."

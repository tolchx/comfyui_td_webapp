"""Puente MOSS-Image ↔ ComfyUI, adaptado a la suite (solo stdlib + Pillow).

Origen del toolkit: github.com/tolchx/MOSS-Image-Colab → local-4090/mosskit (Apache-2.0).
Acá se reemplaza `requests` por urllib para no agregar dependencias a la cátedra.
"""

__all__ = ["layout", "graph", "profiles", "comfy", "typography"]

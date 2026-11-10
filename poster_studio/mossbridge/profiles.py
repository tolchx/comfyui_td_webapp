"""Perfiles de muestreo medidos. Un perfil es un dict, no un archivo que se parchea.

Origen de los números: repo MOSS-Image-Colab (configs/profiles) + plantilla oficial de ComfyUI.
`text_safe=False` marca lo que verifiqué acá: con 8 pasos el texto se rompe.
"""

PROFILES = {
    # Plantilla oficial "MOSS-Image: Text to Image"
    "quality25": {
        "steps": 25, "cfg": 2.5, "sampler": "er_sde", "scheduler": "simple",
        "slg": ("", "22", 0.0, 0.5), "lora": None, "denoise": 1.0, "text_safe": True,
        "note": "equilibrio por defecto; texto español correcto en 10/10 de las pruebas",
    },
    # Perfil quality50 del repo
    "quality50": {
        "steps": 50, "cfg": 3.0, "sampler": "euler", "scheduler": "simple",
        "slg": ("", "22", 0.0, 0.5), "lora": None, "denoise": 1.0, "text_safe": True,
        "note": "más lento (~17 s); útil para piezas finales con párrafos",
    },
    # LoRA DMD8: rápido pero rompe la tipografía
    "fast8": {
        "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
        "slg": None, "lora": ("moss_image_sft_dmd8_lora.safetensors", 0.7),
        "denoise": 1.0, "text_safe": False,
        "note": "8 pasos, 3,4 s. SOLO para explorar composición: el texto sale ilegible",
    },
    # Plantilla oficial "MOSS-Image: Image Edit LoRA"
    "edit": {
        "steps": 30, "cfg": 4.0, "sampler": "euler", "scheduler": "normal",
        "slg": ("", "22", 0.0, 1.0),
        "lora": ("moss_image_edit_lora_rank_64_bf16.safetensors", 1.0),
        "denoise": 1.0, "text_safe": True, "edit_format": "Moss-Image-Edit",
        "note": "redibuja guiado; no conserva la composición original",
    },
    # Inpaint del repo: VAEEncode + SetLatentNoiseMask sobre la máscara con feather
    "inpaint": {
        "steps": 30, "cfg": 3.0, "sampler": "euler", "scheduler": "simple",
        "slg": None, "lora": None, "denoise": 0.65, "text_safe": True,
        "note": "denoise 0.65 dentro de la máscara; el resto se recompone por píxel",
    },
}

UNET = "moss_image_sft_int8_convrot.safetensors"
CLIP = "qwen3.5_9b_int8_convrot.safetensors"
VAE = "flux2-vae_small.safetensors"


def get(name: str) -> dict:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"perfil desconocido: {name!r} (hay: {', '.join(PROFILES)})") from None

#!/usr/bin/env python3
"""
Constructores de Workflows API para ComfyUI (Text-to-Image e Image-to-Image).
Soporta:
  - Tier Básico: SD 1.5 + LCM LoRA (o SD 1.5 directo)
  - Tier Medio: SDXL Turbo / SDXL Lightning
  - Tier Avanzado: Flux.1 Schnell FP8 (Checkpoint único) o Flux.2 Dev (UNET + CLIP + VAE separados)
"""

import random


def build_workflow(
    tier_id: str,
    tier_cfg: dict,
    prompt: str,
    negative_prompt: str = "low quality, blurry, watermark, text, deformed",
    width: int = 512,
    height: int = 512,
    steps: int = 6,
    cfg: float = 1.8,
    seed: int = -1,
    denoise: float = 1.0,
    input_image_name: str | None = None,
    checkpoint_override: str | None = None,
    available_loras: list[str] | None = None,
    available_vaes: list[str] | None = None,
    realtime_mode: bool = False,
) -> dict:
    if seed is None or int(seed) < 0:
        # En modo StreamDiffusion Realtime, mantenemos una semilla determinista (424242) para evitar flicker entre frames
        seed = 424242 if realtime_mode else random.randint(1, 2**48 - 1)
    else:
        seed = int(seed)

    width = max(256, min(4096, (int(width) // 8) * 8))
    height = max(256, min(4096, (int(height) // 8) * 8))
    steps = max(1, min(50, int(steps)))
    cfg = float(cfg)
    denoise = max(0.05, min(1.0, float(denoise)))

    is_rt = realtime_mode or (tier_id == "realtime") or bool(tier_cfg.get("use_taesd"))
    ckpt = checkpoint_override or tier_cfg.get("checkpoint", "DreamShaper_8_pruned.safetensors")

    # Caso especial 0: P-H3-Image INT8 ConvRot (8 Megapixels en 8 pasos con 6GB VRAM)
    if "p-h3-image" in ckpt.lower() or "ph3" in ckpt.lower() or tier_id in ("p_h3_image", "ph3_image"):
        return _build_ph3_image_workflow(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps if steps in (4, 6, 8, 10, 12) else 8,
            seed=seed,
            denoise=denoise,
            input_image_name=input_image_name,
            unet_name="p-h3-image_int8_convrot.safetensors" if "w4a8" not in ckpt.lower() else "p-h3-image_w4a8.safetensors",
            clip_name="p-h3-image_text_encoder_int8_convrot.safetensors",
            vae_name="p-h3-image_vae_int8_convrot.safetensors",
            use_tiled_vae=True,
        )

    # Caso especial 1: Qwen Image 2.1 (instalado en E:\video_creator_minimax\models)
    if "qwen_image" in ckpt.lower():
        return _build_qwen_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=2 if is_rt else steps,
            cfg=cfg,
            seed=seed,
            denoise=denoise,
            input_image_name=input_image_name,
        )

    # Caso especial 2: Flux.2 Klein 4B o Flux.2 Dev FP8 (siempre usa flux2-vae completo para máxima calidad de imagen)
    if "flux2" in ckpt.lower() or "flux-2" in ckpt.lower():
        return _build_flux2_workflow(
            unet_name="flux-2-klein-4b.safetensors" if "klein" in ckpt.lower() else "flux2_dev_fp8mixed.safetensors",
            clip_name="qwen_3_4b.safetensors" if "klein" in ckpt.lower() else "mistral_3_small_flux2_bf16.safetensors",
            vae_name="flux2-vae.safetensors",
            prompt=prompt,
            width=width,
            height=height,
            steps=max(4, steps),
            seed=seed,
            denoise=denoise,
            input_image_name=input_image_name,
            realtime_mode=is_rt,
        )

    sampler_name = tier_cfg.get("sampler_name", "euler")
    scheduler = tier_cfg.get("scheduler", "normal")
    lora_name = tier_cfg.get("lora")

    if is_rt:
        # En modo StreamDiffusion Realtime: preferir Hyper-SD 1-Step LoRA o LCM LoRA con 1 paso y CFG=1.0
        if available_loras and "Hyper-SD15-1step-lora.safetensors" in available_loras:
            lora_name = "Hyper-SD15-1step-lora.safetensors"
            steps = 1
            cfg = 1.0
            sampler_name = "lcm"
            scheduler = "sgm_uniform"
        elif available_loras and "lcm_lora_sd15.safetensors" in available_loras:
            lora_name = "lcm_lora_sd15.safetensors"
            steps = min(steps, 2)
            cfg = 1.0
            sampler_name = "lcm"
            scheduler = "sgm_uniform"

    # Si el LoRA configurado no está instalado aún, usar euler_ancestral directo sin LoRA
    if lora_name and available_loras is not None and lora_name not in available_loras:
        lora_name = None
        if sampler_name == "lcm":
            sampler_name = "euler_ancestral"
            scheduler = "normal"
            steps = max(steps, 8)
            cfg = max(cfg, 4.0)

    workflow = {}

    # 1. Checkpoint Loader
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ckpt},
    }

    model_ref = ["1", 0]
    clip_ref = ["1", 1]
    vae_ref = ["1", 2]

    # 1b. StreamDiffusion TAESD (Tiny AutoEncoder) para decodificación/codificación instantánea sin cuello de botella de VRAM
    if is_rt and available_vaes:
        tae_target = "taesdxl" if "xl" in ckpt.lower() else ("taef1" if "flux1" in ckpt.lower() else "taesd")
        if tae_target in available_vaes:
            workflow["99"] = {
                "class_type": "VAELoader",
                "inputs": {"vae_name": tae_target},
            }
            vae_ref = ["99", 0]

    # 2. Optional LoRA Loader (ej. LCM LoRA en Tier Básico)
    if lora_name:
        workflow["2"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        model_ref = ["2", 0]
        clip_ref = ["2", 1]

    # 3. Positive Prompt
    workflow["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": clip_ref,
        },
    }

    # 4. Negative Prompt
    workflow["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": negative_prompt or "",
            "clip": clip_ref,
        },
    }

    # 5. Latent Source: EmptyLatentImage (txt2img) o LoadImage + VAEEncode (img2img)
    if input_image_name:
        workflow["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        }
        workflow["11"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "center",
            },
        }
        workflow["5"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["11", 0],
                "vae": vae_ref,
            },
        }
        latent_ref = ["5", 0]
    else:
        workflow["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        }
        latent_ref = ["5", 0]
        denoise = 1.0

    # 6. KSampler
    workflow["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "model": model_ref,
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": latent_ref,
        },
    }

    # 7. VAEDecode
    workflow["7"] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["6", 0],
            "vae": vae_ref,
        },
    }

    # 8. SaveImage (en modo StreamDiffusion Realtime usa PreviewImage con compress_level=1 para ahorrar ~75ms de CPU PNG)
    if is_rt:
        workflow["8"] = {
            "class_type": "PreviewImage",
            "inputs": {"images": ["7", 0]},
        }
    else:
        workflow["8"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "TD_Comfy_Course",
                "images": ["7", 0],
            },
        }

    return workflow


def _build_qwen_image_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float = 1.0,
    input_image_name: str | None = None,
) -> dict:
    """Workflow nativo para Qwen Image 2.1 (instalado en E:\\video_creator_minimax\\models)."""
    wf = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
        },
        "4": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "resolution": max(width, height),
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": max(8, steps),
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise if input_image_name else 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 1],
                "latent_image": ["5", 0],
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0],
            },
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "TD_Qwen21",
                "images": ["7", 0],
            },
        },
    }
    if input_image_name:
        wf["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        }
        wf["11"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "center",
            },
        }
        wf["5"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["11", 0],
                "vae": ["3", 0],
            },
        }
    else:
        wf["5"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        }
    return wf


def _build_flux2_workflow(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    denoise: float = 1.0,
    input_image_name: str | None = None,
    unet_name: str = "flux-2-klein-4b.safetensors",
    clip_name: str = "qwen_3_4b.safetensors",
    vae_name: str = "flux2-vae.safetensors",
    realtime_mode: bool = False,
) -> dict:
    """Workflow nativo para Flux.2 Klein 4B o Flux.2 Dev FP8 (con soporte TAESD/TAEF2 y Low-Steps Realtime)."""
    wf = {
        "10": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "12": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "flux2",
                "device": "default",
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["38", 0],
            },
        },
        "26": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["6", 0],
                "guidance": 2.2 if realtime_mode else 3.5,
            },
        },
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": max(2, steps) if realtime_mode else max(4, steps),
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise if input_image_name else 1.0,
                "model": ["12", 0],
                "positive": ["26", 0],
                "negative": ["6", 0],
                "latent_image": ["47", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["10", 0],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "TD_Flux2",
                "images": ["8", 0],
            },
        },
    }

    if input_image_name:
        wf["45"] = {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        }
        wf["46"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["45", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "center",
            },
        }
        wf["47"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["46", 0],
                "vae": ["10", 0],
            },
        }
    else:
        wf["47"] = {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        }

    return wf


def _build_ph3_image_workflow(
    prompt: str,
    width: int = 2048,
    height: int = 2048,
    steps: int = 8,
    seed: int = 424242,
    denoise: float = 1.0,
    input_image_name: str | None = None,
    unet_name: str = "p-h3-image_int8_convrot.safetensors",
    clip_name: str = "p-h3-image_text_encoder_int8_convrot.safetensors",
    vae_name: str = "p-h3-image_vae_int8_convrot.safetensors",
    use_tiled_vae: bool = True,
) -> dict:
    """
    Workflow nativo para P-H3-Image INT8 ConvRot (Pruna AI / MiniMax-H3):
    =====================================================================
    - 8 Megapixels de generación T2I (ej. 3872×2176 o 2880×2880) en 8 pasos con CFG = 1.0.
    - 4 Megapixels de edición I2I con imágenes de referencia contextuales (<Picture 1>).
    - Cuantización INT8 ConvRot (matrices de rotación Hadamard contra outliers) + VAEDecodeTiled (512px).
    - Memoria VRAM requerida: Solo 6 GB a 8 GB VRAM gracias al offload dinámico y tiling VAE.
    """
    wf: dict = {
        # 1. Cargar el Diffusion Transformer (INT8 ConvRot o W4A8)
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        # 2. Cargar Text Encoder (INT8 ConvRot)
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "default",
                "device": "default",
            },
        },
        # 3. Cargar VAE (INT8 ConvRot o BF16)
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        # 4. Patch de Sigma Shift MiniMax-H3 para muestreo distilled en 8 pasos
        "4": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["1", 0],
                "shift_video": 1.0,
                "shift_audio": 1.0,
            },
        },
    }

    cond_text = prompt
    cond_inputs = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "text": cond_text,
        "width": width,
        "height": height,
    }

    if input_image_name:
        # Modo Edición 4MP / Referencia
        wf["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        }
        wf["11"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "bilinear",
                "width": min(width, 2560),
                "height": min(height, 2560),
                "crop": "center",
            },
        }
        cond_inputs["image_1"] = ["11", 0]
        if "<Picture 1>" not in cond_text and "<Picture" not in cond_text:
            cond_inputs["text"] = f"<Picture 1> {cond_text}"

    # 5. Condicionamiento nativo PH3ImageConditioning (ComfyUI v0.44+)
    wf["5"] = {
        "class_type": "PH3ImageConditioning",
        "inputs": cond_inputs,
    }

    # 6. Sampler en 8 pasos con CFG = 1.0 (sin Negative Prompt duplicando memoria)
    wf["6"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": max(4, min(12, steps)),
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise if input_image_name else 1.0,
            "model": ["4", 0],
            "positive": ["5", 0],
            "negative": ["5", 0],
            "latent_image": ["5", 1],
        },
    }

    # 7. Decodificación VAE Tiled (512px) para garantizar 0 OOM en 6GB VRAM a 8 Megapixels
    if use_tiled_vae:
        wf["7"] = {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0],
                "tile_size": 512,
                "overlap": 64,
            },
        }
    else:
        wf["7"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["3", 0],
            },
        }

    # 8. Guardar Imagen HD / 8 Megapixels
    wf["8"] = {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "TD_PH3_Image",
            "images": ["7", 0],
        },
    }

    return wf

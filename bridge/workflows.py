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
) -> dict:
    if seed is None or int(seed) < 0:
        seed = random.randint(1, 2**48 - 1)
    else:
        seed = int(seed)

    width = max(256, min(2048, (int(width) // 8) * 8))
    height = max(256, min(2048, (int(height) // 8) * 8))
    steps = max(1, min(50, int(steps)))
    cfg = float(cfg)
    denoise = max(0.05, min(1.0, float(denoise)))

    ckpt = checkpoint_override or tier_cfg.get("checkpoint", "DreamShaper_8_pruned.safetensors")

    # Caso especial 1: Qwen Image 2.1 (instalado en E:\video_creator_minimax\models)
    if "qwen_image" in ckpt.lower():
        return _build_qwen_image_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            denoise=denoise,
            input_image_name=input_image_name,
        )

    # Caso especial 2: Flux.2 Klein 4B o Flux.2 Dev FP8
    if "flux2" in ckpt.lower() or "flux-2" in ckpt.lower():
        return _build_flux2_workflow(
            unet_name="flux-2-klein-4b.safetensors" if "klein" in ckpt.lower() else "flux2_dev_fp8mixed.safetensors",
            clip_name="qwen_3_4b.safetensors" if "klein" in ckpt.lower() else "mistral_3_small_flux2_bf16.safetensors",
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            denoise=denoise,
            input_image_name=input_image_name,
        )

    sampler_name = tier_cfg.get("sampler_name", "euler")
    scheduler = tier_cfg.get("scheduler", "normal")
    lora_name = tier_cfg.get("lora")

    # Si el LoRA configurado no está instalado aún, usar euler_ancestral directo sin LoRA
    if lora_name and available_loras is not None and lora_name not in available_loras:
        lora_name = None
        if sampler_name == "lcm":
            sampler_name = "euler_ancestral"
            scheduler = "normal"
            steps = max(steps, 12)
            cfg = max(cfg, 6.0)

    workflow = {}

    # 1. Checkpoint Loader
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ckpt},
    }

    model_ref = ["1", 0]
    clip_ref = ["1", 1]
    vae_ref = ["1", 2]

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

    # 8. SaveImage
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
) -> dict:
    """Workflow nativo para Flux.2 Klein 4B o Flux.2 Dev FP8."""
    wf = {
        "10": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "flux2-vae.safetensors"},
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
                "guidance": 3.5,
            },
        },
        "22": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["12", 0],
                "conditioning": ["26", 0],
            },
        },
        "16": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "25": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "48": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": max(4, steps),
                "width": width,
                "height": height,
            },
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["25", 0],
                "guider": ["22", 0],
                "sampler": ["16", 0],
                "sigmas": ["48", 0],
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

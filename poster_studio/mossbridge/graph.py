"""Grafos API de ComfyUI para MOSS-Image. Sin parcheo en runtime: cada grafo sale completo."""

from __future__ import annotations

from . import profiles as P


def _base(unet_dir="", lora=None):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": P.UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": P.CLIP, "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": P.VAE}},
    }
    model = ["1", 0]
    if lora:
        name, strength = lora
        g["10"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = ["10", 0]
    return g, model


def _sampler(g, model, positive, negative, latent, profile, seed):
    slg = profile.get("slg")
    if slg:
        g["11"] = {"class_type": "SkipLayerGuidanceDiTSimple",
                   "inputs": {"model": model, "double_layers": slg[0], "single_layers": slg[1],
                              "start_percent": slg[2], "end_percent": slg[3]}}
        model = ["11", 0]
    g["7"] = {"class_type": "KSampler",
              "inputs": {"model": model, "positive": positive, "negative": negative,
                         "latent_image": latent, "seed": int(seed),
                         "steps": profile["steps"], "cfg": profile["cfg"],
                         "sampler_name": profile["sampler"], "scheduler": profile["scheduler"],
                         "denoise": profile.get("denoise", 1.0)}}
    return ["7", 0]


def _finish(g, samples, prefix):
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": samples, "vae": ["3", 0]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


def _t2i_minimax_h3(prompt, prof, width, height, seed, prefix):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"}},
        "4": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": int(width), "height": int(height)}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["4", 1],
            "seed": int(seed), "steps": prof.get("steps", 6), "cfg": 1.0,
            "sampler_name": prof.get("sampler", "euler"), "scheduler": prof.get("scheduler", "simple"),
            "denoise": 1.0
        }},
        "7": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["6", 0], "vae": ["3", 0], "tile_size": 512, "overlap": 64}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}}
    }
    return g


def t2i(prompt, profile, width=1024, height=1024, seed=42, negative="", prefix="mosskit_t2i"):
    prof = P.get(profile) if isinstance(profile, str) else profile
    if prof.get("backend") == "minimax_h3" or "minimax" in str(profile).lower():
        return _t2i_minimax_h3(prompt, prof, width, height, seed, prefix)
    g, model = _base(lora=prof.get("lora"))
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}}
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}}
    g["6"] = {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
    samples = _sampler(g, model, ["4", 0], ["5", 0], ["6", 0], prof, seed)
    return _finish(g, samples, prefix)


def edit(image_name, instruction, profile="edit", seed=42,
         negative="nsfw, worst quality, bad anatomy, watermark, jpeg artifact", prefix="mosskit_edit"):
    prof = P.get(profile)
    g, model = _base(lora=prof.get("lora"))
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["21"] = {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["20", 0], "upscale_method": "lanczos",
                          "megapixels": 1.0, "resolution_steps": 16}}
    g["22"] = {"class_type": "GetImageSize", "inputs": {"image": ["21", 0]}}
    g["23"] = {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": ["22", 0], "height": ["22", 1], "batch_size": 1}}
    for node, prompt in (("24", instruction), ("25", negative)):
        g[node] = {"class_type": "TextEncodeMOSSImageEdit",
                   "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                              "edit_format": prof.get("edit_format", "Moss-Image-Edit"),
                              "image_strength": 1.0, "images.image_0": ["21", 0]}}
    samples = _sampler(g, model, ["24", 0], ["25", 0], ["23", 0], prof, seed)
    return _finish(g, samples, prefix)


def inpaint(image_name, mask_name, prompt, profile="inpaint", seed=42, negative="",
            prefix="mosskit_inpaint"):
    """VAEEncode + SetLatentNoiseMask: repinta solo lo blanco de la máscara, con denoise parcial."""
    prof = P.get(profile)
    g, model = _base(lora=prof.get("lora"))
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}}
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative}}
    g["11i"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["12"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
    g["13"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11i", 0], "vae": ["3", 0]}}
    g["14"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["12", 0]}}
    samples = _sampler(g, model, ["4", 0], ["5", 0], ["14", 0], prof, seed)
    return _finish(g, samples, prefix)

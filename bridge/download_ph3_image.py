#!/usr/bin/env python3
"""
Descargador Oficial de MiniMax H3 Image / P-H3-Image INT8 para ComfyUI
======================================================================
Descarga los pesos nativos oficiales de Comfy-Org y Pruna AI:
  - Oficial Comfy-Org (blog.comfy.org/p/minimax-h3-image-is-now-available):
      * models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors (~21.0 GB)
      * models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors (~25.28 GB)
      * models/vae/minimax_h3_image_vae_fp16.safetensors (~1.17 GB)
  - Pruna AI (P-H3-Image INT8 ConvRot con Text Encoder compacto de 5.1 GB):
      * models/diffusion_models/p-h3-image_int8_convrot.safetensors (~21.2 GB)
      * models/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors (~5.1 GB)
      * models/vae/p-h3-image_vae_int8_convrot.safetensors (~3.1 GB)

Permite generar a 4MP-16MP y editar con múltiples referencias en 6 pasos en GPUs desde 6 GB VRAM.
Incluye soporte de reanudación automática (.part), cálculo de MB/s y tiempo restante.
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bridge.installer import load_config, resolve_comfy_installation, detect_gpu

PH3_VARIANTS = {
    "comfy_org": {
        "name": "MiniMax-H3-Image Oficial Comfy-Org (Nativo ComfyUI • blog.comfy.org • ~47.4 GB)",
        "files": [
            {
                "name": "minimax_h3_image_pruned_int8_convrot.safetensors",
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                "size_gb": 21.0,
            },
            {
                "name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "subdir": "text_encoders",
                "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "size_gb": 25.28,
            },
            {
                "name": "minimax_h3_image_vae_fp16.safetensors",
                "subdir": "vae",
                "url": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors",
                "size_gb": 1.17,
            },
        ],
    },
    "int8": {
        "name": "MiniMax-H3 / P-H3-Image INT8 (Text Encoder Compacto 5.1GB • Ahorro de Disco • ~29.4 GB)",
        "files": [
            {
                "name": "minimax_h3_image_pruned_int8_convrot.safetensors",
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                "size_gb": 21.0,
            },
            {
                "name": "p-h3-image_text_encoder_int8_convrot.safetensors",
                "subdir": "text_encoders",
                "url": "https://huggingface.co/PrunaAI/P-H3-Image-Comfy/resolve/main/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
                "size_gb": 5.1,
            },
            {
                "name": "minimax_h3_image_vae_fp16.safetensors",
                "subdir": "vae",
                "url": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors",
                "size_gb": 1.17,
            },
        ],
    },
    "w4a8": {
        "name": "MiniMax-H3 / P-H3 W4A8 Ultra-Comprimido (Máximo Ahorro • ~19.7 GB)",
        "files": [
            {
                "name": "p-h3-image_w4a8.safetensors",
                "subdir": "diffusion_models",
                "url": "https://huggingface.co/PrunaAI/P-H3-Image-Comfy/resolve/main/diffusion_models/p-h3-image_w4a8.safetensors",
                "size_gb": 13.4,
            },
            {
                "name": "p-h3-image_text_encoder_int8_convrot.safetensors",
                "subdir": "text_encoders",
                "url": "https://huggingface.co/PrunaAI/P-H3-Image-Comfy/resolve/main/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
                "size_gb": 5.1,
            },
            {
                "name": "minimax_h3_image_vae_fp16.safetensors",
                "subdir": "vae",
                "url": "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors",
                "size_gb": 1.17,
            },
        ],
    },
}


def download_with_resume(url: str, dest_path: Path, label: str = "") -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    downloaded = 0
    if part_path.exists():
        downloaded = part_path.stat().st_size

    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-TD-Downloader/1.0"})
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")

    print(f"\n[DESCARGA] {label or dest_path.name}")
    print(f"  URL   : {url}")
    print(f"  Destino: {dest_path}")
    if downloaded > 0:
        print(f"  Reanudando desde: {downloaded / (1024**3):.2f} GB")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = getattr(resp, "status", 200)
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) + downloaded if total_size else None

            mode = "ab" if downloaded > 0 and status in (206, 200) else "wb"
            if mode == "wb":
                downloaded = 0

            start_t = time.time()
            last_t = start_t
            chunk_size = 1024 * 1024  # 1 MB

            with open(part_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_t >= 0.5:
                        last_t = now
                        elapsed = max(0.001, now - start_t)
                        avg_speed_mb = (downloaded / (1024**2)) / elapsed
                        if total_bytes:
                            pct = (downloaded / total_bytes) * 100.0
                            rem_bytes = max(0, total_bytes - downloaded)
                            eta_s = int(rem_bytes / max(1, avg_speed_mb * 1024 * 1024))
                            eta_str = f"{eta_s // 60}m {eta_s % 60}s"
                            bar = "=" * int(pct / 4) + ">" + " " * (25 - int(pct / 4))
                            print(
                                f"\r  [{bar}] {pct:5.1f}% | {downloaded/(1024**3):.2f}/{total_bytes/(1024**3):.2f} GB | {avg_speed_mb:5.1f} MB/s | ETA: {eta_str}   ",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r  Descargando: {downloaded / (1024**3):.2f} GB | {avg_speed_mb:5.1f} MB/s   ",
                                end="",
                                flush=True,
                            )

        print()
        if part_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
            print(f"  [OK] Guardado exitosamente: {dest_path.name}")
            return True
        return False
    except Exception as e:
        print(f"\n  [ERROR] Falló la descarga de {dest_path.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargador de MiniMax-H3-Image para ComfyUI")
    parser.add_argument("--variant", choices=["comfy_org", "int8", "w4a8"], default=None)
    parser.add_argument("--auto", action="store_true", help="Descargar variante recomendada sin preguntar")
    args = parser.parse_args()

    print("====================================================================")
    print("  DESCARGADOR OFICIAL MINIMAX-H3-IMAGE (Nativo ComfyUI)")
    print("  4MP-16MP • 6 Pasos • Multi-Ref Edit • INT8 ConvRot • 6GB+ VRAM")
    print("  Referencia: https://blog.comfy.org/p/minimax-h3-image-is-now-available")
    print("====================================================================")

    gpu = detect_gpu()
    print(f"  GPU detectada: {gpu.get('name', 'CPU')} ({gpu.get('vram_gb', 0)} GB VRAM)")

    cfg = load_config()
    inst = resolve_comfy_installation(cfg)
    if not inst:
        print("\n[ERROR] No se detectó ninguna instalación de ComfyUI configurada.")
        print("Asegurate de haber instalado ComfyUI Portable o clonado ComfyUI.")
        sys.exit(1)

    models_dir = Path(inst["models_dir"])
    print(f"  Directorio ComfyUI Models: {models_dir}")

    choice = args.variant
    if not choice and not args.auto:
        print("\nSeleccioná la variante de MiniMax-H3-Image a descargar:")
        print("  [1] MiniMax-H3-Image Oficial Comfy-Org (Recomendado oficial: ~47.4 GB total)")
        print("  [2] MiniMax-H3 / P-H3 INT8 Ahorro de Disco (Text Encoder Compacto 5.1GB, ~29.4 GB total)")
        print("  [3] MiniMax-H3 / P-H3 W4A8 Ultra-Comprimido (~19.7 GB total)")
        print("  [0] Salir")
        try:
            val = input("\nOpción [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            val = "0"
        if val == "1":
            choice = "comfy_org"
        elif val == "2":
            choice = "int8"
        elif val == "3":
            choice = "w4a8"
        else:
            print("Operación cancelada.")
            sys.exit(0)
    elif not choice:
        choice = "comfy_org"

    variant_info = PH3_VARIANTS.get(choice, PH3_VARIANTS["comfy_org"])
    print(f"\nVariante seleccionada: {variant_info['name']}")

    for file_spec in variant_info["files"]:
        target_dir = models_dir / file_spec["subdir"]
        dest = target_dir / file_spec["name"]
        if dest.exists() and dest.stat().st_size > 1024 * 1024 * 100:  # > 100 MB
            print(f"\n[EXISTE] {dest.name} ya se encuentra descargado ({dest.stat().st_size / (1024**3):.2f} GB). Saltando.")
            continue

        ok = download_with_resume(
            url=file_spec["url"],
            dest_path=dest,
            label=f"{file_spec['name']} ({file_spec['size_gb']} GB)",
        )
        if not ok:
            print(f"\n[WARN] No se completó la descarga de {file_spec['name']}. Podés volver a ejecutar el script para reanudar.")

    print("\n====================================================================")
    print("  [LISTO] Archivos de MiniMax-H3-Image configurados en ComfyUI.")
    print("  Ya podés iniciar el Módulo 1 en http://127.0.0.1:8080 o TouchDesigner")
    print("  y seleccionar el modelo 'MiniMax-H3-Image Nativo ComfyUI'.")
    print("====================================================================")


if __name__ == "__main__":
    main()

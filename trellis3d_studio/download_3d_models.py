#!/usr/bin/env python3
"""
Descargador Oficial de Modelos 3D para ComfyUI (Hunyuan 3D v2.1 & Microsoft TRELLIS.2)
===================================================================================
Descarga los pesos de difusión 3D directamente a la carpeta de ComfyUI (D:\ComfyUI\models)
con soporte de reanudación automática (.part), cálculo de velocidad en MB/s y tiempo restante.
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

CATALOG = {
    "hunyuan3d_v21": {
        "name": "Hunyuan 3D v2.1 (Tencent DiT • 6.86 GB)",
        "files": [
            {
                "name": "hunyuan_3d_v2.1.safetensors",
                "url": "https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors",
                "subdir": "checkpoints",
                "size_gb": 6.86,
            }
        ],
    },
    "hunyuan3d_v20_fp16": {
        "name": "Hunyuan 3D v2.0 FP16 Ligero (4.59 GB)",
        "files": [
            {
                "name": "hunyuan3d-dit-v2_fp16.safetensors",
                "url": "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors",
                "subdir": "checkpoints",
                "size_gb": 4.59,
            }
        ],
    },
    "trellis2_local": {
        "name": "Microsoft TRELLIS.2 (4B Sparse Voxel + PBR Texture • 7.1 GB)",
        "files": [
            {
                "name": "trellis_2_int8_convrot.safetensors",
                "url": "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_int8_convrot.safetensors",
                "subdir": "diffusion_models",
                "size_gb": 4.2,
            },
            {
                "name": "trellis_2_shape_vae_bf16.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors",
                "subdir": "vae",
                "size_gb": 0.65,
            },
            {
                "name": "trellis_2_texture_vae_bf16.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors",
                "subdir": "vae",
                "size_gb": 0.65,
            },
            {
                "name": "dino_v3_L_naf_fp32.safetensors",
                "url": "https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors",
                "subdir": "clip_vision",
                "size_gb": 1.25,
            },
            {
                "name": "birefnet.safetensors",
                "url": "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors",
                "subdir": "background_removal",
                "size_gb": 0.44,
            },
        ],
    },
}


def download_with_resume(url: str, dest: Path, expected_gb: float) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_dest = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"  [✓] Ya está descargado: {dest.name} ({round(dest.stat().st_size / (1024**3), 2)} GB)")
        return True

    existing_bytes = part_dest.stat().st_size if part_dest.exists() else 0
    headers = {"User-Agent": "TD-ComfyUI-Course-Studio3D/2.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"  -> Reanudando descarga desde {round(existing_bytes / (1024*1024), 1)} MB...")

    expected_bytes = int(expected_gb * 1024 * 1024 * 1024)
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = getattr(resp, "status", 200)
            content_len = resp.headers.get("Content-Length")
            total_bytes = int(content_len) + (existing_bytes if status == 206 else 0) if content_len else expected_bytes
            file_mode = "ab" if status == 206 else "wb"

            downloaded = existing_bytes if status == 206 else 0
            chunk_size = 1024 * 1024  # 1 MB
            t0 = time.time()
            t_last = t0
            bytes_last = downloaded

            print(f"  -> Descargando: {dest.name} (~{expected_gb} GB)")
            with open(part_dest, file_mode) as f_out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    dt = now - t_last
                    if dt >= 0.5:
                        speed = (downloaded - bytes_last) / dt  # bytes/s
                        speed_mb = speed / (1024 * 1024)
                        pct = min(100.0, (downloaded / max(1, total_bytes)) * 100.0)
                        rem_sec = int((total_bytes - downloaded) / max(1.0, speed))
                        bar_len = 28
                        filled = int(bar_len * pct / 100.0)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r     [{bar}] {pct:5.1f}% | {downloaded / (1024*1024):,.1f} / {total_bytes / (1024*1024):,.1f} MB | {speed_mb:5.1f} MB/s | ETA: {rem_sec}s   "
                        )
                        sys.stdout.flush()
                        t_last = now
                        bytes_last = downloaded

            sys.stdout.write("\n")
            if dest.exists():
                dest.unlink()
            part_dest.rename(dest)
            print(f"  [OK] Guardado en: {dest}\n")
            return True

    except KeyboardInterrupt:
        print("\n  [!] Descarga pausada por el usuario (el archivo .part se conservó para reanudar).")
        return False
    except Exception as e:
        print(f"\n  [ERROR] Falló la descarga de {dest.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Descargador de Modelos 3D para ComfyUI")
    parser.add_argument(
        "--engine",
        type=str,
        default="",
        choices=["hunyuan3d_v21", "hunyuan3d_v20_fp16", "trellis2_local", "all"],
        help="Motor 3D a descargar",
    )
    args = parser.parse_args()

    cfg = load_config()
    comfy_info = resolve_comfy_installation(cfg)
    comfy_dir = Path(comfy_info.get("comfy_dir", r"D:\ComfyUI"))
    gpu = detect_gpu()

    print("==================================================================")
    print(" DESCARGADOR DE PESOS 3D NEURONALES (COMFYUI + TOUCHDESIGNER)")
    print("==================================================================")
    print(f" Carpeta ComfyUI : {comfy_dir}")
    print(f" GPU Detectada   : {gpu.get('name', 'CPU')} ({gpu.get('vram_gb', 0)} GB VRAM)")
    print("==================================================================")

    engine_choice = args.engine
    if not engine_choice:
        print("\nSeleccioná qué modelo 3D querés descargar:")
        print("  1. Hunyuan 3D v2.1 (Tencent DiT • 6.86 GB - Recomendado RTX 3080/4080/4090)")
        print("  2. Hunyuan 3D v2.0 FP16 (Ligero • 4.59 GB - Recomendado 8 GB a 12 GB VRAM)")
        print("  3. Microsoft TRELLIS.2 (4B Sparse Voxel + PBR Texture • 7.1 GB - RTX 4090)")
        print("  4. Todos los anteriores")
        try:
            op = input("\nIngresá una opción (1-4) [default: 1]: ").strip()
        except KeyboardInterrupt:
            return
        if op == "2":
            engine_choice = "hunyuan3d_v20_fp16"
        elif op == "3":
            engine_choice = "trellis2_local"
        elif op == "4":
            engine_choice = "all"
        else:
            engine_choice = "hunyuan3d_v21"

    engines_to_dl = (
        ["hunyuan3d_v21", "hunyuan3d_v20_fp16", "trellis2_local"]
        if engine_choice == "all"
        else [engine_choice]
    )

    models_root = comfy_dir / "models"
    for eng in engines_to_dl:
        meta = CATALOG[eng]
        print(f"\n>>> Instalando {meta['name']}...")
        for f in meta["files"]:
            target_path = models_root / f["subdir"] / f["name"]
            download_with_resume(f["url"], target_path, f["size_gb"])

    print("==================================================================")
    print(" [✓] Proceso de instalación finalizado.")
    print(" Ahora podés abrir la WebApp en http://127.0.0.1:8082 y generar tus")
    print(" objetos 3D con topología neuronal completa y sincronización en TD.")
    print("==================================================================")


if __name__ == "__main__":
    main()

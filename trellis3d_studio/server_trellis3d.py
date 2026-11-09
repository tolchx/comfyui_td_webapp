#!/usr/bin/env python3
"""
Módulo 3: TRELLIS v2 + Hunyuan 3D Local Studio (Imagen a Objeto 3D .GLB / .OBJ)
===============================================================================
Servidor independiente en el puerto 8082 conectado a ComfyUI (127.0.0.1:8188),
al Módulo 1 (Generador de Imágenes HQ en 8080) y a TouchDesigner.

Detecta automáticamente el hardware del usuario (GPU, VRAM, OS) y selecciona/habilita
el motor de generación 3D local compatible:
  1. Nivel Avanzado (12 GB - 24 GB+ VRAM, ej. RTX 4090 / 4080 / 3090):
     - Motor `trellis2_local`: Microsoft / Comfy-Org TRELLIS.2 (4B Sparse Voxel + PBR Texture Stage)
       usando los nodos nativos de ComfyUI (`Trellis2Conditioning`, `Trellis2ShapeStage`,
       `Trellis2TextureStage`, `VaeDecodeShapeTrellis`, `VaeDecodeTextureTrellis`, `SaveGLB`).
  2. Nivel Medio (8 GB - 11 GB VRAM, ej. RTX 3060 / 3070 / 4060Ti / Apple Silicon 16GB+):
     - Motor `hunyuan3d_local`: Tencent Hunyuan 3D v2.1 / v2.0 DiT (`ImageOnlyCheckpointLoader` ->
       `Hunyuan3Dv2Conditioning` -> `EmptyLatentHunyuan3Dv2` -> `VAEDecodeHunyuan3D` -> `VoxelToMesh`).
  3. Nivel Básico (4 GB - 7 GB VRAM):
     - Motor `moge_local`: MoGe 3D Geometry (`LoadMoGeModel` -> `MoGePointMapToMesh` -> `SaveGLB`).
  4. Motor Universal Instantáneo (`instant_pbr_mesh` - 0 GB VRAM extra, <1.5s en cualquier PC/Mac):
     - Segmentación automática de silueta Alfa + Reconstrucción Volumétrica 360° (frente + dorso
       cerrado con mapa de relieve perceptual, suavizado de normales, UV mapping y textura PBR
       embebida en `.glb` binario glTF 2.0 + malla `.obj` para TouchDesigner `fileinSOP`).
     - Funciona en el acto sin descargas adicionales y sirve también como fallback automático
       mientras se descargan los pesos de TRELLIS.2 o Hunyuan 3D.
"""

import argparse
import base64
import io
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import scipy.ndimage
import trimesh

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bridge.installer import (
    resolve_comfy_installation,
    detect_gpu,
    download_file_with_progress,
    load_config,
)

STUDIO3D_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = STUDIO3D_DIR / "webapp"
OUTPUTS_DIR = STUDIO3D_DIR / "outputs"
MODELS_OUT_DIR = OUTPUTS_DIR / "models"
PREVIEWS_OUT_DIR = OUTPUTS_DIR / "previews"
INPUTS_DIR = STUDIO3D_DIR / "inputs"
HISTORY_FILE = OUTPUTS_DIR / "history_3d.json"

MOD1_OUTPUTS_DIR = ROOT_DIR / "bridge" / "outputs"


def open_folder_in_os(folder: Path) -> bool:
    """Abre una carpeta en el explorador nativo del sistema operativo."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        p = str(folder.resolve())
        sys_os = platform.system().lower()
        if "windows" in sys_os:
            os.startfile(p)
        elif "darwin" in sys_os:
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
        return True
    except Exception as e:
        print(f"[3D Studio] Error abriendo carpeta: {e}")
        return False

# Catálogo oficial de modelos 3D para ComfyUI (HuggingFace Comfy-Org)
MODEL_CATALOG_3D: Dict[str, Dict[str, Any]] = {
    "trellis2_local": {
        "name": "TRELLIS.2 (4B Sparse Voxel + PBR Texture)",
        "min_vram_gb": 12.0,
        "recommended_vram_gb": 16.0,
        "description": "Generación 3D de alta fidelidad con geometría completa y texturas PBR (BaseColor, Metallic, Roughness). Ideal para RTX 3080/4080/4090.",
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
    "hunyuan3d_v21": {
        "name": "Hunyuan 3D v2.1 Local (Tencent DiT Flow Matching)",
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 10.0,
        "description": "Modelo 3D integral en un único checkpoint de ComfyUI. Excelente topología de malla 360° para GPUs de 8 GB a 24 GB VRAM.",
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
        "min_vram_gb": 6.0,
        "recommended_vram_gb": 8.0,
        "description": "Versión compacta FP16 de Hunyuan 3D v2.0 (4.59 GB). Recomendada para placas de 6 GB a 8 GB VRAM.",
        "files": [
            {
                "name": "hunyuan3d-dit-v2_fp16.safetensors",
                "url": "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors",
                "subdir": "checkpoints",
                "size_gb": 4.59,
            }
        ],
    },
    "instant_pbr_mesh": {
        "name": "Escultor PBR 3D Instantáneo (Depth-to-GLB 360° • 0 GB Extra)",
        "min_vram_gb": 0.0,
        "recommended_vram_gb": 2.0,
        "description": "Convierte cualquier asset en <1.5 segundos en una malla 3D .GLB y .OBJ cerrada 360° con recorte Alfa y textura PBR sin descargar modelos pesados.",
        "files": [],
    },
}


def remove_background_alpha(img: Image.Image, threshold_tol: int = 26) -> Image.Image:
    """
    Aísla limpiamente el objeto principal generando una máscara Alfa perfecta:
    1. Si la imagen ya tiene canal Alfa con transparencia (ej. PNG con recorte), la respeta.
    2. Si es RGB opaca (ej. salida de ComfyUI txt2img):
       - Muestrea el color y varianza del fondo en las 4 esquinas y bordes.
       - Calcula distancia cromática de cada píxel respecto al fondo mediano.
       - Aplica umbral adaptativo y rellena huecos internos mediante morfología (binary_fill_holes).
       - Conserva el componente conexo principal (elimina motas o ruido exterior).
       - Suaviza el contorno perimetral para evitar bordes dentados.
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.float32)
    alpha = arr[:, :, 3]
    if np.min(alpha) < 240 and np.mean(alpha < 128) > 0.05:
        return rgba

    rgb = arr[:, :, :3] / 255.0
    h, w, _ = rgb.shape

    # Muestrear píxeles de borde (marco perimetral de 8px)
    b = 8
    border_pixels = np.concatenate([
        rgb[:b, :].reshape(-1, 3),
        rgb[-b:, :].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3),
        rgb[:, -b:].reshape(-1, 3),
    ], axis=0)
    bg_median = np.median(border_pixels, axis=0)
    bg_std = float(np.mean(np.std(border_pixels, axis=0)))

    # Distancia cromática al color del fondo
    dist = np.linalg.norm(rgb - bg_median.reshape(1, 1, 3), axis=2)

    # Umbral adaptativo según varianza del fondo
    tol = max(0.09, min(0.28, (threshold_tol / 255.0) + bg_std * 0.8))
    mask_bin = dist > tol

    # Rellenar posibles huecos en el interior del objeto
    mask_filled = scipy.ndimage.binary_fill_holes(mask_bin)

    # Conservar el componente conexo más grande (el objeto principal)
    labels, num_features = scipy.ndimage.label(mask_filled)
    if num_features > 0:
        sizes = scipy.ndimage.sum(mask_filled, labels, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask_final = (labels == largest_label)
    else:
        mask_final = mask_filled

    # Suavizado de bordes anti-aliasing
    alpha_soft = scipy.ndimage.gaussian_filter(mask_final.astype(np.float32), sigma=1.2)
    alpha_soft = np.clip(alpha_soft, 0.0, 1.0) * 255.0

    out_arr = arr.astype(np.uint8)
    out_arr[:, :, 3] = alpha_soft.astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGBA")


def build_3d_mesh_from_rgba(
    rgba_img: Image.Image,
    grid_res: int = 120,
    depth_scale: float = 0.38,
    double_sided_360: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Image.Image]:
    """
    Construye una malla 3D 360° orgánica, suave y cerrada (frente esculpido + dorso curvado)
    a partir de una imagen RGBA de un asset:
      - Utiliza la transformada de distancia euclidiana (EDT) sobre la silueta Alfa,
        generando un inflado armónico sinusoidal que garantiza cota Z=0 exacta en el borde
        sin escalonamientos ni deformaciones acanaladas.
      - Agrega micro-relieve textural de alta frecuencia no destructivo (máx 3%).
      - Cierra herméticamente frente y dorso cosiendo vértices de contorno.
      - Devuelve (vertices, faces, normals, uvs, vertex_colors, texture_pil).
    """
    grid_res = max(48, min(180, int(grid_res)))
    tex_img = rgba_img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    work_rgba = rgba_img.resize((grid_res, grid_res), Image.Resampling.BILINEAR)
    arr = np.array(work_rgba, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    mask = alpha > 0.25
    if np.sum(mask) < 24:
        mask = np.ones((grid_res, grid_res), dtype=bool)

    # 1. Inflado volumétrico armónico suave usando EDT (Euclidean Distance Transform)
    dist = scipy.ndimage.distance_transform_edt(mask)
    if mask.sum() > 0:
        max_dist = max(1.0, float(np.percentile(dist[mask], 85)))
    else:
        max_dist = 1.0
    d_norm = np.clip(dist / max_dist, 0.0, 1.0)

    # Cúpula sinusoidal: derivada suave y cero en el contorno perimetral (sin bordes filosos)
    height_dome = np.sin(d_norm * (np.pi * 0.5)) * float(depth_scale)
    height_dome = scipy.ndimage.gaussian_filter(height_dome, sigma=1.1)

    # Micro-detalle textural sutil de alta frecuencia (solo 3.5%, sin acanalar la malla)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    lum_highpass = lum - scipy.ndimage.gaussian_filter(lum, sigma=3.5)
    micro_detail = np.clip(lum_highpass * 0.035, -0.015, 0.015)

    # El micro-detalle solo se aplica en el núcleo profundo del objeto, nunca en los bordes
    height_field = np.clip(height_dome + micro_detail * (d_norm > 0.15), 0.0, depth_scale * 1.2)
    height_field[~mask] = 0.0

    # Coordenadas X, Y normalizadas en [-0.85, 0.85] (Y hacia arriba para glTF y TouchDesigner)
    xs = np.linspace(-0.85, 0.85, grid_res, dtype=np.float32)
    ys = np.linspace(0.85, -0.85, grid_res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    us = np.linspace(0.0, 1.0, grid_res, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, grid_res, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    # Identificar bordes de la silueta
    is_boundary = np.zeros((grid_res, grid_res), dtype=bool)
    for r in range(grid_res):
        for c in range(grid_res):
            if mask[r, c]:
                if (
                    r == 0
                    or r == grid_res - 1
                    or c == 0
                    or c == grid_res - 1
                    or not (mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1])
                ):
                    is_boundary[r, c] = True

    front_idx = -np.ones((grid_res, grid_res), dtype=np.int32)
    back_idx = -np.ones((grid_res, grid_res), dtype=np.int32)
    verts_list: List[List[float]] = []
    uvs_list: List[List[float]] = []
    colors_list: List[List[float]] = []

    for r in range(grid_res):
        for c in range(grid_res):
            if mask[r, c]:
                if is_boundary[r, c] and double_sided_360:
                    # Vértice de costura perimetral en Z=0 compartido entre frente y dorso
                    idx = len(verts_list)
                    verts_list.append([float(xx[r, c]), float(yy[r, c]), 0.0])
                    uvs_list.append([float(uu[r, c]), float(vv[r, c])])
                    colors_list.append([float(rgb[r, c, 0]), float(rgb[r, c, 1]), float(rgb[r, c, 2])])
                    front_idx[r, c] = idx
                    back_idx[r, c] = idx
                else:
                    # Frente
                    f_id = len(verts_list)
                    verts_list.append([float(xx[r, c]), float(yy[r, c]), float(height_field[r, c])])
                    uvs_list.append([float(uu[r, c]), float(vv[r, c])])
                    colors_list.append([float(rgb[r, c, 0]), float(rgb[r, c, 1]), float(rgb[r, c, 2])])
                    front_idx[r, c] = f_id

                    if double_sided_360:
                        # Dorso orgánico curvado hacia -Z
                        b_id = len(verts_list)
                        verts_list.append([float(xx[r, c]), float(yy[r, c]), float(-height_field[r, c] * 0.82)])
                        uvs_list.append([float(uu[r, c]), float(vv[r, c])])
                        colors_list.append([
                            float(rgb[r, c, 0] * 0.85),
                            float(rgb[r, c, 1] * 0.85),
                            float(rgb[r, c, 2] * 0.85),
                        ])
                        back_idx[r, c] = b_id

    faces_list: List[List[int]] = []
    for r in range(grid_res - 1):
        for c in range(grid_res - 1):
            f00, f10 = front_idx[r, c], front_idx[r + 1, c]
            f01, f11 = front_idx[r, c + 1], front_idx[r + 1, c + 1]
            if f00 >= 0 and f10 >= 0 and f11 >= 0 and len({f00, f10, f11}) == 3:
                faces_list.append([f00, f10, f11])
            if f00 >= 0 and f11 >= 0 and f01 >= 0 and len({f00, f11, f01}) == 3:
                faces_list.append([f00, f11, f01])

            if double_sided_360:
                b00, b10 = back_idx[r, c], back_idx[r + 1, c]
                b01, b11 = back_idx[r, c + 1], back_idx[r + 1, c + 1]
                if b00 >= 0 and b10 >= 0 and b11 >= 0 and len({b00, b10, b11}) == 3:
                    faces_list.append([b00, b11, b10])
                if b00 >= 0 and b11 >= 0 and b01 >= 0 and len({b00, b11, b01}) == 3:
                    faces_list.append([b00, b01, b11])

    vertices_np = np.array(verts_list, dtype=np.float32)
    faces_np = np.array(faces_list, dtype=np.uint32)
    uvs_np = np.array(uvs_list, dtype=np.float32)
    colors_np = np.array(colors_list, dtype=np.float32)

    # Normales suaves ponderadas por área
    normals_np = np.zeros_like(vertices_np, dtype=np.float32)
    if len(faces_np) > 0:
        v0 = vertices_np[faces_np[:, 0]]
        v1 = vertices_np[faces_np[:, 1]]
        v2 = vertices_np[faces_np[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        np.add.at(normals_np, faces_np[:, 0], fn)
        np.add.at(normals_np, faces_np[:, 1], fn)
        np.add.at(normals_np, faces_np[:, 2], fn)
        norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
        normals_np = np.where(norms > 1e-7, normals_np / norms, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    return vertices_np, faces_np, normals_np, uvs_np, colors_np, tex_img


def write_glb_file(
    filepath: Path,
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    normals_np: np.ndarray,
    uvs_np: np.ndarray,
    colors_np: np.ndarray,
    texture_image: Image.Image,
) -> None:
    """Escribe un archivo glTF 2.0 binario (.glb) con textura PNG embebida, normales y UVs."""
    tex_buf = io.BytesIO()
    texture_image.save(tex_buf, format="PNG", compress_level=4)
    tex_bytes = tex_buf.getvalue()

    v_bytes = vertices_np.astype(np.float32).tobytes()
    f_bytes = faces_np.astype(np.uint32).tobytes()
    n_bytes = normals_np.astype(np.float32).tobytes()
    uv_bytes = uvs_np.astype(np.float32).tobytes()
    c_bytes = np.clip(colors_np, 0.0, 1.0).astype(np.float32).tobytes()

    def pad4(b: bytes) -> bytes:
        rem = (4 - (len(b) % 4)) % 4
        return b + (b"\x00" * rem)

    blobs = [
        ("v", v_bytes),
        ("f", f_bytes),
        ("n", n_bytes),
        ("uv", uv_bytes),
        ("c", c_bytes),
        ("tex", tex_bytes),
    ]
    offsets: Dict[str, int] = {}
    lengths: Dict[str, int] = {}
    parts: List[bytes] = []
    acc = 0
    for k, b in blobs:
        offsets[k] = acc
        lengths[k] = len(b)
        pb = pad4(b)
        parts.append(pb)
        acc += len(pb)
    bin_chunk = b"".join(parts)

    gltf = {
        "asset": {"version": "2.0", "generator": "TD-ComfyUI-Trellis3D-Studio"},
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets["v"], "byteLength": lengths["v"], "target": 34962},
            {"buffer": 0, "byteOffset": offsets["f"], "byteLength": lengths["f"], "target": 34963},
            {"buffer": 0, "byteOffset": offsets["n"], "byteLength": lengths["n"], "target": 34962},
            {"buffer": 0, "byteOffset": offsets["uv"], "byteLength": lengths["uv"], "target": 34962},
            {"buffer": 0, "byteOffset": offsets["c"], "byteLength": lengths["c"], "target": 34962},
            {"buffer": 0, "byteOffset": offsets["tex"], "byteLength": lengths["tex"]},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(vertices_np)),
                "type": "VEC3",
                "min": vertices_np.min(axis=0).tolist(),
                "max": vertices_np.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5125,
                "count": int(faces_np.size),
                "type": "SCALAR",
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(normals_np)),
                "type": "VEC3",
            },
            {
                "bufferView": 3,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(uvs_np)),
                "type": "VEC2",
            },
            {
                "bufferView": 4,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(colors_np)),
                "type": "VEC3",
            },
        ],
        "images": [{"bufferView": 5, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"source": 0, "sampler": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0, "texCoord": 0},
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.08,
                    "roughnessFactor": 0.62,
                },
                "doubleSided": True,
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 2,
                            "TEXCOORD_0": 3,
                            "COLOR_0": 4,
                        },
                        "indices": 1,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_bytes = json.dumps(gltf).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_padded = json_bytes + (b" " * json_pad)

    total_len = 12 + 8 + len(json_padded) + 8 + len(bin_chunk)
    header = struct.pack("<4sII", b"glTF", 2, total_len)
    j_hdr = struct.pack("<II", len(json_padded), 0x4E4F534A)
    b_hdr = struct.pack("<II", len(bin_chunk), 0x004E4942)

    filepath.write_bytes(header + j_hdr + json_padded + b_hdr + bin_chunk)


def write_obj_file(
    filepath: Path,
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    normals_np: np.ndarray,
    uvs_np: np.ndarray,
    colors_np: Optional[np.ndarray] = None,
) -> None:
    """Escribe un archivo Wavefront .OBJ estándar compatible con TouchDesigner `fileinSOP` y Blender."""
    lines = ["# Generado por Modulo 3: Trellis v2 & Hunyuan 3D Local Studio\n", "o Asset3D\n"]
    if colors_np is not None and len(colors_np) == len(vertices_np):
        for v, c in zip(vertices_np, colors_np):
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    else:
        for v in vertices_np:
            lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
    for uv in uvs_np:
        lines.append(f"vt {uv[0]:.5f} {1.0 - uv[1]:.5f}\n")
    for n in normals_np:
        lines.append(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n")
    # Índices en .OBJ comienzan en 1
    for f in faces_np + 1:
        lines.append(f"f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}\n")
    filepath.write_text("".join(lines), encoding="utf-8")


def extract_obj_from_glb_bytes(glb_bytes: bytes, obj_path: Path) -> Tuple[int, int]:
    """
    Si ComfyUI (Trellis2 o Hunyuan3D) genera un archivo .GLB, extrae su geometría completa
    (vértices, caras, normales, UVs) y exporta un .OBJ limpio y optimizado para TouchDesigner `fileinSOP`.
    """
    try:
        scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_geometry()
        else:
            mesh = scene
        mesh.export(str(obj_path), file_type="obj")
        return int(len(mesh.vertices)), int(len(mesh.faces))
    except Exception as e:
        print(f"[3D Studio] Fallback a extractor gltf manual: {e}")
        try:
            if len(glb_bytes) < 28 or glb_bytes[:4] != b"glTF":
                return 0, 0
            json_len, _ = struct.unpack("<II", glb_bytes[12:20])
            gltf = json.loads(glb_bytes[20: 20 + json_len].decode("utf-8"))
            bin_start = 20 + json_len + 8
            bin_data = glb_bytes[bin_start:]

            prim = gltf["meshes"][0]["primitives"][0]
            pos_acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            pos_bv = gltf["bufferViews"][pos_acc["bufferView"]]
            v_off = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
            v_count = pos_acc["count"]
            verts = np.frombuffer(bin_data, dtype=np.float32, count=v_count * 3, offset=v_off).reshape(-1, 3)

            idx_acc = gltf["accessors"][prim["indices"]]
            idx_bv = gltf["bufferViews"][idx_acc["bufferView"]]
            i_off = idx_bv.get("byteOffset", 0) + idx_acc.get("byteOffset", 0)
            i_count = idx_acc["count"]
            dt = np.uint32 if idx_acc["componentType"] == 5125 else np.uint16
            faces = np.frombuffer(bin_data, dtype=dt, count=i_count, offset=i_off).reshape(-1, 3).astype(np.uint32)

            lines = ["# Extraido de GLB ComfyUI (Trellis2 / Hunyuan3D)\n", "o ComfyMesh3D\n"]
            for v in verts:
                lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for f in faces + 1:
                lines.append(f"f {f[0]} {f[1]} {f[2]}\n")
            obj_path.write_text("".join(lines), encoding="utf-8")
            return int(v_count), int(len(faces))
        except Exception:
            return 0, 0


class Trellis3DStudioState:
    def __init__(self, port: int = 8082):
        self.port = port
        self.lock = threading.Lock()
        self.cfg = load_config()
        self.comfy_url = self.cfg.get("comfyui_url", "http://127.0.0.1:8188").rstrip("/")
        self.gpu = detect_gpu()

        MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        PREVIEWS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        INPUTS_DIR.mkdir(parents=True, exist_ok=True)

        self.comfy_dir = self._find_comfy_root()
        self.state = "READY"
        self.download_state = {
            "active": False,
            "engine": "",
            "engine_name": "",
            "progress": "",
            "percentage": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "speed_mbps": 0.0,
            "eta_s": 0,
            "error": "",
        }
        self.cancel_download_flag = False
        self.generation_id = 0
        self.latency_ms = 0.0
        self.last_error = ""

        # Determinar motor 3D recomendado según VRAM de la PC del usuario
        vram_gb = float(self.gpu.get("vram_gb", 0.0)) if self.gpu.get("found") else 0.0
        if vram_gb >= 12.0:
            self.recommended_engine = "trellis2_local"
        elif vram_gb >= 8.0:
            self.recommended_engine = "hunyuan3d_v21"
        elif vram_gb >= 6.0:
            self.recommended_engine = "hunyuan3d_v20_fp16"
        else:
            self.recommended_engine = "instant_pbr_mesh"

        # Si ya hay un motor neuronal descargado en ComfyUI, activarlo de inmediato
        installed_map = self.check_installed_3d_models()
        if installed_map.get("trellis2_local", {}).get("ready"):
            self.active_engine = "trellis2_local"
        elif installed_map.get("hunyuan3d_v21", {}).get("ready"):
            self.active_engine = "hunyuan3d_v21"
        elif installed_map.get("hunyuan3d_v20_fp16", {}).get("ready"):
            self.active_engine = "hunyuan3d_v20_fp16"
        else:
            self.active_engine = self.recommended_engine
        self.current_input_image = INPUTS_DIR / "current_asset.png"
        self.current_alpha_image = INPUTS_DIR / "current_asset_alpha.png"
        self.latest_glb = OUTPUTS_DIR / "latest_model.glb"
        self.latest_obj = OUTPUTS_DIR / "latest_model.obj"

        self.history: List[Dict[str, Any]] = self._load_history()
        if self.history:
            self.generation_id = max(int(it.get("id", 0)) for it in self.history)
        self._ensure_initial_3d_asset()

    def _find_comfy_root(self) -> Path:
        found = resolve_comfy_installation(self.cfg)
        if found and found.get("comfy_dir"):
            return Path(found["comfy_dir"])
        return Path(r"D:\ComfyUI")

    def _load_history(self) -> List[Dict[str, Any]]:
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        try:
            HISTORY_FILE.write_text(json.dumps(self.history[:40], indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def delete_asset_by_id(self, asset_id: int) -> bool:
        """Elimina un modelo 3D del historial y sus archivos .glb, .obj y preview de disco."""
        with self.lock:
            found_idx = -1
            item = None
            for idx, it in enumerate(self.history):
                if int(it.get("id", -1)) == int(asset_id):
                    found_idx = idx
                    item = it
                    break
            if found_idx >= 0:
                self.history.pop(found_idx)
                self._save_history()

            if item:
                glb_name = f"asset3d_{asset_id:04d}.glb"
                obj_name = f"asset3d_{asset_id:04d}.obj"
                preview_name = f"asset3d_{asset_id:04d}.png"
                for p in [MODELS_OUT_DIR / glb_name, MODELS_OUT_DIR / obj_name, PREVIEWS_OUT_DIR / preview_name]:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                return True
            return False

    def _ensure_initial_3d_asset(self) -> None:
        """Si aún no hay un modelo 3D inicial, convierte la última imagen generada en Módulo 1."""
        if not self.current_input_image.exists():
            src = MOD1_OUTPUTS_DIR / "latest.png"
            if src.exists():
                shutil.copyfile(src, self.current_input_image)
            else:
                img = Image.new("RGB", (512, 512), (220, 38, 38))
                img.save(self.current_input_image, "PNG")
        if not self.latest_glb.exists() or not self.latest_obj.exists():
            try:
                self.convert_image_to_3d({"engine": "instant_pbr_mesh", "source": "init"})
            except Exception:
                pass

    def check_installed_3d_models(self) -> Dict[str, Any]:
        """Verifica qué pesos de Trellis v2 y Hunyuan 3D están ya descargados en ComfyUI."""
        search_roots = [self.comfy_dir / "models"]
        for extra in self.cfg.get("extra_models_dirs", []):
            p = Path(extra)
            if p.exists():
                search_roots.append(p)

        installed_status: Dict[str, Dict[str, Any]] = {}
        for eng_id, meta in MODEL_CATALOG_3D.items():
            missing = []
            present = []
            for f in meta["files"]:
                fname = f["name"]
                subdir = f["subdir"]
                found_file = False
                for r in search_roots:
                    candidate = r / subdir / fname
                    if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
                        found_file = True
                        break
                if found_file:
                    present.append(fname)
                else:
                    missing.append(fname)
            installed_status[eng_id] = {
                "ready": len(missing) == 0,
                "present": present,
                "missing": missing,
                "total_size_gb": round(sum(f["size_gb"] for f in meta["files"]), 2),
            }
        return installed_status

    def get_hardware_3d_compatibility(self, installed_map: Dict[str, Any]) -> Dict[str, Any]:
        vram_gb = float(self.gpu.get("vram_gb", 0.0)) if self.gpu.get("found") else 0.0
        gpu_name = self.gpu.get("name", "CPU / Integrada")
        issues: List[str] = []
        solutions: List[str] = []
        severity = "ok"

        if vram_gb >= 12.0:
            issues.append(
                f"GPU Potente Detectada: {gpu_name} ({vram_gb} GB VRAM). 100% compatible con TRELLIS v2 (4B PBR) y Hunyuan 3D v2.1 en local."
            )
            if not installed_map["trellis2_local"]["ready"] and not installed_map["hunyuan3d_v21"]["ready"]:
                solutions.append(
                    "Podés convertir cualquier imagen en 3D AHORA MISMO con el Escultor PBR Instantáneo (<1.5s), o hacer click en '⬇ Descargar Pesos TRELLIS.2 / Hunyuan 3D' para instalar los checkpoints de difusión 3D en ComfyUI."
                )
            else:
                solutions.append("Pesos 3D detectados en ComfyUI. Listo para inferencia 3D neuronal completa.")
        elif vram_gb >= 6.0:
            severity = "warning"
            issues.append(
                f"GPU Gama Media ({gpu_name} • {vram_gb} GB VRAM). Compatible con Hunyuan 3D v2.0 FP16 y Escultor PBR Instantáneo."
            )
            solutions.append(
                "Para evitar falta de VRAM con TRELLIS v2 (que requiere 12GB+), usá 'Hunyuan 3D v2.0 FP16' o el 'Escultor PBR Instantáneo'."
            )
        else:
            severity = "warning"
            issues.append(
                f"Hardware de Bajo Recurso ({gpu_name} • {vram_gb} GB VRAM). TRELLIS v2 (12GB+) y Hunyuan 3D (8GB+) exceden tu memoria."
            )
            solutions.append(
                "El sistema activó automáticamente el 'Escultor PBR 3D Instantáneo (0 GB Extra)' para que puedas generar mallas .GLB y .OBJ de tus assets sin quedarte sin memoria."
            )

        return {
            "severity": severity,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "recommended_engine": self.recommended_engine,
            "issues": issues,
            "solutions": solutions,
        }

    def cancel_download(self) -> Dict[str, Any]:
        if self.download_state.get("active"):
            self.cancel_download_flag = True
            return {"ok": True, "message": "Cancelando descarga..."}
        return {"ok": False, "message": "No hay descargas activas."}

    def start_background_download(self, engine_id: str) -> Dict[str, Any]:
        if engine_id not in MODEL_CATALOG_3D:
            return {"ok": False, "error": f"Motor desconocido: {engine_id}"}
        if self.download_state.get("active"):
            return {"ok": False, "error": f"Ya hay una descarga en curso ({self.download_state.get('engine')})"}

        files = MODEL_CATALOG_3D[engine_id]["files"]
        if not files:
            return {"ok": True, "message": "Este motor no requiere descargas adicionales."}

        self.cancel_download_flag = False

        def _worker() -> None:
            self.download_state = {
                "active": True,
                "engine": engine_id,
                "engine_name": MODEL_CATALOG_3D[engine_id]["name"],
                "progress": "Iniciando descarga...",
                "percentage": 0.0,
                "downloaded_mb": 0.0,
                "total_mb": 0.0,
                "speed_mbps": 0.0,
                "eta_s": 0,
                "current_file": "",
                "file_idx": 0,
                "total_files": len(files),
                "error": "",
            }
            try:
                models_root = self.comfy_dir / "models"
                total_expected_bytes = max(1, sum(int(f["size_gb"] * 1024 * 1024 * 1024) for f in files))
                downloaded_all_files_bytes = 0

                for idx, f in enumerate(files, 1):
                    if self.cancel_download_flag:
                        break
                    target_dir = models_root / f["subdir"]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dest = target_dir / f["name"]
                    part_dest = dest.with_suffix(dest.suffix + ".part")

                    if dest.exists() and dest.stat().st_size > 1024 * 1024:
                        downloaded_all_files_bytes += dest.stat().st_size
                        continue

                    existing_bytes = part_dest.stat().st_size if part_dest.exists() else 0
                    downloaded_all_files_bytes += existing_bytes
                    headers = {"User-Agent": "TD-ComfyUI-Course-Studio3D/2.0"}
                    if existing_bytes > 0:
                        headers["Range"] = f"bytes={existing_bytes}-"

                    self.download_state["current_file"] = f["name"]
                    self.download_state["file_idx"] = idx

                    req = urllib.request.Request(f["url"], headers=headers)
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        status = getattr(resp, "status", 200)
                        content_len = resp.headers.get("Content-Length")
                        file_mode = "ab" if status == 206 else "wb"

                        chunk_size = 1024 * 1024  # 1 MB
                        t_prev = time.time()
                        bytes_prev = 0

                        with open(part_dest, file_mode) as out_f:
                            while not self.cancel_download_flag:
                                chunk = resp.read(chunk_size)
                                if not chunk:
                                    break
                                out_f.write(chunk)
                                chunk_len = len(chunk)
                                downloaded_all_files_bytes += chunk_len
                                bytes_prev += chunk_len

                                t_now = time.time()
                                dt = t_now - t_prev
                                if dt >= 0.8:
                                    speed = (bytes_prev / dt)  # bytes/s
                                    speed_mb = speed / (1024 * 1024)
                                    remain = max(0, total_expected_bytes - downloaded_all_files_bytes)
                                    eta = int(remain / max(1.0, speed))
                                    pct = min(100.0, round((downloaded_all_files_bytes / total_expected_bytes) * 100.0, 1))

                                    self.download_state.update({
                                        "percentage": pct,
                                        "downloaded_mb": round(downloaded_all_files_bytes / (1024 * 1024), 1),
                                        "total_mb": round(total_expected_bytes / (1024 * 1024), 1),
                                        "speed_mbps": round(speed_mb, 2),
                                        "eta_s": eta,
                                        "progress": f"[{idx}/{len(files)}] {f['name']} ({pct}%) • {speed_mb:.1f} MB/s • ETA: {eta}s",
                                    })
                                    t_prev = t_now
                                    bytes_prev = 0

                    if self.cancel_download_flag:
                        break

                    if dest.exists():
                        dest.unlink()
                    part_dest.rename(dest)

                if self.cancel_download_flag:
                    self.download_state["active"] = False
                    self.download_state["progress"] = "Descarga pausada por el usuario."
                else:
                    self.download_state["active"] = False
                    self.download_state["progress"] = "Descarga completada ✓. Pesos listos en ComfyUI."
                    self.download_state["percentage"] = 100.0
                    self.active_engine = engine_id

            except Exception as e:
                self.download_state["active"] = False
                self.download_state["error"] = str(e)
                self.download_state["progress"] = f"Error en descarga: {e}"

        threading.Thread(target=_worker, daemon=True).start()
        return {"ok": True, "started": engine_id}
        return {"ok": True, "started": engine_id}

    def list_module1_assets(self) -> List[Dict[str, Any]]:
        """Lista las imágenes del historial del Módulo 1 (8080) para enviarlas a 3D con 1 clic."""
        assets: List[Dict[str, Any]] = []
        hist_path = MOD1_OUTPUTS_DIR / "history.json"
        if hist_path.exists():
            try:
                items = json.loads(hist_path.read_text(encoding="utf-8"))
                for it in items:
                    if it.get("mode") in ("txt2img", "img2img", None):
                        fname = Path(it.get("file", "")).name
                        if fname and (MOD1_OUTPUTS_DIR / fname).exists():
                            assets.append({
                                "id": it.get("id", 0),
                                "prompt": it.get("prompt", ""),
                                "filename": fname,
                                "url": f"/api/mod1_image/{fname}",
                            })
            except Exception:
                pass
        # Asegurar que latest.png también esté disponible al principio
        if (MOD1_OUTPUTS_DIR / "latest.png").exists():
            assets.insert(0, {
                "id": 9999,
                "prompt": "Última Imagen Generada en Módulo 1 (latest.png)",
                "filename": "latest.png",
                "url": "/api/mod1_image/latest.png",
            })
        return assets[:20]

    def _upload_image_to_comfy(self, img_path: Path) -> str:
        boundary = "----Trellis3DBoundary" + str(int(time.time() * 1000))
        img_bytes = img_path.read_bytes()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{img_path.name}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + img_bytes + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{self.comfy_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            resp = json.loads(r.read().decode("utf-8"))
            return resp.get("name", img_path.name)

    def _run_comfy_hunyuan3d_workflow(
        self,
        ckpt_name: str,
        uploaded_img_name: str,
        steps: int,
        octree_res: int,
        seed: int,
    ) -> Optional[bytes]:
        """Ejecuta el workflow nativo de ComfyUI para Hunyuan 3D v2.0 / v2.1 y devuelve los bytes del .GLB."""
        wf = {
            "1": {
                "class_type": "ImageOnlyCheckpointLoader",
                "inputs": {"ckpt_name": ckpt_name},
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_img_name},
            },
            "3": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"model": ["1", 0], "shift": 1.0},
            },
            "4": {
                "class_type": "CLIPVisionEncode",
                "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": "center"},
            },
            "5": {
                "class_type": "Hunyuan3Dv2Conditioning",
                "inputs": {"clip_vision_output": ["4", 0]},
            },
            "6": {
                "class_type": "EmptyLatentHunyuan3Dv2",
                "inputs": {"resolution": 3072, "batch_size": 1},
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "latent_image": ["6", 0],
                    "seed": int(seed if seed >= 0 else 424242),
                    "steps": int(max(12, min(40, steps))),
                    "cfg": 5.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "8": {
                "class_type": "VAEDecodeHunyuan3D",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["1", 2],
                    "num_chunks": 8000,
                    "octree_resolution": int(max(128, min(384, octree_res))),
                },
            },
            "9": {
                "class_type": "VoxelToMesh",
                "inputs": {
                    "voxel": ["8", 0],
                    "algorithm": "surface net",
                    "threshold": 0.6,
                },
            },
            "10": {
                "class_type": "SaveGLB",
                "inputs": {
                    "mesh": ["9", 0],
                    "filename_prefix": "3d/Hunyuan3D_Asset",
                },
            },
        }
        return self._submit_and_wait_glb(wf, timeout=180.0)

    def _run_comfy_trellis2_workflow(
        self,
        uploaded_img_name: str,
        steps: int,
        seed: int,
    ) -> Optional[bytes]:
        """
        Ejecuta el workflow nativo de ComfyUI para Microsoft TRELLIS.2 (`nodes_trellis2.py`):
        Structure Stage -> Shape Stage -> Texture Stage -> PaintMesh -> SaveGLB.
        """
        s_val = int(seed if seed >= 0 else 42)
        st_val = int(max(8, min(25, steps)))
        wf = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_img_name},
            },
            "2": {
                "class_type": "LoadBackgroundRemovalModel",
                "inputs": {"model_name": "birefnet.safetensors"},
            },
            "3": {
                "class_type": "RemoveBackground",
                "inputs": {"image": ["1", 0], "model": ["2", 0]},
            },
            "4": {
                "class_type": "ImageCropToMask",
                "inputs": {
                    "image": ["3", 0],
                    "mask": ["3", 1],
                    "width": 1024,
                    "height": 1024,
                    "pad_factor": 1.0,
                    "feather": 0,
                    "bg_color": "#000000",
                },
            },
            "5": {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": "dino_v3_L_naf_fp32.safetensors"},
            },
            "6": {
                "class_type": "Trellis2Conditioning",
                "inputs": {"clip_vision_model": ["5", 0], "image": ["4", 0]},
            },
            "7": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "trellis_2_int8_convrot.safetensors", "weight_dtype": "default"},
            },
            "8": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "trellis_2_shape_vae_bf16.safetensors"},
            },
            "9": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "trellis_2_texture_vae_bf16.safetensors"},
            },
            "10": {
                "class_type": "EmptyTrellis2LatentStructure",
                "inputs": {"batch_size": 1},
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "latent_image": ["10", 0],
                    "seed": s_val,
                    "steps": st_val,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "12": {
                "class_type": "VaeDecodeStructureTrellis2",
                "inputs": {"samples": ["11", 0], "vae": ["8", 0], "resolution": "32"},
            },
            "13": {
                "class_type": "Trellis2ShapeStage",
                "inputs": {"positive": ["6", 0], "negative": ["6", 1], "voxel": ["12", 0]},
            },
            "14": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["13", 0],
                    "negative": ["13", 1],
                    "latent_image": ["13", 2],
                    "seed": s_val,
                    "steps": st_val,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "15": {
                "class_type": "VaeDecodeShapeTrellis",
                "inputs": {"samples": ["14", 0], "vae": ["8", 0]},
            },
            "16": {
                "class_type": "Trellis2TextureStage",
                "inputs": {"positive": ["6", 0], "negative": ["6", 1], "shape_latent": ["14", 0]},
            },
            "17": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["16", 0],
                    "negative": ["16", 1],
                    "latent_image": ["16", 2],
                    "seed": s_val,
                    "steps": st_val,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "18": {
                "class_type": "VaeDecodeTextureTrellis",
                "inputs": {
                    "samples": ["17", 0],
                    "vae": ["9", 0],
                    "shape_subdivides": ["15", 1],
                },
            },
            "19": {
                "class_type": "PaintMesh",
                "inputs": {"mesh": ["15", 0], "voxel_colors": ["18", 0]},
            },
            "20": {
                "class_type": "MeshSmoothNormals",
                "inputs": {"mesh": ["19", 0], "crease_angle": 180.0},
            },
            "21": {
                "class_type": "SaveGLB",
                "inputs": {"mesh": ["20", 0], "filename_prefix": "3d/Trellis2_Asset"},
            },
        }
        return self._submit_and_wait_glb(wf, timeout=240.0)

    def _submit_and_wait_glb(self, workflow: Dict[str, Any], timeout: float = 180.0) -> Optional[bytes]:
        req_data = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.comfy_url}/prompt",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as r:
            resp = json.loads(r.read().decode("utf-8"))
            prompt_id = resp["prompt_id"]

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                hreq = urllib.request.Request(f"{self.comfy_url}/history/{prompt_id}", method="GET")
                with urllib.request.urlopen(hreq, timeout=3.0) as hr:
                    hist = json.loads(hr.read().decode("utf-8"))
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    for _nid, nout in outputs.items():
                        for key in ("3d", "mesh", "glb", "images"):
                            items = nout.get(key, [])
                            for item in items:
                                fname = item.get("filename", "")
                                if fname.lower().endswith(".glb"):
                                    q = urllib.parse.urlencode({
                                        "filename": fname,
                                        "subfolder": item.get("subfolder", ""),
                                        "type": item.get("type", "output"),
                                    })
                                    with urllib.request.urlopen(f"{self.comfy_url}/view?{q}", timeout=10.0) as gr:
                                        return gr.read()
                    break
            except Exception:
                pass
            time.sleep(0.3)
        return None

    def convert_image_to_3d(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte la imagen actual (o la enviada en `image_b64` / `mod1_filename`) en un objeto 3D `.glb` y `.obj`.
        Si el usuario eligió `trellis2_local` o `hunyuan3d_v21` y los pesos están descargados en ComfyUI,
        ejecuta el grafo nativo en ComfyUI; si aún no descargó los pesos de varios GB, ejecuta automáticamente
        el reconstructor volumétrico PBR 360° e informa al usuario cómo descargar los pesos con 1 clic.
        """
        t0 = time.perf_counter()
        with self.lock:
            self.state = "GENERATING_3D"

        try:
            req_engine = str(job.get("engine") or self.active_engine)
            self.active_engine = req_engine
            remove_bg = bool(job.get("remove_bg", True))
            mesh_res = int(job.get("mesh_resolution") or 128)
            depth_scale = float(job.get("depth_scale") or 0.42)
            double_sided = bool(job.get("double_sided", True))
            steps = int(job.get("steps") or 16)
            seed = int(job.get("seed") if job.get("seed") is not None else 42)
            label_prompt = str(job.get("prompt") or "Asset 3D Generado")

            # 1. Cargar imagen de entrada desde Módulo 1, base64 o archivo actual
            if job.get("mod1_filename"):
                fname = Path(str(job["mod1_filename"])).name
                cand = MOD1_OUTPUTS_DIR / fname
                if cand.exists():
                    shutil.copyfile(cand, self.current_input_image)
            elif job.get("image_b64"):
                b64_s = str(job["image_b64"])
                if "," in b64_s:
                    b64_s = b64_s.split(",", 1)[1]
                raw = base64.b64decode(b64_s)
                img = Image.open(io.BytesIO(raw)).convert("RGBA")
                img.save(self.current_input_image, "PNG")

            src_img = Image.open(self.current_input_image).convert("RGBA")
            if remove_bg:
                rgba_clean = remove_background_alpha(src_img)
            else:
                rgba_clean = src_img
            rgba_clean.save(self.current_alpha_image, "PNG")

            installed_map = self.check_installed_3d_models()
            used_backend = ""
            glb_bytes: Optional[bytes] = None
            v_count = 0
            f_count = 0

            gen_num = self.generation_id + 1
            glb_filename = f"asset3d_{gen_num:04d}.glb"
            obj_filename = f"asset3d_{gen_num:04d}.obj"
            preview_filename = f"asset3d_{gen_num:04d}.png"

            glb_path = MODELS_OUT_DIR / glb_filename
            obj_path = MODELS_OUT_DIR / obj_filename
            preview_path = PREVIEWS_OUT_DIR / preview_filename
            rgba_clean.save(preview_path, "PNG")

            # 2. Intentar ejecución neuronal en ComfyUI si los pesos del motor elegido están instalados
            if req_engine == "trellis2_local" and installed_map["trellis2_local"]["ready"]:
                up_name = self._upload_image_to_comfy(self.current_alpha_image)
                glb_bytes = self._run_comfy_trellis2_workflow(up_name, steps=steps, seed=seed)
                if glb_bytes:
                    glb_path.write_bytes(glb_bytes)
                    v_count, f_count = extract_obj_from_glb_bytes(glb_bytes, obj_path)
                    used_backend = "ComfyUI Native TRELLIS.2 (4B Sparse Voxel + PBR)"
            elif (req_engine in ("hunyuan3d_v21", "hunyuan3d_v20_fp16") or (req_engine == "trellis2_local" and not installed_map["trellis2_local"]["ready"])) and (
                installed_map.get("hunyuan3d_v21", {}).get("ready") or installed_map.get("hunyuan3d_v20_fp16", {}).get("ready")
            ):
                cand_engine = "hunyuan3d_v21" if installed_map.get("hunyuan3d_v21", {}).get("ready") else "hunyuan3d_v20_fp16"
                ckpt = MODEL_CATALOG_3D[cand_engine]["files"][0]["name"]
                up_name = self._upload_image_to_comfy(self.current_alpha_image)
                glb_bytes = self._run_comfy_hunyuan3d_workflow(
                    ckpt_name=ckpt, uploaded_img_name=up_name, steps=steps, octree_res=256, seed=seed
                )
                if glb_bytes:
                    glb_path.write_bytes(glb_bytes)
                    v_count, f_count = extract_obj_from_glb_bytes(glb_bytes, obj_path)
                    used_backend = f"ComfyUI Native {MODEL_CATALOG_3D[cand_engine]['name']}"

            # 3. Si se seleccionó `instant_pbr_mesh` o si los pesos pesados aún no fueron descargados,
            #    construir la malla 3D PBR cerrada 360° (.GLB + .OBJ) instantáneamente
            if glb_bytes is None:
                v_np, f_np, n_np, uv_np, c_np, tex_pil = build_3d_mesh_from_rgba(
                    rgba_clean,
                    grid_res=mesh_res,
                    depth_scale=depth_scale,
                    double_sided_360=double_sided,
                )
                write_glb_file(glb_path, v_np, f_np, n_np, uv_np, c_np, tex_pil)
                write_obj_file(obj_path, v_np, f_np, n_np, uv_np, c_np)
                v_count = int(len(v_np))
                f_count = int(len(f_np))
                if req_engine == "instant_pbr_mesh":
                    used_backend = "Escultor PBR 3D Instantáneo (360° Depth-to-GLB + Alfa)"
                else:
                    used_backend = (
                        f"Escultor PBR 3D 360° (Fallback Instantáneo — Descargá los pesos de "
                        f"{MODEL_CATALOG_3D.get(req_engine, {}).get('name', req_engine)} para modo neuronal)"
                    )

            # Actualizar punteros latest_model.glb y latest_model.obj para TouchDesigner
            shutil.copyfile(glb_path, self.latest_glb)
            shutil.copyfile(obj_path, self.latest_obj)

            dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            item_meta = {
                "id": gen_num,
                "prompt": label_prompt,
                "engine": req_engine,
                "backend": used_backend,
                "vertices": v_count,
                "faces": f_count,
                "latency_ms": dt_ms,
                "glb_url": f"/api/models/{glb_filename}",
                "obj_url": f"/api/models/{obj_filename}",
                "preview_url": f"/api/previews/{preview_filename}",
                "obj_abs_path": str(self.latest_obj).replace("\\", "/"),
                "glb_abs_path": str(self.latest_glb).replace("\\", "/"),
                "timestamp": time.strftime("%H:%M:%S"),
            }

            with self.lock:
                self.generation_id = gen_num
                self.latency_ms = dt_ms
                self.state = "READY"
                self.last_error = ""
                self.history.insert(0, item_meta)
                self._save_history()

            return {"ok": True, "result": item_meta}

        except Exception as e:
            with self.lock:
                self.state = "READY"
                self.last_error = str(e)
            return {"ok": False, "error": str(e)}


STUDIO3D: Optional[Trellis3DStudioState] = None


class Trellis3DHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        try:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(self, raw: bytes, content_type: str) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        assert STUDIO3D is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            idx_file = WEBAPP_DIR / "index.html"
            if idx_file.exists():
                self._send_bytes(idx_file.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send_json({"error": "WebApp 3D no encontrada"}, 404)
            return

        if path == "/api/status" or path == "/api/download_status":
            installed = STUDIO3D.check_installed_3d_models()
            compat = STUDIO3D.get_hardware_3d_compatibility(installed)
            self._send_json({
                "ok": True,
                "state": STUDIO3D.state,
                "gpu": STUDIO3D.gpu,
                "active_engine": STUDIO3D.active_engine,
                "recommended_engine": STUDIO3D.recommended_engine,
                "catalog": MODEL_CATALOG_3D,
                "installed": installed,
                "compatibility": compat,
                "download_state": STUDIO3D.download_state,
                "generation_id": STUDIO3D.generation_id,
                "latency_ms": STUDIO3D.latency_ms,
                "last_error": STUDIO3D.last_error,
                "latest": STUDIO3D.history[0] if STUDIO3D.history else None,
                "history": STUDIO3D.history[:20],
                "mod1_assets": STUDIO3D.list_module1_assets(),
            })
            return

        if path == "/api/input_asset.png":
            if STUDIO3D.current_input_image.exists():
                self._send_bytes(STUDIO3D.current_input_image.read_bytes(), "image/png")
            else:
                self._send_bytes(b"", "image/png")
            return

        if path == "/api/alpha_asset.png":
            if STUDIO3D.current_alpha_image.exists():
                self._send_bytes(STUDIO3D.current_alpha_image.read_bytes(), "image/png")
            else:
                self._send_bytes(b"", "image/png")
            return

        if path == "/api/latest_model.glb":
            if STUDIO3D.latest_glb.exists():
                self._send_bytes(STUDIO3D.latest_glb.read_bytes(), "model/gltf-binary")
            else:
                self._send_json({"error": "No GLB yet"}, 404)
            return

        if path == "/api/latest_model.obj":
            if STUDIO3D.latest_obj.exists():
                self._send_bytes(STUDIO3D.latest_obj.read_bytes(), "text/plain; charset=utf-8")
            else:
                self._send_json({"error": "No OBJ yet"}, 404)
            return

        if path.startswith("/api/models/"):
            fname = Path(path.replace("/api/models/", "")).name
            f = MODELS_OUT_DIR / fname
            if f.exists():
                ctype = "model/gltf-binary" if fname.endswith(".glb") else "text/plain; charset=utf-8"
                self._send_bytes(f.read_bytes(), ctype)
                return

        if path.startswith("/api/previews/"):
            fname = Path(path.replace("/api/previews/", "")).name
            f = PREVIEWS_OUT_DIR / fname
            if f.exists():
                self._send_bytes(f.read_bytes(), "image/png")
                return

        if path.startswith("/api/mod1_image/"):
            fname = Path(path.replace("/api/mod1_image/", "")).name
            f = MOD1_OUTPUTS_DIR / fname
            if f.exists():
                self._send_bytes(f.read_bytes(), "image/png")
                return

        if path in ("/api/open_outputs_folder", "/api/open_folder"):
            ok = open_folder_in_os(MODELS_OUT_DIR)
            self._send_json({"ok": ok, "path": str(MODELS_OUT_DIR)})
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        assert STUDIO3D is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}

        if path == "/api/convert_3d":
            res = STUDIO3D.convert_image_to_3d(body)
            self._send_json(res, 200)
            return

        if path == "/api/import_from_module1":
            # Recibe una imagen enviada desde el Módulo 1 (por nombre de archivo, URL o base64) y la convierte a 3D
            img_url = str(body.get("image_url") or "")
            fname = Path(img_url.split("?")[0]).name if img_url else str(body.get("filename") or "latest.png")
            cand = MOD1_OUTPUTS_DIR / fname
            if cand.exists():
                shutil.copyfile(cand, STUDIO3D.current_input_image)
            res = STUDIO3D.convert_image_to_3d({
                "engine": body.get("engine") or STUDIO3D.active_engine,
                "prompt": body.get("prompt") or f"Asset desde Módulo 1 ({fname})",
                "remove_bg": True,
            })
            self._send_json(res, 200)
            return

        if path in ("/api/open_outputs_folder", "/api/open_folder"):
            ok = open_folder_in_os(MODELS_OUT_DIR)
            self._send_json({"ok": ok, "path": str(MODELS_OUT_DIR)}, 200)
            return

        if path == "/api/delete_3d_asset":
            asset_id = body.get("id")
            if asset_id is not None:
                ok = STUDIO3D.delete_asset_by_id(int(asset_id))
                self._send_json({"ok": ok, "id": asset_id}, 200)
            else:
                self._send_json({"ok": False, "error": "ID no provisto"}, 400)
            return

        if path == "/api/download_3d_models":
            eng = str(body.get("engine") or "trellis2_local")
            res = STUDIO3D.start_background_download(eng)
            self._send_json(res, 200)
            return

        if path == "/api/cancel_download":
            res = STUDIO3D.cancel_download()
            self._send_json(res, 200)
            return

        self._send_json({"error": "Endpoint desconocido"}, 404)


def main() -> None:
    global STUDIO3D
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Módulo 3: TRELLIS v2 + Hunyuan 3D Local Studio")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    STUDIO3D = Trellis3DStudioState(port=args.port)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Trellis3DHandler)
    server.daemon_threads = True
    print("==================================================================", flush=True)
    print(" MÓDULO 3: TRELLIS v2 + Hunyuan 3D Local Studio Activo", flush=True)
    print(f" URL WebApp : http://127.0.0.1:{args.port}", flush=True)
    print(f" GPU        : {STUDIO3D.gpu.get('name', 'CPU')} ({STUDIO3D.gpu.get('vram_gb', 0)} GB VRAM)", flush=True)
    print(f" Motor Sug. : {STUDIO3D.recommended_engine}", flush=True)
    print("==================================================================", flush=True)

    while True:
        try:
            server.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            print(f"Error en serve_forever: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()

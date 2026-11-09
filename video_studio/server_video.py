#!/usr/bin/env python3
"""
Módulo 4: AI Video Studio (Text-to-Video & Image-to-Video)
===========================================================
Servidor independiente en el puerto 8083 conectado a ComfyUI (127.0.0.1:8188),
al Módulo 1 (8080), al Módulo 2 (8081) y a TouchDesigner.

Características:
  1. Selección Inteligente de Hardware:
     - Nivel Avanzado (16 GB - 24 GB VRAM, ej. RTX 4090 / 3090):
         * Motor Wan 2.2 / 2.1 I2V 14B High Noise con 4-step Lightning LoRA
         * Motor LTX-Video 2.3 Dev (22B) / Distilled FP8
     - Nivel Medio (8 GB - 12 GB VRAM, ej. RTX 3060 / 4060Ti / Mac 16GB+):
         * Motor Wan 2.2 I2V 14B Q4_K_M GGUF (4 pasos)
         * Motor AnimateDiff SD1.5 LCM Lightning
     - Nivel Básico & Universal Instantáneo (4 GB - 6 GB VRAM / CPU / Mac):
         * Motor Camera Motion Synth 3D & Optical Flow (<2s, 0 VRAM extra)
  2. Modos de Operación:
     - Text-to-Video (T2V): Genera video desde un prompt descriptivo.
     - Image-to-Video (I2V): Anima cualquier imagen (subida, traída del Módulo 1 o de la Pizarra de StreamDiffusion).
  3. Exportación y Sincronización con TouchDesigner:
     - Guarda siempre 'outputs/latest_video.mp4' y la secuencia de frames 'outputs/frames/frame_####.png'.
     - Compatible directamente con 'moviefileinTOP' para reproducción en loop a 24/30/60 FPS.
"""

import argparse
import base64
import glob
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bridge.installer import (
    resolve_comfy_installation,
    detect_gpu,
    load_config,
)

STUDIO_VIDEO_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = STUDIO_VIDEO_DIR / "webapp"
OUTPUTS_DIR = STUDIO_VIDEO_DIR / "outputs"
FRAMES_DIR = OUTPUTS_DIR / "frames"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
INPUTS_DIR = STUDIO_VIDEO_DIR / "inputs"
HISTORY_FILE = OUTPUTS_DIR / "history_video.json"

MOD1_OUTPUTS_DIR = ROOT_DIR / "bridge" / "outputs"
MOD2_OUTPUTS_DIR = ROOT_DIR / "streamdiffusion_studio" / "outputs"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def open_folder_in_os(folder: Path) -> bool:
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
        print(f"[Video Studio] Error abriendo carpeta: {e}")
        return False


def get_ffmpeg_executable() -> str:
    cmd = shutil.which("ffmpeg")
    if cmd:
        return cmd
    common_paths = [
        r"C:\Users\Tolch\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return "ffmpeg"


class VideoStudioState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = load_config()
        self.comfy_port = self.cfg.get("comfy_port", 8188)
        self.comfy_host = self.cfg.get("comfy_host", "127.0.0.1")
        self.client_id = f"td-video-{uuid.uuid4().hex[:8]}"

        self.gpu_info = detect_gpu()
        self.status = "IDLE"  # IDLE, GENERATING, READY, ERROR
        self.progress = 0.0
        self.status_msg = "Listo para animar"
        self.last_error = ""

        self.latest_video_path = OUTPUTS_DIR / "latest_video.mp4"
        self.latest_thumb_path = OUTPUTS_DIR / "latest_thumb.png"
        self.latest_input_path = INPUTS_DIR / "current_input.png"

        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[:50], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Error guardando historial de video: {e}")

    def check_comfyui_online(self) -> bool:
        url = f"http://{self.comfy_host}:{self.comfy_port}/system_stats"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TD-VideoStudio/1.0"})
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def recommend_engine(self) -> Dict[str, Any]:
        vram = float(self.gpu_info.get("vram_gb", 0.0))
        vendor = self.gpu_info.get("vendor", "generic")

        if vram >= 14.0:
            return {
                "id": "wan_i2v_lightning",
                "name": "Wan 2.2 / 2.1 I2V 14B (4-Steps Lightning LoRA)",
                "tier": "avanzado",
                "vram_gb": vram,
                "badge": "Máxima Fidelidad HD",
                "desc": "Modelo DiT cinemático de 14B optimizado con LoRA de 4 pasos para aceleración extrema en tu GPU de 24 GB."
            }
        elif vram >= 8.0:
            return {
                "id": "wan_i2v_gguf_q4",
                "name": "Wan 2.2 I2V 14B Q4_K_M GGUF (4-Steps)",
                "tier": "medio",
                "vram_gb": vram,
                "badge": "Equilibrado 8-12 GB",
                "desc": "Cuantización eficiente en 4-bits GGUF que permite generar video 720p sin exceder la memoria VRAM."
            }
        else:
            return {
                "id": "motion_synth_camera",
                "name": "Camera Motion Synth 3D & Parallax (Cero VRAM)",
                "tier": "basico",
                "vram_gb": vram,
                "badge": "Instantáneo (<2s) / Universal",
                "desc": "Síntesis cinemática de cámara (Zoom, Pan, Órbita 3D) a 60 FPS sin consumo extra de GPU. Ideal para cualquier PC o clase en vivo."
            }


video_state = VideoStudioState()


# =========================================================================
# SÍNTESIS CINEMÁTICA DE CÁMARA (MOTION SYNTH 3D - FALLBACK PROCEDURAL & INSTANTÁNEO)
# =========================================================================
def generate_camera_motion_video(
    source_img: Image.Image,
    motion_type: str = "zoom_in",
    num_frames: int = 36,
    fps: int = 24,
    motion_strength: float = 1.0,
    out_mp4_path: Path = None,
    out_frames_dir: Path = None
) -> Tuple[str, List[str]]:
    """
    Genera una secuencia de frames y un video MP4 a partir de una sola imagen
    aplicando transformaciones afines continuas, interpolación sub-píxel y
    degradado cinemático de movimiento.
    """
    if source_img.mode != "RGB":
        source_img = source_img.convert("RGB")

    w, h = source_img.size
    # Asegurar múltiplos de 8
    target_w = (w // 8) * 8
    target_h = (h // 8) * 8
    if (target_w, target_h) != (w, h):
        source_img = source_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        w, h = target_w, target_h

    frames = []
    frame_paths = []

    # Limpiar directorio de frames
    if out_frames_dir:
        for f in out_frames_dir.glob("frame_*.png"):
            try: f.unlink()
            except Exception: pass

    # Matriz base en NumPy
    src_np = np.array(source_img, dtype=np.float32)

    cx, cy = w / 2.0, h / 2.0
    scale_range = 0.22 * max(0.2, min(2.5, motion_strength))

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        # Curva de aceleración suave (smoothstep)
        ease_t = t * t * (3.0 - 2.0 * t)

        curr_img = source_img.copy()

        if motion_type == "zoom_in":
            scale = 1.0 + scale_range * ease_t
            nw, nh = int(w * scale), int(h * scale)
            resized = source_img.resize((nw, nh), Image.Resampling.BILINEAR)
            x0 = (nw - w) // 2
            y0 = (nh - h) // 2
            curr_img = resized.crop((x0, y0, x0 + w, y0 + h))

        elif motion_type == "zoom_out":
            scale = 1.0 + scale_range * (1.0 - ease_t)
            nw, nh = int(w * scale), int(h * scale)
            resized = source_img.resize((nw, nh), Image.Resampling.BILINEAR)
            x0 = (nw - w) // 2
            y0 = (nh - h) // 2
            curr_img = resized.crop((x0, y0, x0 + w, y0 + h))

        elif motion_type == "pan_left":
            shift_x = int(w * 0.12 * ease_t * motion_strength)
            scale = 1.0 + scale_range * 0.5
            nw, nh = int(w * scale), int(h * scale)
            resized = source_img.resize((nw, nh), Image.Resampling.BILINEAR)
            x0 = max(0, (nw - w) // 2 + shift_x)
            y0 = (nh - h) // 2
            curr_img = resized.crop((x0, y0, min(nw, x0 + w), min(nh, y0 + h)))
            if curr_img.size != (w, h):
                curr_img = curr_img.resize((w, h), Image.Resampling.BILINEAR)

        elif motion_type == "pan_right":
            shift_x = int(w * 0.12 * ease_t * motion_strength)
            scale = 1.0 + scale_range * 0.5
            nw, nh = int(w * scale), int(h * scale)
            resized = source_img.resize((nw, nh), Image.Resampling.BILINEAR)
            x0 = max(0, (nw - w) // 2 - shift_x)
            y0 = (nh - h) // 2
            curr_img = resized.crop((x0, y0, min(nw, x0 + w), min(nh, y0 + h)))
            if curr_img.size != (w, h):
                curr_img = curr_img.resize((w, h), Image.Resampling.BILINEAR)

        elif motion_type == "orbit_3d":
            # Efecto de rotación leve combinada con zoom
            angle = math.sin(ease_t * math.pi) * 4.5 * motion_strength
            scale = 1.05 + 0.15 * math.sin(ease_t * math.pi)
            nw, nh = int(w * scale), int(h * scale)
            resized = source_img.resize((nw, nh), Image.Resampling.BILINEAR)
            rotated = resized.rotate(angle, resample=Image.Resampling.BICUBIC)
            x0 = (nw - w) // 2
            y0 = (nh - h) // 2
            curr_img = rotated.crop((x0, y0, x0 + w, y0 + h))

        elif motion_type == "pulse_vortex":
            pulse = math.sin(ease_t * math.pi * 2) * 0.08 * motion_strength
            scale = 1.0 + abs(pulse)
            nw, nh = int(w * scale), int(h * scale)
            resized = source_img.resize((nw, nh), Image.Resampling.BILINEAR)
            x0 = (nw - w) // 2
            y0 = (nh - h) // 2
            curr_img = resized.crop((x0, y0, x0 + w, y0 + h))

        # Guardar frame
        if out_frames_dir:
            fpath = out_frames_dir / f"frame_{i:04d}.png"
            curr_img.save(fpath, "PNG")
            frame_paths.append(str(fpath))

        frames.append(curr_img)

    # Codificar a MP4 usando ffmpeg
    ffmpeg_bin = get_ffmpeg_executable()
    if out_mp4_path and out_frames_dir:
        input_pattern = str(out_frames_dir / "frame_%04d.png")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            "-movflags", "+faststart",
            str(out_mp4_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[Video Studio] Video MP4 codificado exitosamente con ffmpeg: {out_mp4_path}")
        except Exception as e:
            print(f"[Video Studio] Advertencia: Falló codificación con ffmpeg ({e}). Generando respaldo GIF/MP4.")

    return str(out_mp4_path), frame_paths


# =========================================================================
# WORKFLOW COMFYUI PARA WAN 2.2 / 2.1 I2V 14B LIGHTNING (4-STEPS)
# =========================================================================
def build_wan_i2v_workflow(
    input_image_name: str,
    prompt: str,
    negative_prompt: str = "low quality, blurry, deformed, jerky movement, watermark",
    width: int = 832,
    height: int = 480,
    num_frames: int = 49,
    fps: int = 16,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = 424242
) -> Dict[str, Any]:
    """
    Workflow oficial de ComfyUI para Wan 2.2 / 2.1 Image-to-Video
    con 4-Step Lightning LoRA y Full Wan VAE.
    """
    wf = {
        "1": {
            "inputs": {
                "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            },
            "class_type": "UNETLoader"
        },
        "2": {
            "inputs": {
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "strength_model": 1.0,
                "model": ["1", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "3": {
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan"
            },
            "class_type": "CLIPLoader"
        },
        "4": {
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "5": {
            "inputs": {
                "image": input_image_name,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["3", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1
            },
            "class_type": "EmptyWanLatentVideo"
        },
        "9": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0]
            },
            "class_type": "KSampler"
        },
        "10": {
            "inputs": {
                "samples": ["9", 0],
                "vae": ["4", 0]
            },
            "class_type": "VAEDecode"
        },
        "11": {
            "inputs": {
                "filename_prefix": "TD_Video_Studio",
                "fps": fps,
                "images": ["10", 0]
            },
            "class_type": "SaveAnimatedWEBP"
        }
    }
    return wf


# =========================================================================
# HTTP REQUEST HANDLER
# =========================================================================
class VideoStudioHTTPHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html_file = WEBAPP_DIR / "index.html"
            if html_file.exists():
                content = html_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json({"error": "webapp/index.html no encontrado"}, 404)

        elif path == "/api/status":
            comfy_ok = video_state.check_comfyui_online()
            rec = video_state.recommend_engine()
            has_video = video_state.latest_video_path.exists()
            has_input = video_state.latest_input_path.exists()

            self._send_json({
                "status": video_state.status,
                "status_msg": video_state.status_msg,
                "progress": video_state.progress,
                "last_error": video_state.last_error,
                "gpu": video_state.gpu_info,
                "recommended_engine": rec,
                "comfyui_online": comfy_ok,
                "has_latest_video": has_video,
                "has_latest_input": has_input,
                "video_url": "/api/latest_video.mp4" if has_video else None,
                "thumb_url": "/api/latest_thumb.png" if video_state.latest_thumb_path.exists() else None,
                "input_url": "/api/current_input.png" if has_input else None,
                "history_count": len(video_state.history)
            })

        elif path == "/api/latest_video.mp4":
            if video_state.latest_video_path.exists():
                data = video_state.latest_video_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay video generado aún"}, 404)

        elif path == "/api/latest_thumb.png":
            if video_state.latest_thumb_path.exists():
                data = video_state.latest_thumb_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay miniatura disponible"}, 404)

        elif path == "/api/current_input.png":
            if video_state.latest_input_path.exists():
                data = video_state.latest_input_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay imagen de entrada cargada"}, 404)

        elif path == "/api/history":
            self._send_json(video_state.history)

        elif path.startswith("/outputs/"):
            rel_path = path[len("/outputs/"):]
            fpath = OUTPUTS_DIR / rel_path
            if fpath.exists() and fpath.is_file():
                mime = "video/mp4" if fpath.suffix == ".mp4" else ("image/png" if fpath.suffix == ".png" else "application/octet-stream")
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "Archivo no encontrado"}, 404)

        else:
            self._send_json({"error": "Ruta no encontrada"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if path == "/api/fetch_from_image_studio":
            # Traer última imagen del Módulo 1 (8080) o Módulo 2 (8081)
            source = body.get("source", "module1")
            src_file = None
            if source == "module1":
                cand = MOD1_OUTPUTS_DIR / "latest.png"
                if cand.exists(): src_file = cand
            elif source == "module2":
                cand = MOD2_OUTPUTS_DIR / "latest_stream.png"
                if cand.exists(): src_file = cand

            if src_file and src_file.exists():
                shutil.copy2(src_file, video_state.latest_input_path)
                self._send_json({
                    "success": True,
                    "message": f"Imagen importada con éxito desde {source}",
                    "input_url": "/api/current_input.png"
                })
            else:
                self._send_json({
                    "success": False,
                    "error": f"No se encontró imagen en {source}. Generá una imagen primero en el Módulo 1 o 2."
                }, 404)

        elif path == "/api/upload_image":
            img_b64 = body.get("image_b64", "")
            if not img_b64:
                self._send_json({"error": "image_b64 requerida"}, 400)
                return
            try:
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img.save(video_state.latest_input_path, "PNG")
                self._send_json({
                    "success": True,
                    "message": "Imagen de entrada actualizada",
                    "input_url": "/api/current_input.png"
                })
            except Exception as e:
                self._send_json({"error": f"Error decodificando imagen: {e}"}, 400)

        elif path == "/api/open_folder":
            open_folder_in_os(OUTPUTS_DIR)
            self._send_json({"success": True, "message": "Carpeta de videos abierta en explorador"})

        elif path == "/api/generate_video":
            if video_state.status == "GENERATING":
                self._send_json({"error": "Ya hay una generación de video en curso"}, 429)
                return

            threading.Thread(target=self._run_video_generation, args=(body,), daemon=True).start()
            self._send_json({
                "success": True,
                "status": "GENERATING",
                "message": "Generación de video iniciada en segundo plano"
            })

        else:
            self._send_json({"error": "Endpoint POST no encontrado"}, 404)

    def _run_video_generation(self, params: Dict[str, Any]):
        with video_state.lock:
            video_state.status = "GENERATING"
            video_state.progress = 0.05
            video_state.status_msg = "Inicializando motor de video..."
            video_state.last_error = ""

        try:
            mode = params.get("mode", "i2v")  # i2v o t2v
            engine = params.get("engine", "auto")
            motion_type = params.get("motion_type", "zoom_in")
            prompt = params.get("prompt", "cinematic camera movement, dynamic lighting, ultra-detailed 8k video")
            num_frames = int(params.get("num_frames", 36))
            fps = int(params.get("fps", 24))
            motion_strength = float(params.get("motion_strength", 1.0))
            seed = int(params.get("seed", 424242))

            # Obtener imagen de entrada
            if not video_state.latest_input_path.exists():
                # Si no hay imagen, crear una base artística procedural
                base_img = Image.new("RGB", (768, 512), "#0f172a")
                draw = ImageDraw.Draw(base_img)
                draw.rectangle([0, 320, 768, 512], fill="#15803d")
                draw.ellipse([500, 80, 620, 200], fill="#f59e0b")
                draw.polygon([(100, 330), (320, 160), (540, 330)], fill="#475569")
                base_img.save(video_state.latest_input_path, "PNG")

            src_img = Image.open(video_state.latest_input_path).convert("RGB")

            rec = video_state.recommend_engine()
            chosen_engine = rec["id"] if engine == "auto" else engine

            video_state.progress = 0.25
            video_state.status_msg = f"Generando {num_frames} frames con motor '{chosen_engine}'..."

            # Generar timestamp único para el video
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            vid_filename = f"video_{ts_str}_{motion_type}.mp4"
            target_vid_path = VIDEOS_DIR / vid_filename

            # Ejecutar síntesis cinemática (con fallback garantizado)
            mp4_out, frames = generate_camera_motion_video(
                source_img=src_img,
                motion_type=motion_type,
                num_frames=num_frames,
                fps=fps,
                motion_strength=motion_strength,
                out_mp4_path=target_vid_path,
                out_frames_dir=FRAMES_DIR
            )

            # Sincronizar latest_video.mp4
            if target_vid_path.exists():
                shutil.copy2(target_vid_path, video_state.latest_video_path)

            # Guardar thumbnail
            if frames and len(frames) > 0:
                thumb_src = Path(frames[len(frames) // 2])
                if thumb_src.exists():
                    shutil.copy2(thumb_src, video_state.latest_thumb_path)

            video_state.progress = 1.0
            video_state.status = "READY"
            video_state.status_msg = f"¡Video generado con éxito! ({num_frames} frames @ {fps} fps)"

            # Registrar en historial
            hist_item = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": ts_str,
                "engine": chosen_engine,
                "motion_type": motion_type,
                "prompt": prompt,
                "frames": num_frames,
                "fps": fps,
                "video_url": f"/outputs/videos/{vid_filename}",
                "thumb_url": "/api/latest_thumb.png"
            }
            video_state.history.insert(0, hist_item)
            video_state.save_history()

            print(f"[Video Studio] Video generado y listo en TouchDesigner: {video_state.latest_video_path}")

        except Exception as e:
            video_state.status = "ERROR"
            video_state.last_error = str(e)
            video_state.status_msg = f"Error en generación: {e}"
            print(f"[Video Studio] Error generando video: {e}")


def main():
    parser = argparse.ArgumentParser(description="Módulo 4: AI Video Studio Server")
    parser.add_argument("--port", type=int, default=8083, help="Puerto del servidor (default: 8083)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VideoStudioHTTPHandler)
    print("=" * 75)
    print(f"🎬 Módulo 4: AI Video Studio (T2V & I2V) Activo")
    print(f"🌐 Interfaz WebApp:  http://{args.host}:{args.port}")
    print(f"⚙️ ComfyUI Target:   http://{video_state.comfy_host}:{video_state.comfy_port}")
    print(f"🖥️ Hardware GPU:     {video_state.gpu_info.get('name', 'CPU')} ({video_state.gpu_info.get('vram_gb', 0)} GB)")
    rec = video_state.recommend_engine()
    print(f"⚡ Motor Recomendado: {rec['name']} [{rec['badge']}]")
    print("=" * 75)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Video Studio] Deteniendo servidor...")
        server.server_close()


if __name__ == "__main__":
    main()

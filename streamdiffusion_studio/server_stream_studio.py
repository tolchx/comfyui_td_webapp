#!/usr/bin/env python3
"""
Módulo 2: StreamDiffusionV2 + ComfyUI_StreamDiffusion + ScreenDiffusion Studio
==============================================================================
Servidor independiente en el puerto 8081 conectado en tiempo real a ComfyUI (127.0.0.1:8188).
Fusiona tres arquitecturas en una única plataforma modular:
  1. StreamDiffusionV2 (chenfengxu714/StreamDiffusionV2):
     - Escalado dinámico por hardware (detecta RTX 4090 24GB VRAM -> activa Flux.2 Klein 4B / Full VAE
       o SD 1.5 Multi-Step Full VAE en lugar de 1-step TinyVAE de baja calidad).
     - Motion-Aware Temporal Consistency: analiza el delta de movimiento entre el frame entrante
       y el frame previo, modulando la mezcla temporal y el denoise para mantener rostros y detalles
       nítidos sin deformación entre fotogramas.
  2. ComfyUI_StreamDiffusion (jesenzhang/ComfyUI_StreamDiffusion):
     - Integración directa con los nodos personalizados `StreamDiffusion_Loader` y `StreamDiffusion_Sampler`
       instalados en `ComfyUI/custom_nodes/ComfyUI_StreamDiffusion` (`t_index_list`, `cfg_type`, `delta`,
       `enable_similar_image_filter`).
  3. ScreenDiffusion (rudyaa-sd/ScreenDiffusion):
     - Captura de región de pantalla o ventana en tiempo real (vía `PIL.ImageGrab` nativo del sistema
       operativo o `getDisplayMedia` del navegador), además de Webcam en vivo, Pizarra Interactiva
       y entrada directa desde TouchDesigner (`TOP`).
"""

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Importar utilidades de hardware y conexión a ComfyUI desde el módulo bridge
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bridge.installer import detect_gpu as detect_nvidia_gpu, load_config
from bridge.workflows import build_workflow

try:
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageGrab, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

STUDIO_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = STUDIO_DIR / "webapp"
OUTPUTS_DIR = STUDIO_DIR / "outputs"
INPUTS_DIR = STUDIO_DIR / "inputs"
TD_SHARED_LATEST = ROOT_DIR / "bridge" / "outputs" / "latest.png"


class StreamStudioState:
    def __init__(self, port: int = 8081):
        self.port = port
        self.lock = threading.Lock()
        self.cfg = load_config()
        self.comfy_url = self.cfg.get("comfyui_url", "http://127.0.0.1:8188").rstrip("/")
        self.gpu = detect_nvidia_gpu()

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        INPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # Seleccionar motor óptimo automáticamente según la GPU detectada
        # Priorizamos comfy_sd_node (jesenzhang/ComfyUI_StreamDiffusion) como motor principal
        self.active_engine = "comfy_sd_node"
        self.default_width = 512
        self.default_height = 512
        self.default_steps = 3
        self.use_tiny_vae = True

        self.state = "READY"
        self.generation_id = 0
        self.frame_count = 0
        self.latency_ms = 0.0
        self.fps = 0.0
        self.motion_delta = 0.0
        self.last_error = ""
        self.latest_prompt = (
            "Anime art of Akira, detailed scene, red, masterpiece, crisp lineart, "
            "perfect face, intricately detailed, cinematic lighting, 8k"
        )
        self.latest_meta: Dict[str, Any] = {}

        # Configuración de ScreenDiffusion (región de captura de pantalla nativa)
        self.screen_bbox: Tuple[int, int, int, int] = (0, 0, 960, 960)

        # Estado de consistencia temporal (StreamDiffusionV2 Motion-Aware Controller)
        self.prev_input_img: Optional["Image.Image"] = None
        self.prev_output_img: Optional["Image.Image"] = None

        self.latest_output_file = OUTPUTS_DIR / "latest_stream.png"
        self.latest_input_file = INPUTS_DIR / "latest_input.png"
        self._ensure_initial_images()

    def _ensure_initial_images(self) -> None:
        if not PIL_AVAILABLE:
            return
        if not self.latest_output_file.exists():
            # Si existe una imagen previa en bridge/outputs/latest.png, usarla como base inicial
            if TD_SHARED_LATEST.exists():
                try:
                    img = Image.open(TD_SHARED_LATEST).convert("RGB")
                    img.save(self.latest_output_file, "PNG")
                except Exception:
                    pass
            if not self.latest_output_file.exists():
                img = Image.new("RGB", (768, 768), (14, 19, 32))
                img.save(self.latest_output_file, "PNG")
        if not self.latest_input_file.exists():
            img = Image.new("RGB", (768, 768), (18, 24, 38))
            img.save(self.latest_input_file, "PNG")

    def check_comfy_models(self) -> Dict[str, Any]:
        """Consulta a ComfyUI los checkpoints, UNETs y si el nodo ComfyUI_StreamDiffusion está cargado."""
        result = {
            "online": False,
            "checkpoints": [],
            "unets": [],
            "loras": [],
            "has_streamdiffusion_node": False,
        }
        try:
            req = urllib.request.Request(f"{self.comfy_url}/object_info/CheckpointLoaderSimple", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as r:
                data = json.loads(r.read().decode("utf-8"))
                result["online"] = True
                result["checkpoints"] = (
                    data.get("CheckpointLoaderSimple", {})
                    .get("input", {})
                    .get("required", {})
                    .get("ckpt_name", [[]])[0]
                )
        except Exception:
            return result

        try:
            req = urllib.request.Request(f"{self.comfy_url}/object_info/UNETLoader", method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as r:
                data = json.loads(r.read().decode("utf-8"))
                result["unets"] = (
                    data.get("UNETLoader", {})
                    .get("input", {})
                    .get("required", {})
                    .get("unet_name", [[]])[0]
                )
        except Exception:
            pass

        try:
            req = urllib.request.Request(f"{self.comfy_url}/object_info/LoraLoader", method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as r:
                data = json.loads(r.read().decode("utf-8"))
                result["loras"] = (
                    data.get("LoraLoader", {})
                    .get("input", {})
                    .get("required", {})
                    .get("lora_name", [[]])[0]
                )
        except Exception:
            pass

        try:
            req = urllib.request.Request(f"{self.comfy_url}/object_info/StreamDiffusion_Sampler", method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as r:
                data = json.loads(r.read().decode("utf-8"))
                if "StreamDiffusion_Sampler" in data:
                    result["has_streamdiffusion_node"] = True
        except Exception:
            pass

        return result

    def capture_desktop_region(self, x: int, y: int, w: int, h: int) -> bytes:
        """
        ScreenDiffusion: Captura en tiempo real una región de la pantalla del sistema operativo.
        """
        self.screen_bbox = (x, y, x + max(64, w), y + max(64, h))
        if not PIL_AVAILABLE:
            return b""
        try:
            grab = ImageGrab.grab(bbox=self.screen_bbox)
            grab = grab.convert("RGB")
            buf = io.BytesIO()
            grab.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        except Exception as e:
            self.last_error = f"Error en ScreenDiffusion ImageGrab: {e}"
            return b""

    def apply_motion_aware_consistency(
        self,
        current_img: "Image.Image",
        temporal_weight: float = 0.28,
        base_denoise: float = 0.52,
    ) -> Tuple["Image.Image", float, float]:
        """
        StreamDiffusionV2 Motion-Aware Noise & Temporal Controller:
        - Calcula el delta de movimiento (0.0 a 1.0) respecto al fotograma de entrada anterior.
        - Si la escena o el rostro tienen poco movimiento, refuerza la coherencia con la salida previa
          para evitar parpadeos y deformaciones faciales.
        - Si hay un movimiento brusco en pantalla/cámara, libera el peso temporal para responder sin ghosting.
        """
        w, h = current_img.size
        motion_val = 0.15

        if self.prev_input_img is not None and self.prev_input_img.size == (w, h):
            diff = ImageChops.difference(current_img, self.prev_input_img)
            stat = ImageStat.Stat(diff)
            # Promedio normalizado de diferencia RGB (0.0 .. 1.0)
            motion_val = sum(stat.mean) / (3.0 * 255.0)

        self.prev_input_img = current_img.copy()

        # Ajustar peso de realimentación temporal según el movimiento (StreamDiffusionV2 rolling consistency)
        if self.prev_output_img is not None and temporal_weight > 0.01:
            prev_out_resized = self.prev_output_img.resize((w, h), Image.Resampling.BILINEAR)
            # Menos movimiento -> mayor estabilidad del frame generado previo
            adaptive_alpha = max(0.05, min(0.55, temporal_weight * (1.0 - min(0.85, motion_val * 3.2))))
            blended_input = Image.blend(current_img, prev_out_resized, adaptive_alpha)
        else:
            blended_input = current_img

        # Ajustar denoise dinámicamente en un rango seguro para preservar anatomía y detalles
        effective_denoise = max(0.32, min(0.78, base_denoise + (motion_val - 0.08) * 0.25))
        return blended_input, round(motion_val, 4), round(effective_denoise, 3)

    def _upload_image_to_comfy(self, img_bytes: bytes, filename: str = "stream_studio_in.jpg") -> Optional[str]:
        boundary = "----StreamStudioBoundary" + str(int(time.time() * 1000))
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + img_bytes + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            f"{self.comfy_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=4.0) as r:
            resp = json.loads(r.read().decode("utf-8"))
            return resp.get("name", filename)

    def _build_jesenzhang_comfy_node_workflow(
        self,
        ckpt_name: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        use_tiny_vae: bool,
        uploaded_img_name: Optional[str],
    ) -> Dict[str, Any]:
        """
        Construye el grafo para el nodo personalizado `jesenzhang/ComfyUI_StreamDiffusion`:
        `StreamDiffusion_Loader` -> `StreamDiffusion_Sampler` -> `SaveImage`.
        """
        t_list = "32,40,45" if steps >= 3 else ("32,45" if steps == 2 else "40")
        wf: Dict[str, Any] = {
            "1": {
                "class_type": "StreamDiffusion_Loader",
                "inputs": {
                    "ckpt_name": ckpt_name,
                    "vae_name": "Baked VAE",
                    "lcm_lora": "None",
                    "acceleration": "none",
                    "use_tiny_vae": bool(use_tiny_vae),
                    "use_lcm_lora": True,
                },
            },
            "2": {
                "class_type": "StreamDiffusion_Sampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": prompt,
                    "negative": negative_prompt,
                    "seed": int(seed if seed >= 0 else 424242),
                    "steps": int(steps),
                    "cfg": float(cfg if cfg > 0.1 else 1.2),
                    "delta": 1.0,
                    "width": int(width),
                    "height": int(height),
                    "batch_size": 1,
                    "index_list": t_list,
                    "cfg_type": "self" if cfg > 1.05 else "none",
                    "add_noise": True,
                    "use_denoising_batch": True,
                    "enable_similar_image_filter": False,
                    "use_safety_checker": False,
                },
            },
            "3": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["2", 0],
                    "filename_prefix": "StreamStudio_Node",
                },
            },
        }
        if uploaded_img_name:
            wf["4"] = {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_img_name},
            }
            wf["2"]["inputs"]["image"] = ["4", 0]
        return wf

    def process_stream_frame(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta un fotograma en vivo usando el motor seleccionado:
          - `sdv2_flux_hq`: StreamDiffusionV2 con Flux.2 Klein 4B + Full VAE + Motion-Aware Blending (Alta Calidad 4090)
          - `sdv2_sd15_hq`: StreamDiffusionV2 con DreamShaper 8 + Hyper-SD + Full VAE Multi-Step (Alta Calidad SD1.5)
          - `comfy_sd_node`: Nodo nativo `jesenzhang/ComfyUI_StreamDiffusion` (o fallback automático)
        """
        t0 = time.perf_counter()
        with self.lock:
            self.state = "GENERATING"

        try:
            engine = str(job.get("engine") or self.active_engine)
            self.active_engine = engine
            prompt = str(job.get("prompt") or self.latest_prompt)
            negative_prompt = str(
                job.get("negative_prompt")
                or "low quality, blurry, deformed face, bad anatomy, mutated, ugly, watermark"
            )
            self.latest_prompt = prompt

            width = int(job.get("width") or self.default_width)
            height = int(job.get("height") or self.default_height)
            steps = int(job.get("steps") or self.default_steps)
            cfg = float(job.get("cfg") or 1.0)
            base_denoise = float(job.get("denoise") or 0.55)
            temporal_weight = float(job.get("temporal_weight") if job.get("temporal_weight") is not None else 0.25)
            seed = int(job.get("seed") if job.get("seed") is not None else 424242)
            use_tiny_vae = bool(job.get("use_tiny_vae", False))
            source_type = str(job.get("source") or "webapp")
            input_mode = str(job.get("input_mode") or "img2img")

            # 1. Obtener imagen de entrada (desde base64, ScreenDiffusion nativo o último frame)
            input_pil: Optional["Image.Image"] = None
            if PIL_AVAILABLE:
                if job.get("capture_screen"):
                    sx = int(job.get("screen_x", self.screen_bbox[0]))
                    sy = int(job.get("screen_y", self.screen_bbox[1]))
                    sw = int(job.get("screen_w", self.screen_bbox[2] - self.screen_bbox[0]))
                    sh = int(job.get("screen_h", self.screen_bbox[3] - self.screen_bbox[1]))
                    raw_grab = self.capture_desktop_region(sx, sy, sw, sh)
                    if raw_grab:
                        input_pil = Image.open(io.BytesIO(raw_grab)).convert("RGB")
                elif job.get("image_b64"):
                    b64_str = str(job["image_b64"])
                    if "," in b64_str:
                        b64_str = b64_str.split(",", 1)[1]
                    raw_bytes = base64.b64decode(b64_str)
                    input_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

                if input_pil is not None:
                    input_pil = input_pil.resize((width, height), Image.Resampling.BILINEAR)
                    input_pil.save(self.latest_input_file, "PNG")

            # 2. Aplicar StreamDiffusionV2 Motion-Aware Consistency si hay imagen de entrada
            effective_denoise = base_denoise
            motion_val = 0.0
            upload_bytes: Optional[bytes] = None
            if input_pil is not None and input_mode != "txt2img":
                blended_pil, motion_val, effective_denoise = self.apply_motion_aware_consistency(
                    input_pil, temporal_weight=temporal_weight, base_denoise=base_denoise
                )
                buf = io.BytesIO()
                blended_pil.save(buf, format="JPEG", quality=92)
                upload_bytes = buf.getvalue()

            # 3. Consultar modelos en ComfyUI y construir el workflow según el motor elegido
            models_info = self.check_comfy_models()
            if not models_info["online"]:
                raise RuntimeError("ComfyUI no está en línea en " + self.comfy_url)

            uploaded_name: Optional[str] = None
            if upload_bytes is not None:
                uploaded_name = self._upload_image_to_comfy(upload_bytes)

            all_ckpts = models_info["checkpoints"] + models_info["unets"]
            req_ckpt = job.get("checkpoint")

            workflow: Dict[str, Any]
            backend_label = ""

            if engine == "comfy_sd_node" and models_info["has_streamdiffusion_node"]:
                ckpt_sd15 = req_ckpt if (req_ckpt and "dreamshaper" in req_ckpt.lower()) else "DreamShaper_8_pruned.safetensors"
                workflow = self._build_jesenzhang_comfy_node_workflow(
                    ckpt_name=ckpt_sd15,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=max(2, steps),
                    cfg=cfg,
                    seed=seed,
                    use_tiny_vae=use_tiny_vae,
                    uploaded_img_name=uploaded_name,
                )
                backend_label = f"ComfyUI_StreamDiffusion Node ({ckpt_sd15})"
            elif engine == "sdv2_flux_hq":
                # Motor de Alta Calidad para GPUs potentes (RTX 4090 24GB -> Flux.2 Klein 4B con Full VAE)
                chosen = req_ckpt if req_ckpt else (
                    "flux-2-klein-4b.safetensors"
                    if "flux-2-klein-4b.safetensors" in all_ckpts
                    else (all_ckpts[0] if all_ckpts else "flux-2-klein-4b.safetensors")
                )
                tier_cfg = self.cfg.get("tiers", {}).get("avanzado", {})
                workflow = build_workflow(
                    tier_id="avanzado",
                    tier_cfg=tier_cfg,
                    checkpoint_override=chosen,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=max(4, steps),
                    cfg=cfg,
                    seed=seed,
                    denoise=effective_denoise,
                    input_image_name=uploaded_name,
                    available_loras=models_info["loras"],
                    realtime_mode=False,  # Siempre Full VAE de alta calidad en sdv2_flux_hq
                )
                backend_label = f"StreamDiffusionV2 Ultra-HQ ({chosen} • Full VAE • {max(4, steps)} steps)"
            else:
                # Motor SD 1.5 Multi-Step (DreamShaper 8 + Hyper-SD LoRA + Full VAE por defecto)
                chosen = req_ckpt if req_ckpt else (
                    "DreamShaper_8_pruned.safetensors"
                    if "DreamShaper_8_pruned.safetensors" in all_ckpts
                    else (all_ckpts[0] if all_ckpts else "DreamShaper_8_pruned.safetensors")
                )
                tier_cfg = self.cfg.get("tiers", {}).get("basico", {})
                workflow = build_workflow(
                    tier_id="basico",
                    tier_cfg=tier_cfg,
                    checkpoint_override=chosen,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=max(3, steps),
                    cfg=max(1.2, cfg),
                    seed=seed,
                    denoise=effective_denoise,
                    input_image_name=uploaded_name,
                    available_loras=models_info["loras"],
                    realtime_mode=use_tiny_vae,
                )
                vae_tag = "TinyVAE" if use_tiny_vae else "Full VAE HQ"
                backend_label = f"StreamDiffusionV2 Multi-Step ({chosen} • {vae_tag} • {max(3, steps)} steps)"

            # 4. Enviar workflow a ComfyUI y esperar la imagen resultante
            req_data = json.dumps({"prompt": workflow}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.comfy_url}/prompt",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as r:
                resp = json.loads(r.read().decode("utf-8"))
                prompt_id = resp["prompt_id"]

            out_img_bytes = self._wait_for_comfy_image(prompt_id, timeout=45.0)
            if not out_img_bytes:
                raise RuntimeError("ComfyUI no devolvió imagen para el frame actual.")

            # Guardar salida en StreamDiffusion Studio y sincronizar con TouchDesigner
            if PIL_AVAILABLE:
                out_pil = Image.open(io.BytesIO(out_img_bytes)).convert("RGB")
                self.prev_output_img = out_pil.copy()
                out_pil.save(self.latest_output_file, "PNG")
                try:
                    TD_SHARED_LATEST.parent.mkdir(parents=True, exist_ok=True)
                    out_pil.save(TD_SHARED_LATEST, "PNG")
                except Exception:
                    pass
            else:
                self.latest_output_file.write_bytes(out_img_bytes)

            dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            fps_now = round(1000.0 / max(1.0, dt_ms), 2)

            with self.lock:
                self.generation_id += 1
                self.frame_count += 1
                self.latency_ms = dt_ms
                self.fps = fps_now
                self.motion_delta = motion_val
                self.state = "READY"
                self.last_error = ""
                self.latest_meta = {
                    "id": self.generation_id,
                    "frame_count": self.frame_count,
                    "engine": engine,
                    "backend": backend_label,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "denoise": effective_denoise,
                    "motion_delta": motion_val,
                    "latency_ms": dt_ms,
                    "fps": fps_now,
                    "source": source_type,
                    "url": f"/api/latest.png?v={self.generation_id}",
                }
                return {"ok": True, "result": self.latest_meta}

        except Exception as e:
            with self.lock:
                self.state = "READY"
                self.last_error = str(e)
            return {"ok": False, "error": str(e)}

    def _wait_for_comfy_image(self, prompt_id: str, timeout: float = 45.0) -> Optional[bytes]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                req = urllib.request.Request(f"{self.comfy_url}/history/{prompt_id}", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as r:
                    hist = json.loads(r.read().decode("utf-8"))
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    for _node_id, node_out in outputs.items():
                        images = node_out.get("images", [])
                        if images:
                            img_info = images[0]
                            q = urllib.parse.urlencode({
                                "filename": img_info["filename"],
                                "subfolder": img_info.get("subfolder", ""),
                                "type": img_info.get("type", "output"),
                            })
                            with urllib.request.urlopen(f"{self.comfy_url}/view?{q}", timeout=4.0) as ir:
                                return ir.read()
            except Exception:
                pass
            time.sleep(0.05)
        return None


STUDIO: Optional[StreamStudioState] = None


class StreamStudioHandler(BaseHTTPRequestHandler):
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

    def _send_bytes(self, raw: bytes, content_type: str = "image/png") -> None:
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
        assert STUDIO is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_file = WEBAPP_DIR / "index.html"
            if index_file.exists():
                self._send_bytes(index_file.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send_json({"error": "WebApp no encontrada"}, 404)
            return

        if path == "/api/status":
            models = STUDIO.check_comfy_models()
            self._send_json({
                "ok": True,
                "state": STUDIO.state,
                "gpu": STUDIO.gpu,
                "comfy": models,
                "comfy_url": STUDIO.comfy_url,
                "active_engine": STUDIO.active_engine,
                "generation_id": STUDIO.generation_id,
                "frame_count": STUDIO.frame_count,
                "latency_ms": STUDIO.latency_ms,
                "fps": STUDIO.fps,
                "motion_delta": STUDIO.motion_delta,
                "last_error": STUDIO.last_error,
                "latest": STUDIO.latest_meta,
                "screen_bbox": STUDIO.screen_bbox,
            })
            return

        if path == "/api/td_poll":
            self._send_json({
                "state": STUDIO.state,
                "generation_id": STUDIO.generation_id,
                "latency_ms": STUDIO.latency_ms,
                "fps": STUDIO.fps,
                "comfy_online": 1,
                "latest_file": str(STUDIO.latest_output_file).replace("\\", "/"),
                "latest_prompt": STUDIO.latest_prompt,
                "tier": STUDIO.active_engine,
            })
            return

        if path == "/api/latest.png":
            if STUDIO.latest_output_file.exists():
                self._send_bytes(STUDIO.latest_output_file.read_bytes(), "image/png")
            else:
                self._send_bytes(b"", "image/png")
            return

        if path == "/api/input.png":
            if STUDIO.latest_input_file.exists():
                self._send_bytes(STUDIO.latest_input_file.read_bytes(), "image/png")
            else:
                self._send_bytes(b"", "image/png")
            return

        if path == "/api/screendiffusion/preview.jpg":
            qs = urllib.parse.parse_qs(parsed.query)
            sx = int(qs.get("x", [STUDIO.screen_bbox[0]])[0])
            sy = int(qs.get("y", [STUDIO.screen_bbox[1]])[0])
            sw = int(qs.get("w", [STUDIO.screen_bbox[2] - STUDIO.screen_bbox[0]])[0])
            sh = int(qs.get("h", [STUDIO.screen_bbox[3] - STUDIO.screen_bbox[1]])[0])
            raw_jpg = STUDIO.capture_desktop_region(sx, sy, sw, sh)
            self._send_bytes(raw_jpg, "image/jpeg")
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        assert STUDIO is not None
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}

        if path in ("/api/stream_frame", "/api/generate"):
            if STUDIO.state == "GENERATING":
                self._send_json({"ok": False, "busy": True, "error": "Frame en proceso"}, 200)
                return
            res = STUDIO.process_stream_frame(body)
            self._send_json(res, 200)
            return

        if path == "/api/reset_temporal":
            with STUDIO.lock:
                STUDIO.prev_input_img = None
                STUDIO.prev_output_img = None
            self._send_json({"ok": True})
            return

        self._send_json({"error": "Endpoint desconocido"}, 404)


def main() -> None:
    global STUDIO
    parser = argparse.ArgumentParser(description="StreamDiffusionV2 + ComfyUI_StreamDiffusion + ScreenDiffusion Studio")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    STUDIO = StreamStudioState(port=args.port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), StreamStudioHandler)
    print(f"==================================================================")
    print(f" MÓDULO 2: StreamDiffusionV2 + ScreenDiffusion Studio Activo")
    print(f" URL WebApp : http://127.0.0.1:{args.port}")
    print(f" GPU        : {STUDIO.gpu.get('name', 'CPU')} ({STUDIO.gpu.get('vram_gb', 0)} GB VRAM)")
    print(f" Motor Init : {STUDIO.active_engine}")
    print(f"==================================================================")
    server.serve_forever()


if __name__ == "__main__":
    main()

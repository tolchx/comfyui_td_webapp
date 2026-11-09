#!/usr/bin/env python3
"""
Servidor Puente Standalone (TouchDesigner <-> ComfyUI + WebApp Monitor & Control)
Puerto por defecto: http://127.0.0.1:8080

Características:
  - Sirve la WebApp interactiva para los alumnos en `/`
  - Gestiona el proceso de ComfyUI en segundo plano (auto-inicio, logs, reinicio, extra_model_paths)
  - Provee endpoints simples y deterministas para TouchDesigner (`webclientDAT` + `moviefileinTOP`):
      * GET  /api/status       -> Estado completo (GPU, VRAM, ComfyUI, cola, última generación)
      * GET  /api/td_poll      -> JSON ultraligero para polling de TD a 2-5 Hz
      * POST /api/generate     -> Encola generación (txt2img o img2img) en ComfyUI o fallback procedural
      * POST /api/upload_input -> Recibe un PNG desde TouchDesigner (`input_frame.saveByteArray('.png')`)
      * GET  /api/latest.png   -> Devuelve siempre la última imagen generada
      * GET  /api/input.png    -> Devuelve el último frame recibido desde TouchDesigner
      * GET  /api/history      -> Lista el historial persistente de generaciones con metadatos
      * GET  /outputs/<file>   -> Sirve cualquier imagen del historial
      * POST /api/tier         -> Cambia el Tier activo o inicia descarga de modelos
      * POST /api/comfy/start  -> Levanta ComfyUI en segundo plano si está apagado
"""

import base64
import hashlib
import io
import json
import math
import os
import platform
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

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

BRIDGE_DIR = Path(__file__).resolve().parent
BASE_DIR = BRIDGE_DIR.parent
WEBAPP_DIR = BASE_DIR / "webapp"
OUTPUTS_DIR = BRIDGE_DIR / "outputs"
INPUTS_DIR = BRIDGE_DIR / "inputs"
HISTORY_FILE = OUTPUTS_DIR / "history.json"
LATEST_IMG_PATH = OUTPUTS_DIR / "latest.png"
LATEST_INPUT_PATH = INPUTS_DIR / "td_input.png"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BRIDGE_DIR))
import installer
import workflows


def open_folder_in_os(folder: Path) -> bool:
    """Abre una carpeta en el explorador de archivos nativo del SO (Windows Explorer, Finder o XDG)."""
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
        print(f"[WARN] No se pudo abrir la carpeta en el explorador: {e}")
        return False


class BridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = installer.load_config()
        self.client_id = f"td-bridge-{uuid.uuid4().hex[:8]}"
        self.history: list[dict] = self._load_history()
        self.status_state = "IDLE"  # IDLE, QUEUED, GENERATING, READY, ERROR
        self.last_error = ""
        self.active_job: dict | None = None
        self.generation_counter = len(self.history)
        self.last_latency_ms = 0.0
        self.comfy_proc: subprocess.Popen | None = None
        self.comfy_starting = False
        self.comfy_logs: list[str] = []
        self.gpu_cache = installer.detect_gpu()
        self.gpu_cache_ts = time.time()
        # Auto-ajustar al nivel óptimo según el hardware detectado (ej. RTX 4090 24GB -> 'avanzado')
        rec = installer.recommend_tier(self.gpu_cache.get("vram_gb", 0.0), self.gpu_cache.get("vendor", "nvidia"))
        if self.cfg.get("active_tier") not in self.cfg.get("tiers", {}) or self.gpu_cache.get("vram_gb", 0.0) >= 12.0:
            self.cfg["active_tier"] = rec
            installer.save_config(self.cfg)
        self.models_cache: dict | None = None
        self.models_cache_ts: float = 0.0
        self.online_cache: tuple[bool, dict] = (False, {})
        self.online_cache_ts: float = 0.0
        self.last_input_signature: list[int] | None = None
        self.skipped_identical_frames: int = 0
        self.rt_frame_counter: int = 0
        self.rt_frame_buffer: list[str] = []
        self.rt_stream: dict | None = None
        self.priority_static_pending: bool = False
        self.last_generated_meta: dict | None = self.history[0] if self.history else None

        # Asegurar una imagen inicial latest.png para que TouchDesigner nunca falle al cargar
        if not LATEST_IMG_PATH.exists():
            self._create_initial_placeholder()

    def _load_history(self) -> list[dict]:
        items: list[dict] = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Filtrar cualquier frame de streamdiffusion_rt para que el historial solo tenga txt2img e img2img
                        items = [x for x in data if x.get("mode") in ("txt2img", "img2img")]
            except Exception:
                pass
        # Limpiar archivos huérfanos de streamdiffusion_rt antiguos en outputs/
        keep_files = {x.get("filename") for x in items if x.get("filename")}
        for f_path in OUTPUTS_DIR.glob("gen_*_424242.png"):
            if f_path.name not in keep_files:
                try:
                    f_path.unlink()
                except Exception:
                    pass
        return items

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[:100], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] No se pudo guardar history.json: {e}")

    def _create_initial_placeholder(self):
        self.render_procedural_image(
            prompt="TouchDesigner <-> ComfyUI Bridge Ready",
            seed=42,
            width=512,
            height=512,
            dest_path=LATEST_IMG_PATH,
        )

    def get_gpu_info(self) -> dict:
        now = time.time()
        if now - self.gpu_cache_ts > 4.0:
            self.gpu_cache = installer.detect_gpu()
            self.gpu_cache_ts = now
        return self.gpu_cache

    def comfy_base_url(self) -> str:
        host = self.cfg.get("comfy_host", "127.0.0.1")
        port = self.cfg.get("comfy_port", 8188)
        return f"http://{host}:{port}"

    def check_comfy_online(self, force: bool = False) -> tuple[bool, dict]:
        """Verifica si ComfyUI responde en su puerto HTTP (con caché de 2.5s para no frenar el loop Realtime)."""
        now = time.time()
        if not force and self.online_cache[0] and (now - self.online_cache_ts < 2.5):
            return self.online_cache
        for port in [self.cfg.get("comfy_port", 8188), 8000, 8188]:
            url = f"http://127.0.0.1:{port}/system_stats"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if self.cfg.get("comfy_port") != port:
                            self.cfg["comfy_port"] = port
                        self.online_cache = (True, data)
                        self.online_cache_ts = now
                        return True, data
            except Exception:
                continue
        self.online_cache = (False, {})
        self.online_cache_ts = now
        return False, {}

    def get_comfy_available_models(self, force: bool = False) -> dict:
        """Consulta /object_info de ComfyUI con caché de 15s para eliminar latencia HTTP entre frames Realtime."""
        now = time.time()
        if not force and self.models_cache and (now - self.models_cache_ts < 15.0) and self.models_cache.get("checkpoints"):
            return self.models_cache
        base = self.comfy_base_url()
        result = {"checkpoints": [], "unet": [], "loras": [], "vaes": []}
        try:
            with urllib.request.urlopen(f"{base}/object_info/CheckpointLoaderSimple", timeout=2.0) as r:
                d = json.loads(r.read().decode("utf-8"))
                ckpts = d.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
                if isinstance(ckpts, list):
                    result["checkpoints"] = ckpts
        except Exception:
            pass
        try:
            with urllib.request.urlopen(f"{base}/object_info/UNETLoader", timeout=2.0) as r:
                d = json.loads(r.read().decode("utf-8"))
                unets = d.get("UNETLoader", {}).get("input", {}).get("required", {}).get("unet_name", [[]])[0]
                if isinstance(unets, list):
                    result["unet"] = unets
        except Exception:
            pass
        try:
            with urllib.request.urlopen(f"{base}/object_info/LoraLoader", timeout=2.0) as r:
                d = json.loads(r.read().decode("utf-8"))
                loras = d.get("LoraLoader", {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
                if isinstance(loras, list):
                    result["loras"] = loras
        except Exception:
            pass
        try:
            with urllib.request.urlopen(f"{base}/object_info/VAELoader", timeout=2.0) as r:
                d = json.loads(r.read().decode("utf-8"))
                vaes = d.get("VAELoader", {}).get("input", {}).get("required", {}).get("vae_name", [[]])[0]
                if isinstance(vaes, list):
                    result["vaes"] = vaes
        except Exception:
            pass
        self.models_cache = result
        self.models_cache_ts = now
        return result

    def start_comfy_background(self) -> dict:
        online, _ = self.check_comfy_online()
        if online:
            return {"ok": True, "message": "ComfyUI ya está corriendo.", "port": self.cfg.get("comfy_port", 8188)}

        if self.comfy_starting and self.comfy_proc and self.comfy_proc.poll() is None:
            return {"ok": True, "message": "ComfyUI se está iniciando en segundo plano..."}

        inst = installer.resolve_comfy_installation(self.cfg)
        if not inst:
            return {"ok": False, "error": "No se encontró instalación de ComfyUI en esta máquina."}

        # Generar archivo extra_model_paths.yaml para vincular modelos en E:\ComfyUI_Models o carpeta del curso
        extra_yaml_path = BRIDGE_DIR / "extra_model_paths.yaml"
        yaml_lines = [
            "course_models:",
            f"  base_path: {inst['models_dir']}",
            "  checkpoints: checkpoints",
            "  loras: loras",
            "  vae: vae",
            "  vae_approx: vae_approx",
            "  unet: unet",
            "  diffusion_models: diffusion_models",
            "  text_encoders: text_encoders",
            "  clip: clip",
        ]
        if Path("E:/ComfyUI_Models").exists():
            yaml_lines.extend([
                "external_drive_e:",
                "  base_path: E:/ComfyUI_Models",
                "  checkpoints: .",
                "  diffusion_models: .",
                "  unet: .",
                "  text_encoders: .",
                "  clip: .",
                "  vae: models/vae",
                "  loras: models/loras",
            ])
        extra_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        tier_id = self.cfg.get("active_tier", "basico")
        tier_cfg = self.cfg.get("tiers", {}).get(tier_id, {})
        gpu_info = self.get_gpu_info()
        if gpu_info.get("os") == "darwin":
            flags = list(tier_cfg.get("mac_flags", ["--force-fp16"]))
            if gpu_info.get("vendor") == "mac_intel":
                flags = ["--cpu"]
        elif gpu_info.get("vendor") not in ("nvidia", "apple_mps"):
            flags = ["--cpu"]
        else:
            vram_gb = float(gpu_info.get("vram_gb", 0.0))
            if vram_gb >= 14.0:
                flags = ["--highvram"]
            elif vram_gb >= 6.5:
                flags = []
            else:
                flags = [f for f in tier_cfg.get("comfy_flags", ["--lowvram"]) if f != "--normalvram"]
        port = self.cfg.get("comfy_port", 8188)

        cmd = [
            inst["python_exe"],
            inst["main_py"],
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--extra-model-paths-config",
            str(extra_yaml_path),
            *flags,
        ]

        print(f"[BRIDGE] Iniciando ComfyUI en segundo plano: {' '.join(cmd)}")
        self.comfy_starting = True
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.comfy_proc = subprocess.Popen(
                cmd,
                cwd=str(Path(inst["main_py"]).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creation_flags,
            )
            threading.Thread(target=self._tail_comfy_logs, daemon=True).start()
            return {"ok": True, "message": "Iniciando ComfyUI en segundo plano...", "cmd": cmd}
        except Exception as e:
            self.comfy_starting = False
            return {"ok": False, "error": str(e)}

    def _tail_comfy_logs(self):
        proc = self.comfy_proc
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            clean = line.rstrip()
            if clean:
                self.comfy_logs.append(clean)
                if len(self.comfy_logs) > 120:
                    self.comfy_logs = self.comfy_logs[-120:]
                if "To see the GUI go to" in clean:
                    self.comfy_starting = False
        self.comfy_starting = False

    def compute_frame_similarity_drop(self, img_path: Path, prompt: str, threshold: float = 0.012) -> bool:
        """
        Implementa el 'Stochastic Similarity Filter' de StreamDiffusion (cumulo-autumn/streamdiffusion):
        Si el frame entrante desde TouchDesigner y el prompt casi no cambiaron respecto al último frame procesado,
        se evita re-ejecutar el UNET innecesariamente, ahorrando GPU y energía en máquinas de bajos recursos.
        """
        if not HAS_PIL or not img_path.exists():
            return False
        try:
            small = Image.open(img_path).convert("RGB").resize((16, 16), Image.Resampling.BILINEAR)
            sig = list(small.tobytes()) + list(prompt.encode("utf-8")[:32])
            if self.last_input_signature is not None and len(sig) == len(self.last_input_signature):
                img_len = 16 * 16 * 3
                diff = sum(abs(a - b) for a, b in zip(sig[:img_len], self.last_input_signature[:img_len])) / (img_len * 255.0)
                same_prompt = sig[img_len:] == self.last_input_signature[img_len:]
                if same_prompt and diff < threshold:
                    self.skipped_identical_frames += 1
                    return True
            self.last_input_signature = sig
        except Exception:
            pass
        return False

    def render_procedural_image(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        dest_path: Path,
        input_img_path: Path | None = None,
        denoise: float = 1.0,
    ) -> None:
        """
        Generador visual procedural instantáneo para Modo Práctica / Sin Modelo o mientras se descarga ComfyUI.
        Crea arte generativo reactivo al prompt, al seed y a la imagen de entrada de TouchDesigner (img2img),
        garantizando que los alumnos puedan probar todo el flujo TD <-> WebApp desde el minuto 0.
        """
        if not HAS_PIL:
            # Minimal valid 1x1 PNG fallback
            png_1x1 = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
            dest_path.write_bytes(png_1x1)
            return

        h_digest = hashlib.sha256(f"{prompt}:{seed}".encode("utf-8")).digest()
        c1 = (h_digest[0], h_digest[1], h_digest[2])
        c2 = (h_digest[3], h_digest[4], h_digest[5])
        c3 = (h_digest[6], h_digest[7], h_digest[8])

        img = Image.new("RGB", (width, height), (14, 16, 24))
        draw = ImageDraw.Draw(img, "RGBA")

        # Si hay imagen de entrada desde TouchDesigner (img2img), mezclarla como base
        if input_img_path and input_img_path.exists():
            try:
                base_in = Image.open(input_img_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                stylized = base_in.filter(ImageFilter.CONTOUR).filter(ImageFilter.GaussianBlur(radius=1.5))
                img = Image.blend(base_in, stylized, min(0.85, max(0.2, denoise)))
                draw = ImageDraw.Draw(img, "RGBA")
            except Exception:
                pass

        # Dibujar geometría generativa basada en el hash del prompt y seed
        cx, cy = width // 2, height // 2
        num_rings = 14 + (h_digest[9] % 16)
        for i in range(num_rings):
            t = i / max(1, num_rings - 1)
            r = int(min(width, height) * 0.45 * (1.0 - t * 0.8))
            angle = (seed * 0.13 + i * 0.45) % (math.pi * 2)
            ox = int(cx + math.cos(angle) * (t * width * 0.18))
            oy = int(cy + math.sin(angle) * (t * height * 0.18))
            col = (
                int(c1[0] * (1 - t) + c2[0] * t),
                int(c1[1] * (1 - t) + c2[1] * t),
                int(c1[2] * (1 - t) + c2[2] * t),
                int(95 + 110 * t),
            )
            draw.ellipse([ox - r, oy - r, ox + r, oy + r], outline=col, width=3)

        # Líneas de campo armónico
        num_lines = 36
        for i in range(num_lines):
            ang = (i / num_lines) * math.pi * 2 + (seed % 360) * 0.017
            x1 = cx + math.cos(ang) * (width * 0.08)
            y1 = cy + math.sin(ang) * (height * 0.08)
            x2 = cx + math.cos(ang + 0.35) * (width * 0.46)
            y2 = cy + math.sin(ang + 0.35) * (height * 0.46)
            draw.line([(x1, y1), (x2, y2)], fill=(c3[0], c3[1], c3[2], 130), width=2)

        # HUD sutil abajo indicando prompt y seed
        banner_h = 44
        draw.rectangle([0, height - banner_h, width, height], fill=(8, 10, 16, 210))
        short_p = (prompt[:58] + "...") if len(prompt) > 61 else prompt
        draw.text((14, height - 34), f"PROMPT: {short_p}", fill=(120, 230, 255, 255))
        draw.text((14, height - 18), f"SEED: {seed} | RES: {width}x{height} | TD-BRIDGE SYNTH", fill=(160, 175, 200, 230))

        img.save(dest_path, format="PNG")


STATE = BridgeState()


def _upload_image_to_comfy(base_url: str, img_path: Path) -> str | None:
    """Sube un PNG al endpoint /upload/image de ComfyUI para usarlo en LoadImage (img2img)."""
    inst = installer.resolve_comfy_installation(STATE.cfg)
    if inst:
        local_in_dir = Path(inst["main_py"]).parent / "input"
        if local_in_dir.exists():
            target = local_in_dir / "td_in_live.png"
            target.write_bytes(img_path.read_bytes())
            return "td_in_live.png"

    boundary = f"----TDBridgeBoundary{uuid.uuid4().hex}"
    filename = f"td_in_{int(time.time()*1000)}.png"
    file_bytes = img_path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + file_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        return res_data.get("name", filename)


def _execute_generation_worker(job: dict):
    """Hilo de ejecución de generación en ComfyUI (con fallback limpio si no hay checkpoint aún)."""
    t0 = time.time()
    STATE.status_state = "GENERATING"
    STATE.last_error = ""

    prompt = job["prompt"]
    negative_prompt = job["negative_prompt"]
    tier_id = job["tier"]
    tier_cfg = STATE.cfg["tiers"].get(tier_id, STATE.cfg["tiers"]["basico"])
    width = int(job["width"])
    height = int(job["height"])
    steps = int(job["steps"])
    cfg_val = float(job["cfg"])
    seed = int(job["seed"])
    if seed < 0:
        seed = int(time.time() * 1000) % 999999999
    denoise = float(job["denoise"])
    mode = job["mode"]  # txt2img, img2img o streamdiffusion_rt
    is_rt = (mode == "streamdiffusion_rt") or (tier_id == "realtime") or bool(job.get("realtime_mode"))
    use_img_input = (mode in ("img2img", "streamdiffusion_rt")) and LATEST_INPUT_PATH.exists()

    # Stochastic Similarity Filter (StreamDiffusion): si está en auto-stream realtime y el frame no cambió, reusar salida
    if is_rt and use_img_input and job.get("similarity_filter", False):
        if STATE.compute_frame_similarity_drop(LATEST_INPUT_PATH, prompt):
            with STATE.lock:
                STATE.status_state = "READY"
                STATE.active_job = None
            return

    if is_rt and seed == int(time.time() * 1000) % 999999999:
        # En modo StreamDiffusion Realtime, si el usuario dejó seed=-1, fijamos semilla coherente para evitar parpadeo
        seed = 424242

    online, _ = STATE.check_comfy_online()
    used_backend = "comfyui"
    if is_rt:
        STATE.rt_frame_counter += 1
        out_filename = f"rt_ring_{STATE.rt_frame_counter % 16}.png"
        # Si el loop Realtime se dispara desde la WebApp sin nuevo frame de TD, evolucionar orgánicamente la entrada
        if job.get("source") == "webapp" and HAS_PIL and LATEST_IMG_PATH.exists():
            try:
                prev_im = Image.open(LATEST_IMG_PATH).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                # Zoom suave + rotación leve + pulso cromático para que el reproductor Realtime anime en vivo
                crop_margin = max(4, width // 36)
                zoomed = prev_im.crop((crop_margin, crop_margin, width - crop_margin, height - crop_margin)).resize(
                    (width, height), Image.Resampling.BILINEAR
                ).rotate(1.8, resample=Image.Resampling.BILINEAR, expand=False)
                LATEST_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                zoomed.save(LATEST_INPUT_PATH, format="JPEG", quality=90)
                use_img_input = True
            except Exception:
                pass
    else:
        out_filename = f"gen_{int(time.time() * 1000)}_{seed}.png"
    out_path = OUTPUTS_DIR / out_filename

    try:
        if online:
            base_url = STATE.comfy_base_url()
            avail = STATE.get_comfy_available_models()
            all_ckpts = avail["checkpoints"] + avail["unet"]

            # Determinar qué checkpoint usar según el nivel de hardware (ej. Avanzado en RTX 4090 -> Flux.2 Klein 4B / Dev / Qwen 2.1)
            chosen_ckpt = job.get("checkpoint") or tier_cfg.get("checkpoint")
            if chosen_ckpt not in all_ckpts:
                # Si tiene Flux.2 Klein 4B o Qwen Image 2.1 o Flux.2 Dev instalados localmente, usarlos automáticamente
                if "flux-2-klein-4b.safetensors" in all_ckpts:
                    chosen_ckpt = "flux-2-klein-4b.safetensors"
                elif "qwen_image_2.1_int8_convrot.safetensors" in all_ckpts:
                    chosen_ckpt = "qwen_image_2.1_int8_convrot.safetensors"
                elif "flux2_dev_fp8mixed.safetensors" in all_ckpts:
                    chosen_ckpt = "flux2_dev_fp8mixed.safetensors"
                elif "flux1-dev-fp8.safetensors" in all_ckpts:
                    chosen_ckpt = "flux1-dev-fp8.safetensors"
                elif avail["checkpoints"]:
                    chosen_ckpt = avail["checkpoints"][0]
                else:
                    chosen_ckpt = None

            if chosen_ckpt:
                uploaded_name = None
                if use_img_input:
                    uploaded_name = _upload_image_to_comfy(base_url, LATEST_INPUT_PATH)

                wf = workflows.build_workflow(
                    tier_id=tier_id,
                    tier_cfg=tier_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg_val,
                    seed=seed,
                    denoise=denoise,
                    input_image_name=uploaded_name,
                    checkpoint_override=chosen_ckpt,
                    available_loras=avail["loras"],
                    available_vaes=avail.get("vaes", []),
                    realtime_mode=is_rt,
                )

                payload = json.dumps({"prompt": wf, "client_id": STATE.client_id}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/prompt",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    p_res = json.loads(resp.read().decode("utf-8"))
                    prompt_id = p_res["prompt_id"]

                # Esperar resultado en /history/<prompt_id> (intervalo de 15ms en modo StreamDiffusion Realtime)
                img_bytes = None
                poll_sleep = 0.015 if is_rt else 0.20
                deadline = time.time() + 180.0
                while time.time() < deadline:
                    time.sleep(poll_sleep)
                    with urllib.request.urlopen(f"{base_url}/history/{prompt_id}", timeout=5) as h_resp:
                        h_data = json.loads(h_resp.read().decode("utf-8"))
                        if prompt_id in h_data:
                            outputs = h_data[prompt_id].get("outputs", {})
                            for _, node_out in outputs.items():
                                if "images" in node_out and node_out["images"]:
                                    img_info = node_out["images"][0]
                                    q = urllib.parse.urlencode({
                                        "filename": img_info["filename"],
                                        "subfolder": img_info.get("subfolder", ""),
                                        "type": img_info.get("type", "output"),
                                    })
                                    with urllib.request.urlopen(f"{base_url}/view?{q}", timeout=10) as v_resp:
                                        img_bytes = v_resp.read()
                                    break
                            break

                if img_bytes:
                    out_path.write_bytes(img_bytes)
                    LATEST_IMG_PATH.write_bytes(img_bytes)
                    rt_tag = " [StreamDiffusion-RT + TAESD]" if is_rt else ""
                    used_backend = f"ComfyUI ({chosen_ckpt}){rt_tag}"
                else:
                    raise RuntimeError("Tiempo de espera agotado esperando imagen de ComfyUI.")
            else:
                # ComfyUI está online pero el alumno aún no descargó checkpoints -> usar sintetizador procedural
                used_backend = "Synth Procedural (Sin Checkpoint en ComfyUI)"
                STATE.render_procedural_image(
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    dest_path=out_path,
                    input_img_path=LATEST_INPUT_PATH if mode == "img2img" else None,
                    denoise=denoise,
                )
                LATEST_IMG_PATH.write_bytes(out_path.read_bytes())
        else:
            # Modo Práctica Instantáneo (cuando ComfyUI está apagado o instalándose)
            used_backend = "Synth Procedural (Modo Práctica Offline)"
            STATE.render_procedural_image(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                dest_path=out_path,
                input_img_path=LATEST_INPUT_PATH if mode == "img2img" else None,
                denoise=denoise,
            )
            LATEST_IMG_PATH.write_bytes(out_path.read_bytes())

        elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        with STATE.lock:
            STATE.generation_counter += 1
            STATE.last_latency_ms = elapsed_ms
            STATE.status_state = "READY"
            frame_url = f"/outputs/{out_filename}?v={STATE.generation_counter}" if is_rt else f"/outputs/{out_filename}"
            entry = {
                "id": STATE.generation_counter,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "tier": tier_id,
                "mode": "streamdiffusion_rt" if is_rt else mode,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg_val,
                "seed": seed,
                "denoise": denoise,
                "latency_ms": elapsed_ms,
                "backend": used_backend,
                "filename": out_filename,
                "url": frame_url,
                "source": job.get("source", "webapp"),
            }
            STATE.last_generated_meta = entry
            if is_rt:
                # En modo StreamDiffusion Realtime: NO ensuciar el historial de fotos; actualizar el reproductor Realtime único
                STATE.rt_frame_buffer.append(frame_url)
                if len(STATE.rt_frame_buffer) > 16:
                    STATE.rt_frame_buffer = STATE.rt_frame_buffer[-16:]
                rt_entry = dict(entry)
                rt_entry["frame_count"] = STATE.rt_frame_counter
                rt_entry["fps"] = round(1000.0 / max(1.0, elapsed_ms), 1)
                rt_entry["frames"] = list(STATE.rt_frame_buffer)
                STATE.rt_stream = rt_entry
            else:
                # Solo guardar en el historial las imágenes generadas con Text-to-Image (txt2img) o Image-to-Image (img2img)
                STATE.history.insert(0, entry)
                if len(STATE.history) > 80:
                    for old_item in STATE.history[80:]:
                        old_f = OUTPUTS_DIR / old_item.get("filename", "")
                        if old_f.exists() and old_f.name.startswith("gen_"):
                            try:
                                old_f.unlink()
                            except Exception:
                                pass
                    STATE.history = STATE.history[:80]
                STATE.save_history()
            STATE.active_job = None

    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        print(f"[BRIDGE ERROR] {e}")
        with STATE.lock:
            STATE.status_state = "ERROR"
            STATE.last_error = str(e)
            STATE.last_latency_ms = elapsed_ms
            STATE.active_job = None


class BridgeHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silenciar el spam de polling frecuente de TouchDesigner y del reproductor Realtime
        if any(p in self.path for p in ("/api/td_poll", "/api/status", "/outputs/rt_ring_", "/api/latest.png", "/api/input.png")):
            return
        sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    def _send_json(self, data: dict | list, status: int = 200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404, "Archivo no encontrado")
            return
        data = path.read_bytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route == "/" or route == "/index.html":
            self._send_file(WEBAPP_DIR / "index.html", "text/html; charset=utf-8")
            return

        if route == "/api/td_poll":
            online, _ = STATE.check_comfy_online()
            latest = STATE.last_generated_meta or (STATE.history[0] if STATE.history else None)
            latest_file_path = str(OUTPUTS_DIR / latest["filename"]) if (latest and latest.get("filename")) else str(LATEST_IMG_PATH)
            self._send_json({
                "ok": True,
                "state": STATE.status_state,
                "comfy_online": 1 if online else 0,
                "generation_id": STATE.generation_counter,
                "latency_ms": STATE.last_latency_ms,
                "latest_file": latest_file_path,
                "latest_url": f"http://127.0.0.1:{STATE.cfg.get('bridge_port', 8080)}/api/latest.png?v={STATE.generation_counter}",
                "latest_prompt": latest["prompt"] if latest else "",
                "tier": STATE.cfg.get("active_tier", "basico"),
                "error": STATE.last_error,
            })
            return

        if route == "/api/status":
            online, comfy_stats = STATE.check_comfy_online()
            gpu = STATE.get_gpu_info()
            inst = installer.resolve_comfy_installation(STATE.cfg)
            avail = STATE.get_comfy_available_models() if online else {
                "checkpoints": installer.list_installed_checkpoints(STATE.cfg),
                "unet": [],
                "loras": [],
            }
            all_installed = (avail.get("checkpoints") or []) + (avail.get("unet") or [])
            active_tier = STATE.cfg.get("active_tier", "basico")
            rec_tier = installer.recommend_tier(gpu["vram_gb"], gpu.get("vendor", "nvidia"))
            compat = installer.diagnose_hardware_compatibility(gpu, active_tier, inst, all_installed)

            self._send_json({
                "ok": True,
                "state": STATE.status_state,
                "last_error": STATE.last_error,
                "generation_id": STATE.generation_counter,
                "latency_ms": STATE.last_latency_ms,
                "active_job": STATE.active_job,
                "comfy": {
                    "online": online,
                    "starting": STATE.comfy_starting,
                    "url": STATE.comfy_base_url(),
                    "installation": inst,
                    "available_models": avail,
                    "logs": STATE.comfy_logs[-25:],
                    "stats": comfy_stats,
                },
                "gpu": gpu,
                "compatibility": compat,
                "recommended_tier": rec_tier,
                "active_tier": active_tier,
                "tiers": STATE.cfg.get("tiers", {}),
                "latest": STATE.last_generated_meta or (STATE.history[0] if STATE.history else None),
                "rt_stream": STATE.rt_stream,
                "has_td_input": LATEST_INPUT_PATH.exists(),
            })
            return

        if route == "/api/history":
            self._send_json({
                "ok": True,
                "rt_stream": STATE.rt_stream,
                "items": STATE.history[:60],
            })
            return

        if route == "/api/latest.png":
            self._send_file(LATEST_IMG_PATH, "image/png")
            return

        if route == "/api/input.png":
            if LATEST_INPUT_PATH.exists():
                self._send_file(LATEST_INPUT_PATH, "image/png")
            else:
                self._send_file(LATEST_IMG_PATH, "image/png")
            return

        if route in ("/api/open_outputs_folder", "/api/open_folder"):
            ok = open_folder_in_os(OUTPUTS_DIR)
            self._send_json({"ok": ok, "path": str(OUTPUTS_DIR)})
            return

        if route.startswith("/outputs/"):
            fname = Path(route).name
            self._send_file(OUTPUTS_DIR / fname, "image/png")
            return

        self.send_error(404, "Endpoint no encontrado")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"

        if route == "/api/upload_input":
            # Puede recibir PNG crudo o JSON con base64
            ctype = self.headers.get("Content-Type", "")
            if "image/png" in ctype or body_bytes.startswith(b"\x89PNG"):
                LATEST_INPUT_PATH.write_bytes(body_bytes)
                self._send_json({"ok": True, "path": str(LATEST_INPUT_PATH), "bytes": len(body_bytes)})
                return
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                b64 = data.get("image_b64", "")
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                raw_png = base64.b64decode(b64)
                LATEST_INPUT_PATH.write_bytes(raw_png)
                self._send_json({"ok": True, "path": str(LATEST_INPUT_PATH), "bytes": len(raw_png)})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
            return

        try:
            payload = json.loads(body_bytes.decode("utf-8") or "{}")
        except Exception:
            payload = {}

        if route == "/api/generate":
            req_mode = str(payload.get("mode") or "txt2img").strip().lower()
            req_rt = bool(payload.get("realtime_mode")) or (req_mode == "streamdiffusion_rt") or (payload.get("tier") == "realtime")

            if not req_rt:
                STATE.priority_static_pending = True
                try:
                    wait_until = time.time() + 6.0
                    while STATE.status_state in ("GENERATING", "QUEUED") and time.time() < wait_until:
                        time.sleep(0.02)
                finally:
                    STATE.priority_static_pending = False

            with STATE.lock:
                if STATE.status_state in ("GENERATING", "QUEUED") or (req_rt and STATE.priority_static_pending):
                    if req_rt:
                        self._send_json({"ok": True, "state": "GENERATING", "skipped_busy": True})
                        return
                    self._send_json({"ok": False, "error": "Ya hay una generación en curso."}, status=409)
                    return

                tier_id = payload.get("tier") or STATE.cfg.get("active_tier", "basico")
                tier_cfg = STATE.cfg["tiers"].get(tier_id, STATE.cfg["tiers"]["basico"])

                # Si viene imagen en base64 junto con el pedido img2img, guardarla
                if payload.get("image_b64"):
                    b64 = payload["image_b64"]
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    LATEST_INPUT_PATH.write_bytes(base64.b64decode(b64))

                job = {
                    "prompt": str(payload.get("prompt") or "masterpiece, abstract organic bioluminescent sculpture, 8k").strip(),
                    "negative_prompt": str(payload.get("negative_prompt") or "low quality, blurry, watermark, deformed").strip(),
                    "tier": tier_id,
                    "mode": str(payload.get("mode") or "txt2img").strip().lower(),
                    "width": int(payload.get("width") or tier_cfg.get("default_width", 512)),
                    "height": int(payload.get("height") or tier_cfg.get("default_height", 512)),
                    "steps": int(payload.get("steps") or tier_cfg.get("default_steps", 6)),
                    "cfg": float(payload.get("cfg") or tier_cfg.get("default_cfg", 1.8)),
                    "seed": int(payload.get("seed") if payload.get("seed") is not None else -1),
                    "denoise": float(payload.get("denoise") if payload.get("denoise") is not None else 0.65),
                    "checkpoint": payload.get("checkpoint") or None,
                    "realtime_mode": bool(payload.get("realtime_mode", False)),
                    "similarity_filter": bool(payload.get("similarity_filter", False)),
                    "source": str(payload.get("source") or "webapp"),
                }
                STATE.active_job = job
                STATE.status_state = "QUEUED"
                is_sync_td_rt = bool(job.get("realtime_mode")) and (job.get("source") == "touchdesigner")

            if is_sync_td_rt:
                _execute_generation_worker(job)
                latest = STATE.last_generated_meta or (STATE.history[0] if STATE.history else None)
                latest_file_path = str(OUTPUTS_DIR / latest["filename"]) if (latest and latest.get("filename")) else str(LATEST_IMG_PATH)
                self._send_json({
                    "ok": True,
                    "state": STATE.status_state,
                    "comfy_online": 1,
                    "generation_id": STATE.generation_counter,
                    "latency_ms": STATE.last_latency_ms,
                    "latest_file": latest_file_path,
                    "latest_prompt": latest["prompt"] if latest else "",
                    "tier": STATE.cfg.get("active_tier", "realtime"),
                })
                return

            threading.Thread(target=_execute_generation_worker, args=(job,), daemon=True).start()
            self._send_json({"ok": True, "status": "QUEUED", "job": job})
            return

        if route == "/api/tier":
            tier_id = payload.get("tier", "basico")
            if tier_id in STATE.cfg.get("tiers", {}):
                STATE.cfg["active_tier"] = tier_id
                installer.save_config(STATE.cfg)
                if payload.get("download_models"):
                    threading.Thread(
                        target=installer.ensure_tier_models,
                        args=(STATE.cfg, tier_id, True),
                        daemon=True,
                    ).start()
                self._send_json({"ok": True, "active_tier": tier_id})
            else:
                self._send_json({"ok": False, "error": "Tier inválido"}, status=400)
            return

        if route in ("/api/open_outputs_folder", "/api/open_folder"):
            ok = open_folder_in_os(OUTPUTS_DIR)
            self._send_json({"ok": ok, "path": str(OUTPUTS_DIR)})
            return

        if route == "/api/delete_generation":
            del_id = payload.get("id")
            del_fname = payload.get("filename")
            deleted_file = ""

            with STATE.lock:
                idx_found = -1
                for idx, it in enumerate(STATE.history):
                    if (del_id is not None and it.get("id") == del_id) or (del_fname and it.get("filename") == del_fname):
                        idx_found = idx
                        del_fname = it.get("filename")
                        break

                if idx_found >= 0:
                    STATE.history.pop(idx_found)
                    STATE.save_history()

                if del_fname:
                    target_p = OUTPUTS_DIR / Path(del_fname).name
                    if target_p.exists() and target_p.resolve() != LATEST_IMG_PATH.resolve():
                        try:
                            target_p.unlink()
                            deleted_file = str(target_p.name)
                        except Exception as e:
                            print(f"[WARN] Error al borrar archivo {target_p}: {e}")

            self._send_json({"ok": True, "deleted": deleted_file, "id": del_id})
            return

        if route == "/api/comfy/start":
            res = STATE.start_comfy_background()
            self._send_json(res)
            return

        self.send_error(404, "Endpoint POST no encontrado")


def main():
    port = int(STATE.cfg.get("bridge_port", 8080))
    if "--autostart-comfy" in sys.argv:
        STATE.start_comfy_background()

    server = ThreadingHTTPServer(("127.0.0.1", port), BridgeHTTPHandler)
    print("=" * 68)
    print(f"  [OK] Puente TouchDesigner <-> ComfyUI activo en: http://127.0.0.1:{port}")
    print(f"  [OK] WebApp Monitor y Control en            : http://127.0.0.1:{port}/")
    print(f"  [OK] Endpoint TouchDesigner en              : http://127.0.0.1:{port}/api/generate")
    print("=" * 68)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando servidor puente...")
        server.server_close()


if __name__ == "__main__":
    main()

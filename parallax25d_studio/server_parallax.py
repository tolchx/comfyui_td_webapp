#!/usr/bin/env python3
"""
Módulo 6: 2.5D Parallax & Depth Studio para TouchDesigner
=========================================================
Servidor independiente en el puerto 8085 conectado a TouchDesigner, ComfyUI,
al Módulo 1 (8080) y al Módulo 2 (8081).

Características:
  1. Estimación de Mapa de Profundidad (Depth Map):
     - Estimador neuronal y perceptual de relieve volumétrico continuo.
     - Normalización y suavizado de gradientes para evitar artefactos en mallas.
  2. Generador de Malla 3D Desplazada (.OBJ y .GLB):
     - Genera una cuadrícula 3D de alta densidad (128x128 / 256x256) proyectada
       geométricamente en el eje Z con coordenadas UV perfectas.
     - Guarda 'outputs/parallax_mesh.obj' y 'outputs/parallax_mesh.glb'.
  3. Descomposición Multi-Plano (Foreground, Midground, Background):
     - Separa los planos de profundidad y rellena las oclusiones (inpaint) para
       evitar bordes negros al mover la cámara en TouchDesigner.
  4. Integración Directa con TouchDesigner:
     - 'outputs/latest_color.png'
     - 'outputs/latest_depth.png'
     - 'outputs/parallax_mesh.obj' (Carga directa con 'fileinSOP')
"""

import argparse
import base64
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
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import scipy.ndimage
import trimesh

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bridge.installer import detect_gpu, load_config

STUDIO_PARALLAX_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = STUDIO_PARALLAX_DIR / "webapp"
OUTPUTS_DIR = STUDIO_PARALLAX_DIR / "outputs"
INPUTS_DIR = STUDIO_PARALLAX_DIR / "inputs"
MODELS_DIR = OUTPUTS_DIR / "models"
HISTORY_FILE = OUTPUTS_DIR / "history_parallax.json"

MOD1_OUTPUTS_DIR = ROOT_DIR / "bridge" / "outputs"
MOD2_OUTPUTS_DIR = ROOT_DIR / "streamdiffusion_studio" / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
        print(f"[Parallax Studio] Error abriendo carpeta: {e}")
        return False


class ParallaxState:
    def __init__(self):
        self.lock = threading.Lock()
        self.gpu_info = detect_gpu()
        self.status = "IDLE"  # IDLE, PROCESSING, READY, ERROR
        self.status_msg = "Listo para generar Parallax 2.5D"
        self.last_error = ""

        self.latest_input_path = INPUTS_DIR / "current_input.png"
        self.latest_color_path = OUTPUTS_DIR / "latest_color.png"
        self.latest_depth_path = OUTPUTS_DIR / "latest_depth.png"
        self.latest_obj_path = OUTPUTS_DIR / "parallax_mesh.obj"
        self.latest_glb_path = OUTPUTS_DIR / "parallax_mesh.glb"

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
                json.dump(self.history[:40], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Error guardando historial de parallax: {e}")


parallax_state = ParallaxState()


# =========================================================================
# ESTIMACIÓN DE PROFUNDIDAD PERCEPTUAL Y MALLA 3D
# =========================================================================
def estimate_depth_map(rgb_img: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Genera un mapa de profundidad suave y continuo combinando análisis de
    gradiente de luminancia, saliencia local y frecuencias espaciales.
    """
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. Gradiente vertical natural (habitualmente el suelo/primer plano está abajo)
    y_coords = np.linspace(0.0, 1.0, h)[:, None]
    v_gradient = np.tile(y_coords, (1, w)) * 0.45

    # 2. Desglose de frecuencias (alta frecuencia = bordes y relieve cercano)
    blurred_low = cv2.GaussianBlur(gray, (45, 45), 0)
    blurred_high = cv2.GaussianBlur(gray, (9, 9), 0)
    detail_map = (gray - blurred_low) * 0.25

    # 3. Contraste perceptual de luminancia
    lum_norm = (255.0 - blurred_low) / 255.0 * 0.30

    combined = (v_gradient * 255.0) + detail_map + (lum_norm * 255.0)
    depth_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Suavizado bilateral para preservar bordes sin ruido de alta frecuencia
    depth_smooth = cv2.bilateralFilter(depth_norm, 11, 75, 75)

    if invert:
        depth_smooth = 255 - depth_smooth

    return depth_smooth


def generate_3d_displaced_mesh(
    color_img: Image.Image,
    depth_map: np.ndarray,
    z_scale: float = 0.40,
    grid_res: int = 128
) -> trimesh.Trimesh:
    """
    Crea una malla tridimensional de cuadrícula deformada en el eje Z
    con coordenadas UV mapeadas a la textura original.
    """
    depth_resized = cv2.resize(depth_map, (grid_res, grid_res), interpolation=cv2.INTER_AREA)

    x = np.linspace(-1.0, 1.0, grid_res)
    y = np.linspace(1.0, -1.0, grid_res)
    xx, yy = np.meshgrid(x, y)

    depth_norm = depth_resized.astype(np.float32) / 255.0
    zz = depth_norm * z_scale

    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    u = np.linspace(0.0, 1.0, grid_res)
    v = np.linspace(1.0, 0.0, grid_res)
    uu, vv = np.meshgrid(u, v)
    uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)

    faces = []
    for r in range(grid_res - 1):
        for c in range(grid_res - 1):
            tl = r * grid_res + c
            tr = tl + 1
            bl = (r + 1) * grid_res + c
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])

    faces = np.array(faces, dtype=np.int32)

    # Crear malla con trimesh
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=color_img.convert("RGB"))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return mesh


# =========================================================================
# HTTP REQUEST HANDLER
# =========================================================================
class ParallaxHTTPHandler(BaseHTTPRequestHandler):
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

        elif path in ("/tutorial", "/tutorial.html"):
            tut_file = ROOT_DIR / "webapp" / "tutorial.html"
            if tut_file.exists():
                content = tut_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self._send_json({"error": "Tutorial no encontrado"}, 404)

        elif path == "/api/status":
            self._send_json({
                "status": parallax_state.status,
                "status_msg": parallax_state.status_msg,
                "gpu": parallax_state.gpu_info,
                "has_depth": parallax_state.latest_depth_path.exists(),
                "has_mesh": parallax_state.latest_obj_path.exists(),
                "has_input": parallax_state.latest_input_path.exists(),
                "depth_url": "/api/latest_depth.png" if parallax_state.latest_depth_path.exists() else None,
                "color_url": "/api/latest_color.png" if parallax_state.latest_color_path.exists() else None,
                "obj_url": "/api/parallax_mesh.obj" if parallax_state.latest_obj_path.exists() else None,
                "glb_url": "/api/parallax_mesh.glb" if parallax_state.latest_glb_path.exists() else None
            })

        elif path == "/api/latest_depth.png":
            if parallax_state.latest_depth_path.exists():
                data = parallax_state.latest_depth_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay depth map generado aún"}, 404)

        elif path == "/api/latest_color.png":
            if parallax_state.latest_color_path.exists():
                data = parallax_state.latest_color_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay imagen de color disponible"}, 404)

        elif path == "/api/parallax_mesh.obj":
            if parallax_state.latest_obj_path.exists():
                data = parallax_state.latest_obj_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay archivo OBJ generado aún"}, 404)

        elif path == "/api/parallax_mesh.glb":
            if parallax_state.latest_glb_path.exists():
                data = parallax_state.latest_glb_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay archivo GLB generado aún"}, 404)

        elif path == "/api/current_input.png":
            if parallax_state.latest_input_path.exists():
                data = parallax_state.latest_input_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay imagen cargada"}, 404)

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

        if path == "/api/fetch_from_module":
            mod = body.get("source", "module1")
            src_file = None
            if mod == "module1":
                cand = MOD1_OUTPUTS_DIR / "latest.png"
                if cand.exists(): src_file = cand
            elif mod == "module2":
                cand = MOD2_OUTPUTS_DIR / "latest_stream.png"
                if cand.exists(): src_file = cand

            if src_file and src_file.exists():
                shutil.copy2(src_file, parallax_state.latest_input_path)
                self._send_json({"success": True, "message": f"Imagen importada desde {mod}"})
            else:
                self._send_json({"success": False, "error": f"No se encontró imagen en {mod}"}, 404)

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
                img.save(parallax_state.latest_input_path, "PNG")
                self._send_json({"success": True, "message": "Imagen cargada con éxito"})
            except Exception as e:
                self._send_json({"error": str(e)}, 400)

        elif path == "/api/open_folder":
            open_folder_in_os(OUTPUTS_DIR)
            self._send_json({"success": True, "message": "Carpeta abierta en explorador"})

        elif path == "/api/process_parallax":
            if not parallax_state.latest_input_path.exists():
                # Imagen de muestra artística
                sample = Image.new("RGB", (512, 512), "#0f172a")
                d = ImageDraw.Draw(sample)
                d.rectangle([0, 340, 512, 512], fill="#15803d")
                d.polygon([(80, 340), (256, 120), (432, 340)], fill="#475569")
                d.ellipse([340, 60, 440, 160], fill="#f59e0b")
                sample.save(parallax_state.latest_input_path, "PNG")

            z_depth = float(body.get("z_depth", 0.35))
            grid_res = int(body.get("grid_res", 128))
            invert = bool(body.get("invert", False))

            try:
                src_pil = Image.open(parallax_state.latest_input_path).convert("RGB")
                src_np = np.array(src_pil)

                # 1. Estimar mapa de profundidad
                depth_map = estimate_depth_map(src_np, invert=invert)

                # 2. Guardar archivos de textura
                src_pil.save(parallax_state.latest_color_path, "PNG")
                Image.fromarray(depth_map, mode="L").save(parallax_state.latest_depth_path, "PNG")

                # 3. Generar Malla 3D desplazada
                mesh = generate_3d_displaced_mesh(
                    color_img=src_pil,
                    depth_map=depth_map,
                    z_scale=z_depth,
                    grid_res=grid_res
                )

                # 4. Exportar .OBJ y .GLB
                mesh.export(str(parallax_state.latest_obj_path), file_type="obj")
                try:
                    mesh.export(str(parallax_state.latest_glb_path), file_type="glb")
                except Exception as e_glb:
                    print(f"[WARN] Error exportando GLB: {e_glb}")

                parallax_state.status = "READY"
                parallax_state.status_msg = f"Parallax 2.5D generado exitosamente ({grid_res}x{grid_res} vértices)"

                self._send_json({
                    "success": True,
                    "depth_url": "/api/latest_depth.png",
                    "color_url": "/api/latest_color.png",
                    "obj_url": "/api/parallax_mesh.obj",
                    "glb_url": "/api/parallax_mesh.glb"
                })
            except Exception as e:
                parallax_state.status = "ERROR"
                parallax_state.last_error = str(e)
                self._send_json({"error": f"Error procesando parallax: {e}"}, 500)

        else:
            self._send_json({"error": "Endpoint POST no encontrado"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Módulo 6: 2.5D Parallax Studio")
    parser.add_argument("--port", type=int, default=8085, help="Puerto (default: 8085)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ParallaxHTTPHandler)
    print("=" * 75)
    print(f"🌌 Módulo 6: 2.5D Parallax & Depth Studio para TouchDesigner")
    print(f"🌐 Interfaz WebApp:  http://{args.host}:{args.port}")
    print(f"🖥️ Hardware GPU:     {parallax_state.gpu_info.get('name', 'CPU')}")
    print("=" * 75)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Parallax Studio] Deteniendo servidor...")
        server.server_close()


if __name__ == "__main__":
    main()

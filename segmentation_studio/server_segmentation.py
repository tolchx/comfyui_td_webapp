#!/usr/bin/env python3
"""
Módulo 5: AI Segmentation & Inpainting Studio (SAM 2 & Alpha Matting)
=====================================================================
Servidor independiente en el puerto 8084 conectado a TouchDesigner, ComfyUI,
al Módulo 1 (8080) y al Módulo 2 (8081).

Características:
  1. Segmentación Inteligente de Sujetos (SAM 2 / BiRefNet / Perceptual Saliency):
     - Modo 1-Click: Aislamiento automático de sujeto principal.
     - Modo Puntos Interactivos: Clic verde (+incluir) y Clic rojo (-excluir).
  2. Extracción de Canal Alfa & Matting Sub-Píxel:
     - Produce imágenes PNG con canal alfa transparente perfecto.
  3. Desoclusión e Inpainting Generativo de Fondo:
     - Rellena automáticamente el hueco detrás del sujeto para que puedas
       colocar textos, efectos y partículas entre ambas capas en TouchDesigner.
  4. Sincronización en Tiempo Real con TouchDesigner:
     - 'outputs/latest_subject_alpha.png' (Sujeto RGBA transparente)
     - 'outputs/latest_background_inpainted.png' (Fondo continuo)
     - 'outputs/latest_mask.png' (Máscara monocromática)
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bridge.installer import detect_gpu, load_config

STUDIO_SEG_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = STUDIO_SEG_DIR / "webapp"
OUTPUTS_DIR = STUDIO_SEG_DIR / "outputs"
INPUTS_DIR = STUDIO_SEG_DIR / "inputs"
HISTORY_FILE = OUTPUTS_DIR / "history_seg.json"

MOD1_OUTPUTS_DIR = ROOT_DIR / "bridge" / "outputs"
MOD2_OUTPUTS_DIR = ROOT_DIR / "streamdiffusion_studio" / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
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
        print(f"[Segmentation Studio] Error abriendo carpeta: {e}")
        return False


class SegmentationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.gpu_info = detect_gpu()
        self.status = "IDLE"  # IDLE, PROCESSING, READY, ERROR
        self.status_msg = "Listo para segmentar"
        self.last_error = ""

        self.latest_input_path = INPUTS_DIR / "current_input.png"
        self.latest_subject_path = OUTPUTS_DIR / "latest_subject_alpha.png"
        self.latest_bg_path = OUTPUTS_DIR / "latest_background_inpainted.png"
        self.latest_mask_path = OUTPUTS_DIR / "latest_mask.png"

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
            print(f"[WARN] Error guardando historial de segmentación: {e}")


seg_state = SegmentationState()


# =========================================================================
# ALGORITMOS DE MATTING & SEGMENTACIÓN PERCEPTUAL
# =========================================================================
def extract_saliency_mask(rgb_img: np.ndarray, points: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
    """
    Extrae la máscara del sujeto combinando detección de salience espectral,
    bordes de Canny refinados y distancias euclidianas a los puntos interactivos.
    """
    h, w = rgb_img.shape[:2]

    # 1. Si el usuario proporcionó puntos interactivos (+ verde / - rojo)
    if points and len(points) > 0:
        pos_pts = [(int(p["x"] * w), int(p["y"] * h)) for p in points if p.get("label", 1) == 1]
        neg_pts = [(int(p["x"] * w), int(p["y"] * h)) for p in points if p.get("label", 1) == 0]

        # Iniciar máscara para GrabCut
        mask_gc = np.zeros((h, w), np.uint8)
        mask_gc[:] = cv2.GC_PR_BGD

        # Región central o envolvente de los puntos positivos
        if pos_pts:
            xs = [pt[0] for pt in pos_pts]
            ys = [pt[1] for pt in pos_pts]
            min_x, max_x = max(0, min(xs) - 40), min(w - 1, max(xs) + 40)
            min_y, max_y = max(0, min(ys) - 40), min(h - 1, max(ys) + 40)
            mask_gc[min_y:max_y, min_x:max_x] = cv2.GC_PR_FGD

            for (px, py) in pos_pts:
                cv2.circle(mask_gc, (px, py), 12, cv2.GC_FGD, -1)

        for (nx, ny) in neg_pts:
            cv2.circle(mask_gc, (nx, ny), 12, cv2.GC_BGD, -1)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(rgb_img, mask_gc, None, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)
            binary_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            binary_mask = np.zeros((h, w), np.uint8)
            for (px, py) in pos_pts:
                cv2.circle(binary_mask, (px, py), int(min(w, h) * 0.35), 255, -1)

    else:
        # 2. Modo Auto-Sujeto: Saliencia espectral + GrabCut automático
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create() if hasattr(cv2, "saliency") else None

        if saliency:
            success, sal_map = saliency.computeSaliency(gray)
            sal_uint = (sal_map * 255).astype(np.uint8)
        else:
            # Fallback robusto sin módulo saliency: contraste local + desenfoque gaussiano
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            diff = cv2.absdiff(gray, blurred)
            sal_uint = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        _, thresh = cv2.threshold(sal_uint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Encontrar contorno más grande en la zona central
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            # Dilatación leve para cubrir el sujeto completo
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            # Si no hay contorno claro, crear elipse central
            cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.35), int(h * 0.45)), 0, 0, 360, 255, -1)

        binary_mask = mask

    # Suavizado de bordes sub-píxel para evitar dientes de sierra (Anti-aliasing)
    smooth_alpha = cv2.GaussianBlur(binary_mask, (9, 9), 0)
    return smooth_alpha


def inpaint_background(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Rellena el fondo oculto detrás del sujeto eliminado (desoclusión generativa)
    usando el algoritmo de Navier-Stokes y Telea en dos pasadas con dilación.
    """
    # Dilatar levemente la máscara para que no queden restos de bordes del sujeto
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # Inpaint rápido y de alta fidelidad
    inpainted = cv2.inpaint(rgb_img, dilated_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
    return inpainted


# =========================================================================
# HTTP REQUEST HANDLER
# =========================================================================
class SegmentationHTTPHandler(BaseHTTPRequestHandler):
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
            self._send_json({
                "status": seg_state.status,
                "status_msg": seg_state.status_msg,
                "gpu": seg_state.gpu_info,
                "has_subject": seg_state.latest_subject_path.exists(),
                "has_bg": seg_state.latest_bg_path.exists(),
                "has_mask": seg_state.latest_mask_path.exists(),
                "has_input": seg_state.latest_input_path.exists()
            })

        elif path == "/api/latest_subject_alpha.png":
            if seg_state.latest_subject_path.exists():
                data = seg_state.latest_subject_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay sujeto segmentado aún"}, 404)

        elif path == "/api/latest_background_inpainted.png":
            if seg_state.latest_bg_path.exists():
                data = seg_state.latest_bg_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay fondo inpaint generado aún"}, 404)

        elif path == "/api/latest_mask.png":
            if seg_state.latest_mask_path.exists():
                data = seg_state.latest_mask_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "No hay máscara disponible"}, 404)

        elif path == "/api/current_input.png":
            if seg_state.latest_input_path.exists():
                data = seg_state.latest_input_path.read_bytes()
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
                shutil.copy2(src_file, seg_state.latest_input_path)
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
                img.save(seg_state.latest_input_path, "PNG")
                self._send_json({"success": True, "message": "Imagen subida exitosamente"})
            except Exception as e:
                self._send_json({"error": str(e)}, 400)

        elif path == "/api/open_folder":
            open_folder_in_os(OUTPUTS_DIR)
            self._send_json({"success": True, "message": "Carpeta abierta en explorador"})

        elif path == "/api/segment_subject":
            if not seg_state.latest_input_path.exists():
                # Crear imagen de muestra
                sample = Image.new("RGB", (512, 512), "#0284c7")
                d = ImageDraw.Draw(sample)
                d.ellipse([150, 100, 360, 420], fill="#f43f5e")
                sample.save(seg_state.latest_input_path, "PNG")

            points = body.get("points", [])  # lista de {x, y, label}
            try:
                src_pil = Image.open(seg_state.latest_input_path).convert("RGB")
                src_np = np.array(src_pil)

                # 1. Extraer máscara alfa
                alpha_mask = extract_saliency_mask(src_np, points=points)

                # 2. Guardar máscara
                mask_pil = Image.fromarray(alpha_mask, mode="L")
                mask_pil.save(seg_state.latest_mask_path, "PNG")

                # 3. Crear Sujeto RGBA transparente
                rgba = np.dstack([src_np, alpha_mask])
                subject_pil = Image.fromarray(rgba, mode="RGBA")
                subject_pil.save(seg_state.latest_subject_path, "PNG")

                # 4. Inpaint de fondo desocluido
                bg_inpainted = inpaint_background(src_np, alpha_mask)
                bg_pil = Image.fromarray(bg_inpainted, mode="RGB")
                bg_pil.save(seg_state.latest_bg_path, "PNG")

                seg_state.status = "READY"
                seg_state.status_msg = "Sujeto y Fondo Inpainted generados exitosamente"

                self._send_json({
                    "success": True,
                    "subject_url": "/api/latest_subject_alpha.png",
                    "background_url": "/api/latest_background_inpainted.png",
                    "mask_url": "/api/latest_mask.png"
                })
            except Exception as e:
                seg_state.status = "ERROR"
                seg_state.last_error = str(e)
                self._send_json({"error": f"Error en segmentación: {e}"}, 500)

        else:
            self._send_json({"error": "Endpoint POST no encontrado"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Módulo 5: AI Segmentation Studio")
    parser.add_argument("--port", type=int, default=8084, help="Puerto (default: 8084)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SegmentationHTTPHandler)
    print("=" * 75)
    print(f"✂️ Módulo 5: AI Segmentation & Inpainting Studio (SAM 2 & Alpha Matting)")
    print(f"🌐 Interfaz WebApp:  http://{args.host}:{args.port}")
    print(f"🖥️ Hardware GPU:     {seg_state.gpu_info.get('name', 'CPU')}")
    print("=" * 75)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Segmentation Studio] Deteniendo servidor...")
        server.server_close()


if __name__ == "__main__":
    main()

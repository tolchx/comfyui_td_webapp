#!/usr/bin/env python3
"""
Módulo 7 — Poster Studio (MOSS-Image)  ·  Puerto 8086

Dos modos:
  · híbrido  (default): MOSS-Image genera el ARTE y la tipografía se compone con fuentes
                         reales del sistema (Pillow). Texto exacto, editable, sin regenerar.
  · ia       : MOSS-Image dibuja también el texto, con layout por cajas (modo demo/cátedra).

Habla con la instancia de ComfyUI que tenga MOSS (por defecto 127.0.0.1:8189, aparte del 8188
de la suite, porque MOSS necesita ComfyUI >= 0.44 con nodes_moss_image).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mossbridge import graph, layout, typography          # noqa: E402
from mossbridge.comfy import ComfyClient, EngineError      # noqa: E402
from mossbridge.profiles import PROFILES                   # noqa: E402

WEBAPP = ROOT / "webapp"
CONFIG_PATH = ROOT / "config.json"

CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUTS = (ROOT / CFG.get("outputs_dir", "outputs")).resolve()
OUTPUTS.mkdir(parents=True, exist_ok=True)

LATEST = OUTPUTS / "latest_poster.png"
LATEST_ALIAS = OUTPUTS / "latest.png"          # por si TD usa la convención de los otros módulos

CLIENT = ComfyClient(CFG.get("comfy_host", "127.0.0.1"), int(CFG.get("comfy_port", 8189)))

JOBS: dict[str, dict] = {}
HISTORY: list[dict] = []
GPU_LOCK = threading.Lock()

RATIOS = {
    "1:1": (1024, 1024),
    "4:5": (896, 1120),
    "2:3": (832, 1248),
    "3:4": (896, 1200),
    "16:9": (1536, 864),
    "9:16": (864, 1536),
    "A4": (1248, 1760),
}


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def gpu_vram() -> dict | None:
    """VRAM usada/libre en MB (nvidia-smi). None si no hay NVIDIA."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        used, free, total = (int(v.strip()) for v in out.split(","))
        return {"used_mb": used, "free_mb": free, "total_mb": total}
    except Exception:
        return None


def free_engine_vram(min_free_mb: int = 18000) -> dict:
    """Pide al motor que descargue modelos si la VRAM libre no alcanza para MOSS.

    MOSS ocupa ~15 GB entre encoder y backbone. Si el motor tiene otros modelos
    cargados (o quedaron residentes de un job anterior), hay que pedirle que los suelte
    antes de arrancar; si no, el intercambio CPU↔GPU lo vuelve inusablemente lento.
    """
    before = gpu_vram()
    if before is None or before["free_mb"] >= min_free_mb:
        return {"freed": False, "before": before, "after": before}
    try:
        CLIENT._request("/free", data=b'{"unload_models":true,"free_memory":true}',
                        headers={"Content-Type": "application/json"}, timeout=60)
    except Exception:
        pass
    time.sleep(4)
    after = gpu_vram()
    return {"freed": True, "before": before, "after": after}



def clamp_canvas(w: int, h: int) -> tuple[int, int]:
    w = max(512, min(2048, int(w)))
    h = max(512, min(2048, int(h)))
    return (w // 16) * 16, (h // 16) * 16


def build_art_prompt(spec: dict) -> str:
    art = spec.get("art") or {}
    parts = [str(art.get("caption", "")).strip()]
    style = str(art.get("style", "")).strip()
    if style:
        parts.append(style)
    palette = str(art.get("palette", "")).strip()
    if palette:
        parts.append(f"Color palette: {palette}.")
    parts.append("Leave clean empty areas where text will sit. No typography, no letters, "
                 "no words, no watermarks." if spec.get("mode", "hybrid") == "hybrid" else "")
    return " ".join(p for p in parts if p)


def normalize_boxes(spec: dict, width: int, height: int) -> tuple[list[dict], list[dict]]:
    """Separa cajas de texto (van a Pillow en híbrido) de cajas de objeto (van al modelo)."""
    text_boxes, obj_boxes = [], []
    for raw in spec.get("boxes", []) or []:
        box = {
            "label": raw.get("label", "text"),
            "text": str(raw.get("text", "")),
            "caption": str(raw.get("caption", "") or "").strip(),
            "x": float(raw.get("x", 0.1)), "y": float(raw.get("y", 0.1)),
            "w": float(raw.get("w", 0.6)), "h": float(raw.get("h", 0.12)),
        }
        if box["label"] == "text":
            if not box["text"].strip():
                continue
            text_boxes.append({**box, **_font_fields(raw)})
        else:
            if not box["caption"]:
                continue
            obj_boxes.append(box)
    return text_boxes, obj_boxes


def _font_fields(raw: dict) -> dict:
    return {
        "font": raw.get("font") or None,
        "size": float(raw.get("size", 64)),
        "color": raw.get("color", "#ffffff"),
        "align": raw.get("align", "left"),
        "valign": raw.get("valign", "top"),
        "line_height": float(raw.get("line_height", 1.12)),
        "tracking": float(raw.get("tracking", 0)),
        "uppercase": bool(raw.get("uppercase", False)),
        "fit": bool(raw.get("fit", True)),
        "shadow": bool(raw.get("shadow", True)),
        "pad": float(raw.get("pad", 0)),
    }


def make_prompt(spec: dict, width: int, height: int, mode: str) -> str:
    """Híbrido: solo objetos como layout (o caption suelto). IA: texto y objetos como layout."""
    text_boxes, obj_boxes = normalize_boxes(spec, width, height)
    caption = build_art_prompt(spec)

    items = []
    for box in obj_boxes:
        items.append({"label": "object", "caption": box["caption"],
                      "bbox": (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])})
    if mode == "ia":
        for box in text_boxes:
            items.append({"label": "text", "text": box["text"],
                          "bbox": (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])})

    if not items:
        return caption
    try:
        compiled = layout.compile_layout(caption or "poster", items, width, height)
        return layout.dumps(compiled)
    except layout.LayoutError:
        # una caja fuera de rango no debe romper el job: se cae a prompt suelto
        return caption


def run_job(job_id: str, spec: dict) -> None:
    job = JOBS[job_id]
    started = time.time()
    try:
        mode = "ia" if spec.get("mode") == "ia" else "hybrid"
        width, height = clamp_canvas(spec.get("width", 1024), spec.get("height", 1024))
        profile = spec.get("profile") or CFG.get("default_profile", "quality25")
        if profile not in PROFILES:
            raise ValueError(f"perfil desconocido: {profile}")
        seed = int(spec.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big") % 2**31

        job.update(status="queued", message="esperando GPU", width=width, height=height,
                   mode=mode, profile=profile, seed=seed, started=now_iso())

        with GPU_LOCK:
            job.update(status="running", message="verificando VRAM")
            vram_info = free_engine_vram()
            job["vram"] = vram_info
            after = vram_info.get("after") or {}
            if after and after.get("free_mb", 0) < 14000:
                raise RuntimeError(
                    "VRAM insuficiente para MOSS: quedan "
                    f"{after['free_mb']} MB libres y necesita ~15 GB. "
                    "Cerrá el otro ComfyUI (8188), juegos o apps con GPU y reintentá."
                )
            job["message"] = (f"generando arte con MOSS-Image · "
                              f"VRAM libre {after.get('free_mb', '?')} MB")
            prompt = make_prompt(spec, width, height, mode)
            job["prompt_chars"] = len(prompt)
            g = graph.t2i(prompt, profile, width=width, height=height, seed=seed,
                          prefix="poster_moss")
            pid = CLIENT.submit(g)
            info = CLIENT.wait(pid, timeout=1200)
            raw = CLIENT.fetch(info["files"][0])

        tmp = OUTPUTS / f"_raw_{job_id}.png"
        tmp.write_bytes(raw)
        art = Image.open(tmp).convert("RGB")

        if mode == "hybrid":
            text_boxes, _ = normalize_boxes(spec, art.width, art.height)
            job["message"] = f"componiendo {len(text_boxes)} caja(s) de texto"
            final = typography.render_boxes(art, text_boxes)
        else:
            final = art

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"poster_{stamp}_{seed}.png"
        out_path = OUTPUTS / name
        final.save(out_path, "PNG")
        final.save(LATEST, "PNG")
        try:
            final.save(LATEST_ALIAS, "PNG")
        except Exception:
            pass
        tmp.unlink(missing_ok=True)

        record = {
            "file": name,
            "url": f"/outputs/{name}",
            "width": final.width, "height": final.height,
            "mode": mode, "profile": profile, "seed": seed,
            "seconds": round(time.time() - started, 1),
            "engine_seconds": info["seconds"],
            "created": now_iso(),
            "caption": (spec.get("art") or {}).get("caption", "")[:200],
        }
        HISTORY.insert(0, record)
        del HISTORY[24:]
        job.update(status="done", message="listo", result=record,
                   seconds=record["seconds"])
        print(f"[poster] OK {name} · {record['seconds']}s ({mode}, {profile}, seed {seed})", flush=True)
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", message=str(exc)[:500])
        print(f"[poster] ERROR {exc}\n{traceback.format_exc()}", flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "PosterStudio/1.0"

    def log_message(self, fmt, *args):  # silencio: el log propio alcanza
        return

    # ---------- helpers ----------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200):
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path, ctype: str):
        if not path.exists():
            self._json({"error": "no existe"}, 404)
            return
        self._send(200, path.read_bytes(), ctype)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ---------- GET ----------
    def do_GET(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._file(WEBAPP / "index.html", "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(self._status())
        elif path == "/api/fonts":
            self._json({"fonts": [f for f in CFG.get("fonts", []) if os.path.exists(f["path"])],
                        "ratios": RATIOS, "profiles": {
                            k: {"steps": v["steps"], "cfg": v["cfg"], "text_safe": v["text_safe"],
                                "note": v["note"]} for k, v in PROFILES.items()}})
        elif path == "/api/history":
            self._json({"history": HISTORY})
        elif path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            job = JOBS.get(job_id)
            self._json(job or {"error": "job desconocido"}, 200 if job else 404)
        elif path == "/api/latest.png":
            self._file(LATEST, "image/png")
        elif path.startswith("/outputs/"):
            name = os.path.basename(path)
            self._file(OUTPUTS / name, "image/png")
        elif path == "/api/preview_prompt":
            self._json({"error": "usar POST"}, 405)
        else:
            self._json({"error": f"ruta desconocida {path}"}, 404)

    # ---------- POST ----------
    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        body = self._body()

        if path == "/api/generate":
            width, height = clamp_canvas(body.get("width", 1024), body.get("height", 1024))
            body["width"], body["height"] = width, height
            job_id = uuid.uuid4().hex[:12]
            JOBS[job_id] = {"id": job_id, "status": "starting", "created": now_iso()}
            threading.Thread(target=run_job, args=(job_id, body), daemon=True).start()
            self._json({"success": True, "job_id": job_id, "width": width, "height": height})

        elif path == "/api/preview_prompt":
            mode = "ia" if body.get("mode") == "ia" else "hybrid"
            width, height = clamp_canvas(body.get("width", 1024), body.get("height", 1024))
            self._json({"prompt": make_prompt(body, width, height, mode)})

        elif path == "/api/open_folder":
            try:
                os.startfile(OUTPUTS)  # type: ignore[attr-defined]
                self._json({"success": True})
            except Exception as exc:  # noqa: BLE001
                self._json({"success": False, "error": str(exc)}, 500)

        elif path == "/api/free_vram":
            self._json({"success": True, **free_engine_vram(min_free_mb=10**9)})
        else:
            self._json({"error": f"ruta desconocida {path}"}, 404)

    # ---------- estado ----------
    def _status(self) -> dict:
        stats = None
        try:
            stats = CLIENT.stats()
        except Exception:
            pass
        vram = None
        try:
            vram = gpu_vram()
        except Exception:
            pass
        return {
            "ok": True,
            "engine": {
                "url": CLIENT.base,
                "online": stats is not None,
                "version": ((stats or {}).get("system") or {}).get("comfyui_version"),
                "mode_moss": bool(stats and self._has_moss()),
                "queue": CLIENT.queue_depth() if stats else -1,
            },
            "vram": vram,
            "outputs": str(OUTPUTS),
            "latest": HISTORY[0] if HISTORY else None,
            "running": any(j.get("status") in ("queued", "running") for j in JOBS.values()),
        }

    def _has_moss(self) -> bool:
        try:
            info = CLIENT.object_info()
            return "TextEncodeMOSSImageEdit" in info and "MOSSImageLayoutPrompt" in info
        except Exception:
            return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Poster Studio (MOSS-Image) — Módulo 7")
    ap.add_argument("--port", type=int, default=int(CFG.get("port", 8086)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print("=" * 70)
    print("  MÓDULO 7 · POSTER STUDIO (MOSS-Image)")
    print(f"  WebApp:  http://{args.host}:{args.port}")
    print(f"  Motor :  {CLIENT.base}  ({'online' if CLIENT.alive() else 'OFFLINE'})")
    print(f"  Salida:  {OUTPUTS}")
    print("=" * 70, flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[poster] cerrando", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

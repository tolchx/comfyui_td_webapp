"""Prueba de humo del Módulo 7: dispara un job por modo y espera el resultado."""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8086"


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


SPEC = {
    "width": 1024, "height": 1024, "profile": "quality25", "seed": 7,
    "art": {
        "caption": "Cartel nocturno para una fiesta de música electrónica en un puerto: fondo negro absoluto con retícula fina, halo cian detrás de una pila de parlantes en silueta.",
        "style": "tipografía suiza, minimal, mucho aire",
        "palette": "negro, cian",
    },
    "boxes": [
        {"label": "text", "text": "NOCTURNO", "x": 0.08, "y": 0.10, "w": 0.84, "h": 0.16,
         "size": 118, "color": "#ffffff", "align": "left", "valign": "top", "uppercase": True,
         "line_height": 1.05, "tracking": -1, "fit": True, "shadow": True,
         "font": "C:/Windows/Fonts/segoeuib.ttf"},
        {"label": "text", "text": "Música electrónica · Puerto de Buenos Aires · Sábado 22",
         "x": 0.08, "y": 0.285, "w": 0.84, "h": 0.06, "size": 25, "color": "#7dd3fc",
         "align": "left", "valign": "top", "uppercase": False, "line_height": 1.2,
         "tracking": 1.5, "fit": True, "shadow": True, "font": "C:/Windows/Fonts/segoeui.ttf"},
        {"label": "object", "caption": "pila de parlantes en silueta con halo cian",
         "x": 0.28, "y": 0.55, "w": 0.44, "h": 0.36},
    ],
}


def run(mode, seed):
    spec = dict(SPEC, mode=mode, seed=seed)
    print(f"\n>>> modo {mode} (seed {seed})", flush=True)
    out = post("/api/generate", spec)
    print("    job:", out.get("job_id"), out.get("error", ""), flush=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(2)
        job = get("/api/job/" + out["job_id"])
        if job.get("status") in ("done", "error"):
            print("   ", job.get("status"), json.dumps(job.get("result"), ensure_ascii=False)[:300],
                  job.get("message", ""), flush=True)
            return job
        print("   ", job.get("status"), job.get("message", ""), flush=True)
    print("    TIMEOUT", flush=True)
    return None


if __name__ == "__main__":
    modes = sys.argv[1:] or ["hybrid", "ia"]
    for i, mode in enumerate(modes):
        run(mode, 7 + i)

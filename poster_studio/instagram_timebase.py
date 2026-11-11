"""Post de Instagram: actualización de timeBase a TD 2025.32460 — 4:5 (1088×1360).

Modo híbrido: MOSS hace el arte, la tipografía se compone con fuentes reales.
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8086"

F_SANS_B = "C:/Windows/Fonts/segoeuib.ttf"
F_SANS = "C:/Windows/Fonts/segoeui.ttf"
F_MONO_B = "C:/Windows/Fonts/consolab.ttf"

SPEC = {
    "mode": "hybrid",
    "width": 896,
    "height": 1120,
    "upscale_to": {"width": 1088, "height": 1360},
    "profile": "quality25",
    "seed": int(sys.argv[1]) if len(sys.argv) > 1 else 314,
    "art": {
        "caption": (
            "Abstract real-time media server interface, very dark charcoal background: "
            "stacked timeline layers in perspective made of thin glowing cyan and violet lines, "
            "soft waveform curves, faint HUD grid, subtle volumetric haze, a few small node dots "
            "connected by hairlines. Centered vertical composition with large empty dark areas at "
            "the top and around the middle. Cinematic, precise, technical, high contrast."
        ),
        "style": "UI técnico abstracto, glow suave, mucho espacio negativo",
        "palette": "negro carbón, cian eléctrico, violeta",
    },
    "boxes": [
        {"label": "text", "text": "TOUCHDESIGNER · 2025.32460",
         "x": .075, "y": .055, "w": .85, "h": .026, "size": 23,
         "font": F_MONO_B, "color": "#7dd3fc", "tracking": 3.4, "uppercase": True,
         "align": "left", "valign": "top", "shadow": False, "fit": True},
        {"label": "text", "text": "timeBase",
         "x": .068, "y": .090, "w": .86, "h": .115, "size": 152,
         "font": F_SANS_B, "color": "#ffffff", "tracking": -4.5, "uppercase": False,
         "align": "left", "valign": "top", "line_height": 1.0, "shadow": True, "fit": True},
        {"label": "text", "text": "Timeline multi-capa de Greg Hermanovic",
         "x": .075, "y": .232, "w": .85, "h": .040, "size": 33,
         "font": F_SANS, "color": "#c7cede", "tracking": 0.2, "uppercase": False,
         "align": "left", "valign": "top", "shadow": True, "fit": True},
        {"label": "text", "text": "0 errores · 0 warnings",
         "x": .075, "y": .318, "w": .85, "h": .058, "size": 57,
         "font": F_SANS_B, "color": "#38bdf8", "tracking": -1.2, "uppercase": False,
         "align": "left", "valign": "top", "line_height": 1.05, "shadow": True, "fit": True},
        {"label": "text", "text": "CAMBIOS CLAVE",
         "x": .075, "y": .414, "w": .85, "h": .024, "size": 19,
         "font": F_MONO_B, "color": "#64748b", "tracking": 3.6, "uppercase": True,
         "align": "left", "valign": "top", "shadow": False, "fit": True},
        {"label": "text",
         "text": ("· 585 exports huérfanos eliminados y bucles de UI TUIK resueltos\n"
                  "· Rutas dinámicas de samples, moviePlayer/trimming y fuentes\n"
                  "· Fixes de Python (saveRestore, startup, copiar/pegar)\n"
                  "· Soporte para 10+ escenas, capas y elementos"),
         "x": .075, "y": .452, "w": .855, "h": .205, "size": 25,
         "font": F_SANS, "color": "#e8ecf6", "tracking": 0.1, "uppercase": False,
         "align": "left", "valign": "top", "line_height": 1.62, "shadow": True, "fit": True,
         "bg": "#05070ccc", "bg_pad": 18, "bg_radius": 10, "pad": 7},
        {"label": "text", "text": "Gratis · .tox + proyecto .toe demo",
         "x": .075, "y": .903, "w": .85, "h": .030, "size": 24,
         "font": F_SANS, "color": "#aeb6c8", "bg": "#05070c99", "bg_pad": 9, "bg_radius": 6, "uppercase": False,
         "align": "left", "valign": "top", "shadow": True, "fit": True},
        {"label": "text", "text": "@tolch.x",
         "x": .70, "y": .903, "w": .225, "h": .030, "size": 25,
         "font": F_SANS_B, "color": "#ffffff", "uppercase": False,
         "align": "right", "valign": "top", "shadow": True, "fit": True},
    ],
}


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


job = post("/api/generate", SPEC)
print("job", job.get("job_id"), job.get("error", ""))
t0 = time.time()
while time.time() - t0 < 900:
    time.sleep(2)
    j = get("/api/job/" + job["job_id"])
    if j.get("status") in ("done", "error"):
        print(j.get("status"), json.dumps(j.get("result"), ensure_ascii=False, indent=1))
        if j.get("status") == "error":
            print("MSG:", j.get("message"))
            sys.exit(1)
        break
    print("  ", j.get("status"), j.get("message", ""), flush=True)

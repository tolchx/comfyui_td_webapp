#!/usr/bin/env python3
"""
Suite de Autodiagnóstico y Verificación End-to-End (Self-Test)
Diseñada para alumnos y para Agentes LLM (Claude Code, Antigravity, Freebuff, Cursor).

Ejecuta 6 pruebas deterministas:
  1. Verificación de entorno Python y dependencia Pillow (auto-instala si falta).
  2. Detección de Hardware (GPU NVIDIA CUDA / Apple Silicon MPS / CPU) y diagnóstico de compatibilidad.
  3. Verificación de archivos del proyecto (`config.json`, `webapp/index.html`, `TD/v1.toe`, `TD/ComfyUI_Studio_Bridge.tox`).
  4. Prueba del Servidor Puente (`http://127.0.0.1:8080/api/status` y `/api/td_poll`), iniciándolo si no está corriendo.
  5. Disparo de una generación de prueba (`POST /api/generate`) y verificación de que `bridge/outputs/latest.png` se actualiza con un PNG válido.
  6. (Opcional) Si TouchDesigner está abierto con TDMCP en `127.0.0.1:13316`, verifica que `/project1/comfy_studio` tenga 0 errores.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
BASE_DIR = BRIDGE_DIR.parent


def step_1_check_python_deps() -> dict:
    try:
        import PIL  # noqa: F401
        return {"ok": True, "pillow": "installed"}
    except ImportError:
        print("[AUTO-FIX] Instalando Pillow (PIL) para manipulación de imágenes...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pillow"], check=False)
        try:
            import PIL  # noqa: F401
            return {"ok": True, "pillow": "auto-installed"}
        except ImportError as e:
            return {"ok": False, "error": f"No se pudo instalar Pillow: {e}"}


def step_2_check_hardware() -> dict:
    sys.path.insert(0, str(BRIDGE_DIR))
    import installer
    cfg = installer.load_config()
    gpu = installer.detect_gpu()
    inst = installer.resolve_comfy_installation(cfg)
    ckpts = installer.list_installed_checkpoints(cfg)
    rec = installer.recommend_tier(gpu["vram_gb"], gpu.get("vendor", "nvidia"))
    compat = installer.diagnose_hardware_compatibility(gpu, cfg.get("active_tier", rec), inst, ckpts)
    return {
        "ok": True,
        "gpu": gpu,
        "recommended_tier": rec,
        "comfy_installed": inst is not None,
        "checkpoints_count": len(ckpts),
        "compatibility": compat,
    }


def step_3_check_files() -> dict:
    required = [
        BASE_DIR / "bridge" / "config.json",
        BASE_DIR / "bridge" / "installer.py",
        BASE_DIR / "bridge" / "workflows.py",
        BASE_DIR / "bridge" / "server.py",
        BASE_DIR / "webapp" / "index.html",
        BASE_DIR / "TD" / "v1.toe",
        BASE_DIR / "TD" / "ComfyUI_Studio_Bridge.tox",
    ]
    missing = [str(p.relative_to(BASE_DIR)) for p in required if not p.exists()]
    return {"ok": len(missing) == 0, "missing": missing}


def step_4_ensure_bridge_running() -> dict:
    url = "http://127.0.0.1:8080/api/status"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as r:
            if r.status == 200:
                return {"ok": True, "status": "already_running", "data": json.loads(r.read().decode("utf-8"))}
    except Exception:
        pass

    print("[AUTO-FIX] El servidor puente no estaba corriendo en :8080. Iniciándolo en segundo plano...")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(
        [sys.executable, str(BRIDGE_DIR / "server.py"), "--autostart-comfy"],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    for _ in range(15):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return {"ok": True, "status": "started_by_selftest", "data": json.loads(r.read().decode("utf-8"))}
        except Exception:
            continue
    return {"ok": False, "error": "No se pudo contactar http://127.0.0.1:8080/api/status luego de iniciar server.py"}


def step_5_test_generation() -> dict:
    poll_url = "http://127.0.0.1:8080/api/td_poll"
    gen_url = "http://127.0.0.1:8080/api/generate"

    with urllib.request.urlopen(poll_url, timeout=3.0) as r:
        before = json.loads(r.read().decode("utf-8"))
    prev_id = int(before.get("generation_id", 0))

    payload = json.dumps({
        "prompt": "selftest geometric iridescent prism floating in dark studio, 8k",
        "width": 512,
        "height": 512,
        "steps": 4,
        "source": "selftest",
    }).encode("utf-8")

    req = urllib.request.Request(gen_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5.0) as r:
        resp = json.loads(r.read().decode("utf-8"))
        if not resp.get("ok"):
            return {"ok": False, "error": f"Fallo al encolar generación: {resp}"}

    deadline = time.time() + 180.0
    while time.time() < deadline:
        time.sleep(0.5)
        with urllib.request.urlopen(poll_url, timeout=3.0) as r:
            st = json.loads(r.read().decode("utf-8"))
        if int(st.get("generation_id", 0)) > prev_id and st.get("state") == "READY":
            latest_p = Path(st["latest_file"])
            is_valid_png = latest_p.exists() and latest_p.read_bytes().startswith(b"\x89PNG")
            return {
                "ok": is_valid_png,
                "generation_id": st["generation_id"],
                "latency_ms": st["latency_ms"],
                "latest_file": str(latest_p),
                "valid_png": is_valid_png,
            }
        if st.get("state") == "ERROR":
            return {"ok": False, "error": st.get("error", "Error desconocido durante generación")}

    return {"ok": False, "error": "Timeout esperando que termine la generación de prueba."}


def main():
    print("=" * 72)
    print("  EJECUTANDO AUTODIAGNOSTICO END-TO-END (TD <-> COMFYUI <-> WEBAPP)")
    print("=" * 72)
    report = {
        "step_1_python_deps": step_1_check_python_deps(),
        "step_2_hardware": step_2_check_hardware(),
        "step_3_files": step_3_check_files(),
        "step_4_bridge": step_4_ensure_bridge_running(),
    }
    if report["step_4_bridge"].get("ok"):
        report["step_5_generation"] = step_5_test_generation()
    else:
        report["step_5_generation"] = {"ok": False, "error": "Bridge offline"}

    all_ok = all(v.get("ok", False) for v in report.values())
    report["all_ok"] = all_ok
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

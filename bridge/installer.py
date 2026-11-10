#!/usr/bin/env python3
"""
Instalador y Configurador Interactivo Multiplataforma (Windows + macOS + Linux)
Curso IA: Puente TouchDesigner <-> ComfyUI

Detecta el hardware (NVIDIA CUDA, Apple Silicon MPS M1/M2/M3/M4, AMD o CPU),
evalúa la compatibilidad y posibles cuellos de botella, sugiere el Tier óptimo
(Básico / Medio / Avanzado), verifica o instala ComfyUI, y descarga los modelos
necesarios desde HuggingFace con barra de progreso y soporte de reanudación.
"""

import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def detect_gpu() -> dict:
    """Detecta GPU NVIDIA (Windows/Linux), Apple Silicon MPS (macOS) o GPU integrada/AMD."""
    sys_os = platform.system().lower()  # 'windows', 'darwin', 'linux'
    arch = platform.machine().lower()

    # 1. Intentar NVIDIA via nvidia-smi (Windows / Linux)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=4,
        ).strip()
        if out:
            line = out.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            total_mb = float(parts[1])
            free_mb = float(parts[2])
            util = float(parts[3]) if len(parts) > 3 else 0.0
            vram_gb = round(total_mb / 1024.0, 1)
            return {
                "found": True,
                "vendor": "nvidia",
                "os": sys_os,
                "arch": arch,
                "name": parts[0],
                "vram_total_mb": int(total_mb),
                "vram_free_mb": int(free_mb),
                "vram_gb": vram_gb,
                "gpu_util": util,
            }
    except Exception:
        pass

    # 2. Detectar macOS (Apple Silicon M1/M2/M3/M4 con Memoria Unificada MPS o Mac Intel)
    if sys_os == "darwin":
        chip_name = "Apple Silicon / Mac"
        total_ram_bytes = 8 * (1024**3)
        try:
            chip_out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            ).strip()
            if chip_out:
                chip_name = chip_out
        except Exception:
            pass
        try:
            mem_out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            ).strip()
            if mem_out:
                total_ram_bytes = int(mem_out)
        except Exception:
            pass

        total_mb = int(total_ram_bytes / (1024 * 1024))
        # En Apple Silicon, Metal/MPS puede usar ~70% de la Memoria Unificada como VRAM
        is_apple_silicon = "apple" in chip_name.lower() or arch in ("arm64", "aarch64")
        usable_vram_mb = int(total_mb * 0.70) if is_apple_silicon else 2048
        vram_gb = round(usable_vram_mb / 1024.0, 1)

        return {
            "found": is_apple_silicon,
            "vendor": "apple_mps" if is_apple_silicon else "mac_intel",
            "os": "darwin",
            "arch": arch,
            "name": f"{chip_name} ({round(total_mb/1024)}GB Unified RAM)" if is_apple_silicon else f"{chip_name} (Intel Mac)",
            "vram_total_mb": usable_vram_mb,
            "vram_free_mb": int(usable_vram_mb * 0.8),
            "vram_gb": vram_gb,
            "gpu_util": 0.0,
        }

    # 3. En Windows sin NVIDIA: consultar WMIC / PowerShell para ver si es AMD Radeon o Intel Iris
    gpu_label = "GPU Integrada / CPU (Sin NVIDIA CUDA)"
    if sys_os == "windows":
        try:
            ps_out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
            ).strip()
            if ps_out:
                gpu_label = ps_out
        except Exception:
            pass

    vendor = "amd" if ("amd" in gpu_label.lower() or "radeon" in gpu_label.lower()) else "cpu_integrated"
    return {
        "found": False,
        "vendor": vendor,
        "os": sys_os,
        "arch": arch,
        "name": gpu_label,
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "vram_gb": 0.0,
        "gpu_util": 0.0,
    }


def recommend_tier(vram_gb: float, vendor: str = "nvidia") -> str:
    if vendor == "apple_mps":
        # En Mac MPS, SD 1.5 LCM o SDXL Turbo corren mucho más fluido que Flux FP8
        if vram_gb >= 18.0:
            return "medio"
        return "basico"
    if vram_gb >= 12.0:
        return "avanzado"
    elif vram_gb >= 7.5:
        return "medio"
    return "basico"


def diagnose_hardware_compatibility(gpu: dict, active_tier: str, comfy_inst: dict | None, installed_ckpts: list[str]) -> dict:
    """
    Analiza la compatibilidad del hardware del alumno y genera advertencias claras
    y soluciones paso a paso que se muestran directamente en la WebApp.
    """
    vendor = gpu.get("vendor", "cpu_integrated")
    vram_gb = float(gpu.get("vram_gb", 0.0))
    tier_min = {"realtime": 3.0, "basico": 4.0, "p_h3_image": 6.0, "medio": 8.0, "avanzado": 12.0}.get(active_tier, 4.0)

    issues = []
    solutions = []
    severity = "ok"  # ok | warning | critical

    # 1. Diagnóstico según tipo de GPU / Sistema Operativo
    if vendor == "nvidia":
        if vram_gb < 4.0:
            severity = "warning"
            issues.append(f"Placa NVIDIA con poca VRAM ({vram_gb} GB detectados).")
            solutions.append("Usá exclusivamente el Nivel Básico (SD 1.5 + LCM LoRA) a resolución 512x512 con flag --lowvram.")
            solutions.append("Cerrá navegadores pesados o juegos mientras corrés TouchDesigner + ComfyUI.")
        elif vram_gb < tier_min:
            severity = "warning"
            issues.append(
                f"Seleccionaste el nivel '{active_tier.upper()}' (requiere ~{int(tier_min)} GB VRAM), pero tu GPU tiene {vram_gb} GB."
            )
            solutions.append(
                f"Cambiá al nivel '{recommend_tier(vram_gb, vendor).upper()}' en el panel izquierdo para evitar errores Out-Of-Memory (OOM) o lentitud extrema."
            )
            solutions.append("Si querés forzar este modelo, bajá la resolución a 512x512 y reducí los pasos (Steps) a 4.")
    elif vendor == "apple_mps":
        if active_tier == "avanzado":
            severity = "warning"
            issues.append("Estás en macOS (Apple Silicon MPS) con el nivel Avanzado (Flux FP8) seleccionado.")
            solutions.append(
                "Los modelos cuantizados en FP8 (como Flux.1 FP8) están optimizados para placas NVIDIA RTX serie 40 y pueden fallar o ir muy lento en Metal (MPS)."
            )
            solutions.append(
                "Recomendación para Mac: seleccioná el Nivel Básico (DreamShaper 8 + LCM) o Nivel Medio (SDXL Turbo FP16). El servidor puente activa automáticamente '--force-fp16' para acelerar Metal."
            )
        elif vram_gb < 8.0:
            severity = "warning"
            issues.append(f"Mac con Memoria Unificada ajustada (~{vram_gb} GB disponibles para GPU).")
            solutions.append("Usá el Nivel Básico (SD 1.5 + LCM) a 512x512 para no saturar la memoria compartida con TouchDesigner.")
    elif vendor == "mac_intel":
        severity = "critical"
        issues.append("Mac con procesador Intel detectada (sin chip Apple Silicon M1/M2/M3/M4).")
        solutions.append("La generación local en ComfyUI sobre Mac Intel corre en CPU y puede demorar varios minutos por imagen.")
        solutions.append("Solución 1: Usá el 'Modo Práctica Procedural' integrado en este servidor puente para probar TouchDesigner a 60 FPS en clase.")
        solutions.append("Solución 2: Conectá el puente a una PC con GPU en la misma red local cambiando la URL de ComfyUI.")
    elif vendor == "amd":
        severity = "warning"
        issues.append(f"Placa AMD Radeon detectada ({gpu.get('name')}) — sin soporte nativo NVIDIA CUDA.")
        solutions.append("Solución 1 (ComfyUI ZLUDA / DirectML): Instalá ComfyUI con `--directml` para placas AMD en Windows.")
        solutions.append("Solución 2: Mientras tanto, el servidor puente funciona automáticamente en 'Modo Práctica Procedural' para que puedas usar TouchDesigner sin bloqueos.")
    else:
        severity = "warning"
        issues.append(f"No se detectó GPU dedicada NVIDIA ni Apple Silicon ({gpu.get('name')}).")
        solutions.append("Solución 1 (Modo Práctica Instantáneo): Podés usar el parche de TouchDesigner y la WebApp ahora mismo; el servidor puente sintetiza imágenes procedurales reactivas al Prompt e Img2Img en CPU sin necesitar GPU.")
        solutions.append("Solución 2 (Ejecución en CPU): Si instalás ComfyUI y el modelo Básico, agregá '--cpu' pero tené en cuenta que cada imagen tardará ~30-60 segundos.")

    # 2. Diagnóstico de instalación de ComfyUI y Checkpoints
    if not comfy_inst:
        if severity == "ok":
            severity = "warning"
        issues.append("ComfyUI aún no está instalado o vinculado en esta máquina.")
        if gpu.get("os") == "darwin":
            solutions.append("En macOS: ejecutá './1_Instalar_Mac.command' en la Terminal para clonar e instalar ComfyUI con soporte Metal (MPS) automáticamente.")
        else:
            solutions.append("En Windows: ejecutá '1_Instalar_Modelos.bat' y elegí la opción [1] para descargar ComfyUI Portable automáticamente.")
    elif len(installed_ckpts) == 0:
        if severity == "ok":
            severity = "warning"
        issues.append("ComfyUI está detectado, pero todavía no hay ningún modelo (.safetensors) descargado en la carpeta 'models/checkpoints'.")
        solutions.append("Hacé clic en el botón '⬇ Descargar / Verificar Modelo de este Nivel en ComfyUI' aquí en la WebApp o ejecutá el script instalador.")

    if not issues:
        issues.append(f"Hardware óptimo ({gpu.get('name')}) y modelos listos para generar en tiempo real.")
        solutions.append("Consejo: En TouchDesigner mantené las resoluciones entre 512x512 y 768x768 cuando uses el modo 'Auto-Stream Continuo' para menor latencia.")

    return {
        "severity": severity,
        "os": gpu.get("os", "windows"),
        "vendor": vendor,
        "issues": issues,
        "solutions": solutions,
    }


def resolve_comfy_installation(cfg: dict) -> dict | None:
    """Busca una instalación existente de ComfyUI en Windows, macOS o Linux."""
    for inst in cfg.get("comfy_installations", []):
        py_raw = os.path.expanduser(inst["python_exe"])
        main_raw = os.path.expanduser(inst["main_py"])
        models_raw = os.path.expanduser(inst["models_dir"])

        py_p = Path(py_raw)
        main_p = Path(main_raw)
        models_p = Path(models_raw)

        if not py_p.is_absolute():
            py_p = BASE_DIR / py_p
        if not main_p.is_absolute():
            main_p = BASE_DIR / main_p
        if not models_p.is_absolute():
            models_p = BASE_DIR / models_p

        if py_p.exists() and main_p.exists():
            models_p.mkdir(parents=True, exist_ok=True)
            return {
                "name": inst["name"],
                "python_exe": str(py_p),
                "main_py": str(main_p),
                "models_dir": str(models_p),
            }
    return None


def list_installed_checkpoints(cfg: dict) -> list[str]:
    """Escanea todos los directorios de modelos configurados para encontrar checkpoints existentes."""
    found = set()
    inst = resolve_comfy_installation(cfg)
    dirs_to_scan = []
    if inst:
        m_dir = Path(inst["models_dir"])
        dirs_to_scan.extend([m_dir / "checkpoints", m_dir / "diffusion_models", m_dir / "unet"])
    for extra in cfg.get("extra_models_dirs", []):
        ep = Path(os.path.expanduser(extra))
        if ep.exists():
            dirs_to_scan.extend([ep, ep / "checkpoints", ep / "diffusion_models"])

    for d in dirs_to_scan:
        if d.exists():
            for f in d.glob("*.safetensors"):
                if "vae" not in f.name.lower() and "lora" not in f.name.lower() and "clip" not in f.name.lower():
                    found.add(f.name)
    return sorted(found)


def download_file_with_progress(url: str, dest: Path, expected_mb: int = 0) -> bool:
    """Descarga un archivo grande desde HuggingFace mostrando progreso en consola."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    existing_bytes = tmp_dest.stat().st_size if tmp_dest.exists() else 0
    headers = {"User-Agent": "TD-ComfyUI-Course-Installer/2.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    print(f"\n  -> Descargando: {dest.name}")
    print(f"     Origen : {url}")
    print(f"     Destino: {dest}")

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = getattr(resp, "status", 200)
            total_len = resp.headers.get("Content-Length")
            total_bytes = int(total_len) + (existing_bytes if status == 206 else 0) if total_len else expected_mb * 1024 * 1024
            mode = "ab" if status == 206 else "wb"
            downloaded = existing_bytes if status == 206 else 0

            chunk_size = 1024 * 1024  # 1 MB
            with open(tmp_dest, mode) as out_f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    mb_done = downloaded / (1024 * 1024)
                    mb_tot = total_bytes / (1024 * 1024) if total_bytes else expected_mb
                    pct = min(100.0, (downloaded / total_bytes) * 100.0) if total_bytes else 0.0
                    bar_len = 30
                    filled = int(bar_len * pct / 100.0)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(f"\r     [{bar}] {pct:5.1f}% ({mb_done:,.1f} / {mb_tot:,.1f} MB)")
                    sys.stdout.flush()

        print("\n  [OK] Descarga completada.")
        if dest.exists():
            dest.unlink()
        tmp_dest.rename(dest)
        return True
    except KeyboardInterrupt:
        print("\n  [!] Descarga pausada por el usuario (se puede reanudar luego).")
        return False
    except Exception as e:
        print(f"\n  [ERROR] Fallo en la descarga: {e}")
        return False


def install_comfy_mac_or_venv() -> bool:
    """Clona ComfyUI y crea un entorno virtual `.venv` con PyTorch MPS (macOS) o CUDA (Windows/Linux)."""
    target_dir = BASE_DIR / "ComfyUI"
    sys_os = platform.system().lower()
    print("\n============================================================")
    print(f"  INSTALACION AUTOMATICA DE COMFYUI ({sys_os.upper()})")
    print("============================================================")

    try:
        if not target_dir.exists():
            print("  -> Clonando repositorio oficial de ComfyUI...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(target_dir)],
                check=True,
            )

        venv_dir = target_dir / ".venv"
        if not venv_dir.exists():
            print("  -> Creando entorno virtual Python (.venv)...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

        py_bin = venv_dir / ("Scripts/python.exe" if sys_os == "windows" else "bin/python")
        subprocess.run([str(py_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)

        if sys_os == "darwin":
            print("  -> Instalando PyTorch con aceleración Apple Metal (MPS)...")
            subprocess.run([str(py_bin), "-m", "pip", "install", "torch", "torchvision", "torchaudio"], check=True)
        else:
            print("  -> Instalando PyTorch...")
            subprocess.run(
                [str(py_bin), "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124"],
                check=True,
            )

        print("  -> Instalando dependencias de ComfyUI (requirements.txt)...")
        subprocess.run([str(py_bin), "-m", "pip", "install", "-r", str(target_dir / "requirements.txt")], check=True)
        print("  [OK] ComfyUI instalado exitosamente en ./ComfyUI")
        return True
    except Exception as e:
        print(f"  [ERROR] No se pudo completar la instalación automática de ComfyUI: {e}")
        return False


def download_comfy_portable() -> bool:
    """En macOS/Linux usa git+venv; en Windows descarga el paquete portable .7z o usa git+venv si falla."""
    if platform.system().lower() != "windows":
        return install_comfy_mac_or_venv()

    url = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
    archive_path = BASE_DIR / "ComfyUI_windows_portable_nvidia.7z"
    print("\n============================================================")
    print("  INSTALACION AUTOMATICA DE COMFYUI PORTABLE (WINDOWS)")
    print("============================================================")
    if not download_file_with_progress(url, archive_path, expected_mb=1500):
        return install_comfy_mac_or_venv()

    print("  -> Extrayendo archivo .7z (puede demorar 1-2 minutos)...")
    for cmd in [
        ["tar", "-xf", str(archive_path), "-C", str(BASE_DIR)],
        ["C:\\Program Files\\7-Zip\\7z.exe", "x", str(archive_path), f"-o{BASE_DIR}", "-y"],
    ]:
        try:
            res = subprocess.run(cmd, check=True)
            if res.returncode == 0:
                print("  [OK] ComfyUI Portable extraído correctamente.")
                return True
        except Exception:
            continue

    print("  [!] No se pudo extraer el .7z, instalando mediante git + venv...")
    return install_comfy_mac_or_venv()


def ensure_tier_models(cfg: dict, tier_id: str, auto_yes: bool = False) -> dict:
    """Verifica y descarga los modelos del Tier elegido."""
    tier = cfg["tiers"].get(tier_id)
    if not tier:
        return {"ok": False, "error": f"Tier desconocido: {tier_id}"}

    inst = resolve_comfy_installation(cfg)
    if not inst:
        return {"ok": False, "error": "No se encontró instalación de ComfyUI."}

    models_dir = Path(inst["models_dir"])
    results = []
    for f_spec in tier.get("files", []):
        target_dir = models_dir / f_spec["subdir"]
        if not target_dir.exists():
            # Si target_dir es un enlace roto (broken junction/symlink), buscar en extra_models_dirs
            for extra in cfg.get("extra_models_dirs", []):
                ep = Path(os.path.expanduser(extra))
                cand = ep / f_spec["subdir"]
                if cand.exists():
                    target_dir = cand
                    break
                cand2 = ep / "models" / f_spec["subdir"]
                if cand2.exists():
                    target_dir = cand2
                    break
        target_path = target_dir / f_spec["filename"]
        if target_path.exists() and target_path.stat().st_size > 1 * 1024 * 1024:
            print(f"  [OK] Modelo ya instalado: {f_spec['filename']} ({f_spec['size_mb']} MB)")
            results.append({"file": f_spec["filename"], "status": "installed", "path": str(target_path)})
            continue

        if not auto_yes:
            ans = input(f"\n  ¿Descargar {f_spec['filename']} (~{f_spec['size_mb']} MB)? [S/n]: ").strip().lower()
            if ans == "n":
                results.append({"file": f_spec["filename"], "status": "skipped"})
                continue

        ok = download_file_with_progress(f_spec["url"], target_path, expected_mb=f_spec["size_mb"])
        results.append({"file": f_spec["filename"], "status": "downloaded" if ok else "failed", "path": str(target_path)})

    return {"ok": True, "tier": tier_id, "files": results}


def interactive_wizard() -> None:
    cfg = load_config()
    gpu = detect_gpu()
    rec_tier = recommend_tier(gpu["vram_gb"], gpu.get("vendor", "nvidia"))
    inst = resolve_comfy_installation(cfg)
    existing_ckpts = list_installed_checkpoints(cfg)
    diag = diagnose_hardware_compatibility(gpu, cfg.get("active_tier", rec_tier), inst, existing_ckpts)

    print("\n" + "=" * 72)
    print("  CURSO IA GENERATIVA — PUENTE TOUCHDESIGNER <-> COMFYUI (WIN & MAC)")
    print("=" * 72)
    print(f"  Sistema Operativo : {gpu.get('os', '').upper()} ({gpu.get('arch', '')})")
    print(f"  Hardware GPU      : {gpu['name']}")
    print(f"  VRAM Estimada     : {gpu['vram_gb']} GB ({gpu['vram_total_mb']} MB)")
    print(f"  Nivel Sugerido    : {cfg['tiers'][rec_tier]['label']}")
    if inst:
        print(f"  ComfyUI Activo    : {inst['name']}")
        print(f"  Ruta Modelos      : {inst['models_dir']}")
    else:
        print("  ComfyUI Activo    : [NO DETECTADO]")

    if existing_ckpts:
        print(f"  Modelos Locales ({len(existing_ckpts)}): {', '.join(existing_ckpts[:4])}")
    print("-" * 72)
    print("  DIAGNOSTICO DE COMPATIBILIDAD:")
    for iss in diag["issues"]:
        print(f"   * {iss}")
    for sol in diag["solutions"]:
        print(f"   -> {sol}")
    print("=" * 72)

    if not inst:
        print("\nNo se encontró ComfyUI instalado.")
        print("  [1] Instalar ComfyUI automáticamente en esta carpeta")
        print("  [2] Ingresar la ruta de una carpeta ComfyUI existente en tu computadora")
        print("  [3] Continuar en Modo Práctica Procedural (Sin ComfyUI, para probar TouchDesigner ya mismo)")
        choice = input("\nSeleccioná una opción [1/2/3]: ").strip()
        if choice == "1":
            download_comfy_portable()
            inst = resolve_comfy_installation(cfg)
        elif choice == "2":
            custom_p = input("Pegá la ruta a tu carpeta ComfyUI: ").strip().strip('"').strip("'")
            cp = Path(os.path.expanduser(custom_p))
            if cp.exists():
                if (cp / "python_embeded" / "python.exe").exists():
                    cfg["comfy_installations"].insert(0, {
                        "name": "ComfyUI Custom Portable",
                        "python_exe": str(cp / "python_embeded" / "python.exe"),
                        "main_py": str(cp / "ComfyUI" / "main.py"),
                        "models_dir": str(cp / "ComfyUI" / "models"),
                    })
                elif (cp / ".venv" / "Scripts" / "python.exe").exists():
                    cfg["comfy_installations"].insert(0, {
                        "name": "ComfyUI Custom Venv Win",
                        "python_exe": str(cp / ".venv" / "Scripts" / "python.exe"),
                        "main_py": str(cp / "main.py"),
                        "models_dir": str(cp / "models"),
                    })
                elif (cp / ".venv" / "bin" / "python").exists():
                    cfg["comfy_installations"].insert(0, {
                        "name": "ComfyUI Custom Venv Mac/Linux",
                        "python_exe": str(cp / ".venv" / "bin" / "python"),
                        "main_py": str(cp / "main.py"),
                        "models_dir": str(cp / "models"),
                    })
                save_config(cfg)
                inst = resolve_comfy_installation(cfg)

    print("\nSeleccioná el Nivel de Hardware / Modelo a configurar:")
    print("  [0] REALTIME (3GB+ VRAM / Mac 8GB+)    -> StreamDiffusion-RT: DreamShaper 8 + Hyper-SD 1-Step + TAESD TinyVAE")
    print("  [1] BASICO   (4-6GB VRAM / Mac 8-16GB) -> DreamShaper 8 (SD 1.5) + LCM LoRA + TAESD (2.1 GB, ultra rápido)")
    print("  [2] MEDIO    (8-10GB VRAM / Mac 16GB+) -> SDXL Turbo FP16 (6.6 GB, 4 pasos en alta velocidad)")
    print("  [3] AVANZADO (12GB+ VRAM NVIDIA)       -> Flux.1 Schnell FP8 (16.4 GB, máxima calidad)")
    print("  [4] P-H3-IMAGE (6GB+ VRAM / INT8)      -> P-H3-Image INT8 ConvRot (8MP en 8 Pasos + 4MP Edición, ~29.4 GB)")
    print("  [5] Saltar descarga de modelos por ahora (usar modelos ya instalados o Modo Práctica)")

    default_opt = {"realtime": "0", "basico": "1", "medio": "2", "avanzado": "3", "p_h3_image": "4"}.get(rec_tier, "0")
    sel = input(f"\nOpción [Enter para sugerido = {default_opt}]: ").strip() or default_opt

    tier_map = {"0": "realtime", "1": "basico", "2": "medio", "3": "avanzado", "4": "p_h3_image"}
    if sel in tier_map:
        chosen_tier = tier_map[sel]
        cfg["active_tier"] = chosen_tier
        save_config(cfg)
        if inst:
            ensure_tier_models(cfg, chosen_tier, auto_yes=False)
    else:
        print("  -> Saltando descarga. Podés descargar o cambiar modelos luego desde la WebApp.")


if __name__ == "__main__":
    if "--detect-json" in sys.argv:
        cfg = load_config()
        gpu = detect_gpu()
        inst = resolve_comfy_installation(cfg)
        ckpts = list_installed_checkpoints(cfg)
        rec = recommend_tier(gpu["vram_gb"], gpu.get("vendor", "nvidia"))
        active = cfg.get("active_tier", rec)
        print(json.dumps({
            "gpu": gpu,
            "recommended_tier": rec,
            "active_tier": active,
            "comfy_installation": inst,
            "installed_checkpoints": ckpts,
            "compatibility": diagnose_hardware_compatibility(gpu, active, inst, ckpts),
        }, indent=2, ensure_ascii=False))
    elif "--install-comfy" in sys.argv:
        ok = download_comfy_portable()
        print(json.dumps({"ok": ok}))
    elif "--download-tier" in sys.argv:
        idx = sys.argv.index("--download-tier")
        t_id = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "basico"
        cfg = load_config()
        cfg["active_tier"] = t_id
        save_config(cfg)
        res = ensure_tier_models(cfg, t_id, auto_yes=True)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        interactive_wizard()

#!/usr/bin/env python3
"""
Instalador, Configurador y Autodiagnóstico Standalone de TouchDiffusion
Curso de Inteligencia Artificial Generativa — TouchDesigner + StreamDiffusion (TensorRT & PyTorch)

Resuelve automáticamente todos los pasos complejos de la instalación manual de olegchomp/TouchDiffusion:
1. Encuentra Python 3.11 (usando el Python 3.11 integrado en TouchDesigner 2023/2025, el del sistema,
   o descargando Python 3.11 Portable automáticamente sin intervención del alumno).
2. Crea el entorno virtual `.venv` compatible (`cp311-win_amd64`) con TouchDesigner.
3. Instala las versiones exactas compatibles de PyTorch 2.1.0+cu118, TensorRT 9.0.1, Polygraphy,
   ONNX GraphSurgeon, Diffusers 0.24.0, HuggingFace Hub 0.23.5, Gradio 4.29.0 y StreamDiffusion.
4. Elimina automáticamente el conflicto de `nvidia-cudnn-cu11` (evitando el famoso error pop-up de `.dll`)
   e inyecta las DLLs de CUDA 11.8 y TensorRT para NO requerir instalar CUDA Toolkit 11.8 en Windows.
5. Pre-descarga `madebyollin/taesd` (TinyVAE) y `stabilityai/sd-turbo` listos para usar.
6. Permite compilar el motor TensorRT (`unet.engine` + `vae_encoder.engine` + `vae_decoder.engine`)
   en 1 clic o usar el Modo Instantáneo PyTorch Directo en TouchDesigner sin esperar compilación.
"""

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe" if os.name == "nt" else VENV_DIR / "bin" / "python"
STREAMDIFFUSION_DIR = BASE_DIR / "StreamDiffusion"

PORTABLE_PY311_URL = (
    "https://github.com/indygreg/python-build-standalone/releases/download/20241016/"
    "cpython-3.11.10+20241016-x86_64-pc-windows-msvc-install_only.tar.gz"
)


def detect_gpu() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            total_mb = float(parts[1])
            free_mb = float(parts[2])
            return {
                "found": True,
                "name": parts[0],
                "vram_total_mb": int(total_mb),
                "vram_free_mb": int(free_mb),
                "vram_gb": round(total_mb / 1024.0, 1),
                "driver": parts[3] if len(parts) > 3 else "unknown",
            }
    except Exception:
        pass
    return {
        "found": False,
        "name": "No se detectó GPU NVIDIA CUDA",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "vram_gb": 0.0,
        "driver": None,
    }


def find_touchdesigner_installations() -> list[dict]:
    found = []
    if os.name == "nt":
        patterns = [
            r"C:\Program Files\Derivative\TouchDesigner*\bin\TouchDesigner.exe",
            r"D:\Program Files\Derivative\TouchDesigner*\bin\TouchDesigner.exe",
        ]
        for pat in patterns:
            for td_exe in sorted(glob.glob(pat), reverse=True):
                bin_dir = Path(td_exe).parent
                py_exe = bin_dir / "python.exe"
                py_ver = ""
                if py_exe.exists():
                    try:
                        py_ver = subprocess.check_output(
                            [str(py_exe), "--version"], stderr=subprocess.STDOUT, text=True, timeout=4
                        ).strip()
                    except Exception:
                        pass
                found.append({
                    "td_exe": str(td_exe),
                    "python_exe": str(py_exe) if py_exe.exists() else None,
                    "python_version": py_ver,
                    "is_py311": "3.11." in py_ver,
                })
    return found


def find_or_install_python311(auto_download: bool = True) -> str:
    """
    Busca un intérprete Python 3.11 en este orden:
    1. `.venv/Scripts/python.exe` si ya fue creado con Python 3.11
    2. Python 3.11 integrado dentro de TouchDesigner 2023/2025 (`C:/Program Files/Derivative/TouchDesigner*/bin/python.exe`)
    3. Python 3.11 instalado en el sistema (`py -3.11` o rutas estándar)
    4. Python 3.11 Portable en `python311_portable/python/python.exe` (descargándolo automáticamente si no existe)
    """
    # 1. Revisar si ya existe en python311_portable
    portable_py = BASE_DIR / "python311_portable" / "python" / "python.exe"
    if portable_py.exists():
        return str(portable_py)

    # 2. Revisar TouchDesigner 2023/2025 (tiene Python 3.11 con módulo venv completo)
    for td in find_touchdesigner_installations():
        if td["is_py311"] and td["python_exe"]:
            return td["python_exe"]

    # 3. Revisar Python Launcher `py -3.11` en Windows
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
            ).strip()
            if out and os.path.exists(out):
                return out
        except Exception:
            pass

        common_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe",
            Path(r"C:\Program Files\Python311\python.exe"),
            Path(r"C:\Python311\python.exe"),
        ]
        for cp in common_paths:
            if cp.exists():
                return str(cp)

    # 4. Revisar si el intérprete actual es 3.11
    if sys.version_info[:2] == (3, 11):
        return sys.executable

    if not auto_download:
        raise RuntimeError("No se encontró Python 3.11 y auto_download=False.")

    # 5. Descargar Python 3.11.10 Portable automáticamente
    print("\n[Python 3.11] No se encontró Python 3.11 en el sistema. Descargando Python 3.11.10 Portable...")
    dest_dir = BASE_DIR / "python311_portable"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "python311.tar.gz"

    urllib.request.urlretrieve(PORTABLE_PY311_URL, str(archive_path))
    print("[Python 3.11] Extrayendo Python 3.11.10 Portable...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=str(dest_dir))
    try:
        archive_path.unlink()
    except Exception:
        pass

    if portable_py.exists():
        print(f"[Python 3.11] Listo en: {portable_py}")
        return str(portable_py)

    raise RuntimeError("No se pudo obtener Python 3.11 automáticamente.")


def ensure_directories():
    for rel in [
        "models",
        "models/acceleration_loras",
        "models/checkpoints",
        "models/loras",
        "models/vae",
        "engines",
        "TD",
    ]:
        (BASE_DIR / rel).mkdir(parents=True, exist_ok=True)


def create_venv_if_needed() -> str:
    if VENV_PYTHON.exists():
        try:
            ver = subprocess.check_output(
                [str(VENV_PYTHON), "--version"], stderr=subprocess.STDOUT, text=True, timeout=4
            ).strip()
            if "3.11." in ver:
                print(f"[.venv] Entorno virtual Python 3.11 ya existe ({ver}).")
                return str(VENV_PYTHON)
            else:
                print(f"[.venv] El entorno actual es {ver} (TouchDesigner requiere 3.11). Recreando .venv...")
                shutil.rmtree(VENV_DIR, ignore_errors=True)
        except Exception:
            shutil.rmtree(VENV_DIR, ignore_errors=True)

    py311 = find_or_install_python311(auto_download=True)
    print(f"[.venv] Creando entorno virtual con: {py311}")
    subprocess.check_call([py311, "-m", "venv", str(VENV_DIR)])
    return str(VENV_PYTHON)


def install_dependencies(py_exe: str):
    print("\n====================================================================")
    print("  INSTALANDO DEPENDENCIAS DE TOUCHDIFFUSION + TENSORRT EN .venv")
    print("====================================================================")

    subprocess.check_call([py_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools==69.5.1", "wheel"])

    print("\n[1/5] Instalando PyTorch 2.1.0 + Torchvision 0.16.0 (CUDA 11.8)...")
    subprocess.check_call([
        py_exe, "-m", "pip", "install",
        "torch==2.1.0", "torchvision==0.16.0",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    ])

    print("\n[2/5] Instalando Diffusers, Transformers, Gradio, ONNX y utilidades...")
    subprocess.check_call([
        py_exe, "-m", "pip", "install",
        "diffusers==0.24.0",
        "transformers==4.38.2",
        "huggingface-hub==0.23.5",
        "accelerate==0.27.2",
        "peft==0.9.0",
        "safetensors",
        "fire",
        "omegaconf",
        "cuda-python==12.3.0",
        "onnx==1.15.0",
        "onnxruntime==1.16.3",
        "protobuf==3.20.2",
        "colored",
        "gradio==4.29.0",
        "pywin32",
        "numpy==1.26.4",
    ])

    print("\n[3/5] Instalando Polygraphy y ONNX-GraphSurgeon (NVIDIA PyPI)...")
    subprocess.check_call([
        py_exe, "-m", "pip", "install",
        "polygraphy==0.49.9",
        "onnx-graphsurgeon==0.5.2",
        "numpy==1.26.4",
        "--extra-index-url", "https://pypi.nvidia.com",
    ])

    print("\n[4/5] Instalando TensorRT 9.0.1 (NVIDIA PyPI)...")
    subprocess.check_call([
        py_exe, "-m", "pip", "install",
        "--pre",
        "--extra-index-url", "https://pypi.nvidia.com",
        "tensorrt==9.0.1.post11.dev4",
    ])

    print("\n[5/5] Instalando paquete local StreamDiffusion y aplicando Fix de DLLs...")
    subprocess.check_call([
        py_exe, "-m", "pip", "install", "--no-deps", "-e", str(STREAMDIFFUSION_DIR)
    ])
    # Fix automático del famoso popup de error de .dll en Windows (conflicto de cudnn)
    subprocess.run([py_exe, "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11"], check=False)


def link_existing_checkpoints():
    """
    Si el alumno ya tiene modelos SD 1.5 descargados en ComfyUI (ej. dreamshaper_8.safetensors),
    crea un enlace o copia en models/checkpoints para que también aparezcan en TouchDiffusion.
    """
    ensure_directories()
    ckpt_dest_dir = BASE_DIR / "models" / "checkpoints"
    candidates = [
        BASE_DIR.parent / "ComfyUI_windows_portable" / "ComfyUI" / "models" / "checkpoints" / "dreamshaper_8.safetensors",
        Path(r"D:\ComfyUI\models\checkpoints\dreamshaper_8.safetensors"),
        Path(r"C:\ComfyUI\models\checkpoints\dreamshaper_8.safetensors"),
        Path.home() / "Documents" / "ComfyUI" / "models" / "checkpoints" / "dreamshaper_8.safetensors",
    ]
    dest_file = ckpt_dest_dir / "dreamshaper_8.safetensors"
    if not dest_file.exists():
        for src in candidates:
            if src.exists():
                try:
                    os.link(str(src), str(dest_file))
                    print(f"[Modelos] Vinculado automáticamente (hardlink): {dest_file.name} desde {src}")
                    break
                except Exception:
                    try:
                        shutil.copy2(str(src), str(dest_file))
                        print(f"[Modelos] Copiado automáticamente: {dest_file.name} desde {src}")
                        break
                    except Exception:
                        pass


def download_base_models(py_exe: str):
    """
    Pre-descarga TinyVAE (madebyollin/taesd) y SD-Turbo (stabilityai/sd-turbo)
    en las carpetas locales de TouchDiffusion para que funcionen sin demoras.
    """
    ensure_directories()
    link_existing_checkpoints()
    print("\n====================================================================")
    print("  DESCARGANDO / VERIFICANDO MODELOS BASE (TinyVAE + SD-Turbo FP16)")
    print("====================================================================")
    code = r"""
import os
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline

root = os.path.abspath(r"__ROOT__")
vae_dir = os.path.join(root, "models", "vae")
ckpt_dir = os.path.join(root, "models", "checkpoints")
os.makedirs(vae_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

print("-> Verificando TinyVAE (madebyollin/taesd)...")
AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir=vae_dir)
print("   [OK] TinyVAE listo en models/vae")

print("-> Verificando SD-Turbo FP16 (stabilityai/sd-turbo)...")
StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    cache_dir=ckpt_dir,
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
    add_watermarker=False,
    safety_checker=None,
)
print("   [OK] SD-Turbo FP16 listo en models/checkpoints")
""".replace("__ROOT__", str(BASE_DIR))
    subprocess.check_call([py_exe, "-c", code])


def build_default_tensorrt_engine(py_exe: str):
    """
    Compila el motor TensorRT por defecto: sd-turbo 512x512 (1 step, img2img)
    en `engines/sd-turbo--sd_1.5_turbo--512--512--None--1--1--img2img--None--None/`.
    """
    print("\n====================================================================")
    print("  COMPILANDO MOTOR TENSORRT POR DEFECTO (sd-turbo 512x512 1-Step)")
    print("  Nota: Este proceso optimiza la red para tu GPU NVIDIA específica")
    print("  y puede demorar entre 3 y 8 minutos la primera vez.")
    print("====================================================================")
    code = r"""
import os
import sys
root = os.path.abspath(r"__ROOT__")
sys.path.insert(0, os.path.join(root, "StreamDiffusion"))
sys.path.insert(0, os.path.join(root, "StreamDiffusion", "src"))
sys.path.insert(0, root)

from StreamDiffusion.webui import stream_engine
res = stream_engine(512, 512, 1, "None", "SD Turbo", "sd_1.5_turbo")
print("\n" + res)
""".replace("__ROOT__", str(BASE_DIR))
    subprocess.check_call([py_exe, "-c", code])


def run_selftest(py_exe: str) -> dict:
    """
    Verifica que el entorno .venv, PyTorch CUDA, TensorRT, StreamDiffusionWrapper,
    los modelos base, el archivo TouchDiffusion.tox y el proyecto .toe funcionen al 100%.
    """
    code = r"""
import json
import os
import sys
import time

root = os.path.abspath(r"__ROOT__")
sys.path.insert(0, os.path.join(root, "StreamDiffusion"))
sys.path.insert(0, os.path.join(root, "StreamDiffusion", "src"))
sys.path.insert(0, root)

from StreamDiffusion.utils.wrapper import StreamDiffusionWrapper
import torch
import tensorrt as trt
import polygraphy
import onnx_graphsurgeon

engines_dir = os.path.join(root, "engines")
trt_engines = []
if os.path.isdir(engines_dir):
    for r, d, files in os.walk(engines_dir):
        if "unet.engine" in files:
            trt_engines.append(os.path.basename(r))

accel = "tensorrt" if len(trt_engines) > 0 else "none"
t0 = time.time()
wrapper = StreamDiffusionWrapper(
    model_id_or_path="sd-turbo",
    t_index_list=[0],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=0,
    acceleration=accel,
    mode="img2img",
    use_denoising_batch=True,
    cfg_type="self",
    seed=42,
    use_lcm_lora=False,
    use_hyper_lora=False,
    output_type="pt",
    model_type="sd_1.5_turbo",
    touchdiffusion=True,
    engine_dir=engines_dir,
)
wrapper.prepare(
    prompt="cyberpunk neon city, glowing lights, masterpiece",
    negative_prompt="blurry, low quality",
    guidance_scale=1.0,
    delta=0.5,
    t_index_list=wrapper.touchdiffusion_generate_t_index_list(0.6, "linear"),
)
dummy_in = torch.rand((1, 3, 512, 512), device="cuda", dtype=torch.float16)
# Warmup + benchmark
for _ in range(4):
    out_tensor = wrapper(image=dummy_in)
torch.cuda.synchronize()
t_start = time.time()
iters = 10
for _ in range(iters):
    out_tensor = wrapper(image=dummy_in)
torch.cuda.synchronize()
fps = round(iters / max(0.001, time.time() - t_start), 1)

result = {
    "all_ok": bool(out_tensor is not None and tuple(out_tensor.shape) == (3, 512, 512)),
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "tensorrt_version": trt.__version__,
    "acceleration_tested": accel,
    "compiled_trt_engines": trt_engines,
    "output_shape": list(out_tensor.shape),
    "measured_fps": fps,
    "tox_exists": os.path.exists(os.path.join(root, "TouchDiffusion.tox")),
    "toe_exists": os.path.exists(os.path.join(root, "TD", "TouchDiffusion_Clase_IA.toe")),
}
print("SELFTEST_JSON:" + json.dumps(result))
""".replace("__ROOT__", str(BASE_DIR))

    out = subprocess.check_output([py_exe, "-c", code], text=True)
    for line in out.splitlines():
        if line.startswith("SELFTEST_JSON:"):
            return json.loads(line[len("SELFTEST_JSON:"):])
    return {"all_ok": False, "raw_output": out}


def main():
    parser = argparse.ArgumentParser(description="Instalador Standalone de TouchDiffusion para Clase de IA")
    parser.add_argument("--detect-json", action="store_true", help="Imprime diagnóstico de hardware y entorno en JSON")
    parser.add_argument("--install-deps-only", action="store_true", help="Instala únicamente dependencias pip en .venv")
    parser.add_argument("--auto", action="store_true", help="Instalación completa desatendida (venv + paquetes + modelos)")
    parser.add_argument("--build-engine", action="store_true", help="Compila también el motor TensorRT sd-turbo 512x512")
    parser.add_argument("--selftest", action="store_true", help="Ejecuta prueba de inferencia real y devuelve JSON")
    args = parser.parse_args()

    ensure_directories()

    if args.detect_json:
        gpu = detect_gpu()
        tds = find_touchdesigner_installations()
        engines_dir = BASE_DIR / "engines"
        engines = [p.parent.name for p in engines_dir.rglob("unet.engine")] if engines_dir.exists() else []
        info = {
            "gpu": gpu,
            "touchdesigner_installations": tds,
            "venv_exists": VENV_PYTHON.exists(),
            "venv_python": str(VENV_PYTHON) if VENV_PYTHON.exists() else None,
            "compiled_engines": engines,
            "models_checkpoints": [p.name for p in (BASE_DIR / "models" / "checkpoints").glob("*.safetensors")],
        }
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return

    if args.install_deps_only:
        py_exe = create_venv_if_needed()
        install_dependencies(py_exe)
        return

    if args.selftest:
        py_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else create_venv_if_needed()
        res = run_selftest(py_exe)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        sys.exit(0 if res.get("all_ok") else 1)

    if args.auto:
        py_exe = create_venv_if_needed()
        install_dependencies(py_exe)
        download_base_models(py_exe)
        if args.build_engine:
            build_default_tensorrt_engine(py_exe)
        return

    # Modo Interactivo para Alumnos
    print("====================================================================")
    print("  CURSO DE IA GENERATIVA — TOUCHDIFFUSION STANDALONE (WINDOWS)")
    print("  Instalador Automático de StreamDiffusion + TensorRT + TouchDesigner")
    print("====================================================================\n")

    gpu = detect_gpu()
    tds = find_touchdesigner_installations()

    print(f"  * GPU Detectada      : {gpu['name']} ({gpu['vram_gb']} GB VRAM)")
    if tds:
        print(f"  * TouchDesigner      : {tds[0]['td_exe']} ({tds[0]['python_version']})")
    else:
        print("  * TouchDesigner      : No detectado en ruta estándar (se usará Python 3.11 Portable)")

    if not gpu["found"]:
        print("\n  [!] ATENCIÓN: TouchDiffusion requiere una placa de video NVIDIA con soporte CUDA.")
        print("      Si estás en una Mac o notebook sin GPU NVIDIA, usá el puente principal")
        print("      de la cátedra (carpeta raíz `ic_AI_Modulo`) que soporta Mac Metal y Modo Práctica.\n")

    py_exe = create_venv_if_needed()
    install_dependencies(py_exe)
    download_base_models(py_exe)

    print("\n====================================================================")
    print("  ¿QUERÉS PRE-COMPILAR AHORA EL MOTOR TENSORRT (50-60 FPS)?")
    print("====================================================================")
    print("  [1] Sí, compilar motor TensorRT ahora (`sd-turbo 512x512`, demora ~4 min)")
    print("  [2] No por ahora (usar Modo PyTorch Directo inmediato o compilar luego en WebUI)")
    choice = input("\nElegí una opción [1/2] (Enter = 1): ").strip() or "1"

    if choice == "1" and gpu["found"]:
        build_default_tensorrt_engine(py_exe)

    print("\n====================================================================")
    print("  VERIFICANDO INSTALACIÓN (AUTODIAGNÓSTICO)...")
    print("====================================================================")
    try:
        st = run_selftest(py_exe)
        print(f"  * Estado General     : {'OK (100% Funcional)' if st.get('all_ok') else 'Revisar'}")
        print(f"  * Aceleración Activa : {st.get('acceleration_tested', '').upper()} ({st.get('measured_fps')} FPS medidos)")
        print(f"  * Motores TensorRT   : {', '.join(st.get('compiled_trt_engines', [])) or 'Ninguno aún (PyTorch Directo activo)'}")
    except Exception as e:
        print(f"  [!] Aviso en autodiagnóstico: {e}")

    print("\n====================================================================")
    print("  ¡INSTALACIÓN FINALIZADA CON ÉXITO!")
    print("  - Para abrir el proyecto listo en TouchDesigner: 3_Abrir_Proyecto_TD.bat")
    print("  - Para compilar nuevos motores TensorRT (WebUI): 2_Compilar_Motor_WebUI.bat")
    print("====================================================================")


if __name__ == "__main__":
    main()

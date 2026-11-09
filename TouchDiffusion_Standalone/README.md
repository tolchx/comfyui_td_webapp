# ⚡ TouchDiffusion Standalone — Edición Clase de Inteligencia Artificial (Windows)

Versión **100% Standalone, pre-configurada y de instalación en 1 clic** del proyecto [**TouchDiffusion (`olegchomp/TouchDiffusion`)**](https://github.com/olegchomp/TouchDiffusion) + [**StreamDiffusion (`olegchomp/StreamDiffusion`)**](https://github.com/olegchomp/StreamDiffusion), preparada especialmente para los alumnos de la cátedra de Inteligencia Artificial Generativa con **TouchDesigner**.

---

## 🛠️ ¿Qué resuelve esta versión Standalone frente al repositorio original?

Instalar el repositorio original de `olegchomp/TouchDiffusion` manualmente solía requerir más de 10 pasos propensos a errores en clase:
1. ❌ **Antes**: Había que instalar Python 3.11 a mano, abrir `webui.bat` con el Bloc de Notas y editar `set PYTHON_PATH=`.
   ✅ **Ahora**: El instalador detecta automáticamente el **Python 3.11 integrado dentro de TouchDesigner 2023/2025** (`C:\Program Files\Derivative\TouchDesigner*\bin\python.exe`), el Python 3.11 del sistema, o **descarga Python 3.11.10 Portable automáticamente** sin tocar el sistema.
2. ❌ **Antes**: Fallaba con `UserWarning: Failed to initialize NumPy: _ARRAY_API not found ... RuntimeError: Numpy is not available` en TouchDesigner 2024 / 2025.
   ✅ **Ahora**: Se actualizó a **`torch==2.4.1+cu118`** y **`torchvision==0.19.1+cu118`**, garantizando compatibilidad nativa con el **NumPy 2.1.2** bundled en TouchDesigner 2025.
3. ❌ **Antes**: Había que descargar e instalar **NVIDIA CUDA Toolkit 11.8** (más de 3 GB) y reiniciar la PC.
   ✅ **Ahora**: Los scripts y el componente `.tox` inyectan automáticamente las librerías dinámicas (`torch\lib`, `tensorrt_libs` y `nvidia\cudnn\bin`) en memoria mediante `os.add_dll_directory()`. **No hace falta instalar CUDA Toolkit 11.8 en Windows ni reiniciar la PC**.
4. ❌ **Antes**: El servidor `pypi.ngc.nvidia.com` usado por el script original dejó de existir y versiones nuevas de `huggingface_hub`, `diffusers`, `onnx-graphsurgeon` y `numpy 2.x` rompían la instalación. Además aparecía un cartel de error de `.dll` (`nvidia-cudnn-cu11`) que obligaba a tocar el botón *"Fix pop up"*.
   ✅ **Ahora**: Todas las dependencias están fijadas y verificadas (`torch==2.4.1+cu118`, `tensorrt==9.0.1.post11.dev4`, `nvidia-cudnn-cu11==8.9.4.25`, `polygraphy==0.49.9`, `onnx-graphsurgeon==0.5.2`, `diffusers==0.24.0`, `huggingface-hub==0.23.5`) y `TouchDiffusionExt.py` filtra automáticamente librerías conflictivas de otras versiones de Python (ej. Python 3.13) en `sys.path`.
5. ❌ **Antes**: En TouchDesigner había que arrastrar `TouchDiffusion.tox`, configurar a mano la ruta de la carpeta en `Settings`, guardar el proyecto `.toe`, cerrar TouchDesigner y volverlo a abrir, y era obligatorio esperar ~10 minutos compilando un motor TensorRT antes de poder generar una sola imagen.
   ✅ **Ahora**:
   - Incluye el proyecto listo **`TD/TouchDiffusion_Clase_IA.toe`** que detecta la carpeta `.venv` y carga las librerías **en caliente sin reiniciar TouchDesigner**.
   - Incluye tanto el **Modo TensorRT Nivo (35–60 FPS)** como un **Modo `[PyTorch-Direct] sd-turbo-512-512-1 (Sin compilar TensorRT)`** que funciona en el acto apenas termina la instalación.

---

## 🚀 Instalación y Uso para Alumnos (En 2 clics)

### Paso 1: Instalar el Entorno y Modelos Base
Hacé doble clic en **`1_Instalar_TouchDiffusion.bat`**.
- Creará el entorno `.venv` con Python 3.11 compatible con TouchDesigner.
- Instalará PyTorch CUDA 11.8, TensorRT 9, Polygraphy, Diffusers y StreamDiffusion.
- Descargará automáticamente **TinyVAE (`madebyollin/taesd`)** y **SD-Turbo FP16 (`stabilityai/sd-turbo`)** en la carpeta `models/`.
- Si ya tenés `dreamshaper_8.safetensors` descargado del módulo principal de ComfyUI, lo vinculará automáticamente en `models/checkpoints/` sin volver a descargarlo.
- Te preguntará si querés **pre-compilar el motor TensorRT (`[1]`)** en ese momento o pasar directo a TouchDesigner (`[2]`).

### Paso 2: Abrir el Proyecto en TouchDesigner
Hacé doble clic en **`3_Abrir_Proyecto_TD.bat`** (o abrí **`TD/TouchDiffusion_Clase_IA.toe`**).
1. Seleccioná el nodo **`/project1/TouchDiffusion`**.
2. En la pestaña **`Settings`** de los parámetros:
   - La ruta **`TouchDiffusion folder`** ya se autoconfigura sola apuntando a esta carpeta.
   - En **`Select engine`**, elegí tu motor TensorRT compilado (`[TensorRT] sd-turbo-512-512-1`) o elegí **`[PyTorch-Direct] sd-turbo-512-512-1 (Sin compilar TensorRT)`** para usar sin espera.
   - Hacé clic en **`Load engine`**.
3. En la pestaña **`Stream`**:
   - Escribí tu **`Prompt`** en inglés (ej. `cyberpunk neon city, synthwave art, glowing reflections`) y ajustá **`Denoising strength`** (`0.50` a `0.85`).
   - El nodo recibe en vivo la animación conectada a su entrada (`entrada_img2img`) y devuelve el video generado en tiempo real en **`SALIDA_COMFYUI`** y **`comparador_img2img`**.

---

## ⚙️ ¿Cómo compilar nuevos motores TensorRT personalizados (WebUI)?

Si querés compilar un motor TensorRT para otro modelo `.safetensors` (ubicado en `models/checkpoints/`), otra resolución o distinta cantidad de pasos (`Batch size`):

1. Hacé doble clic en **`2_Compilar_Motor_WebUI.bat`** (o `webui.bat`).
2. Se abrirá automáticamente `http://127.0.0.1:7860` en tu navegador.
3. Elegí:
   - **Select model type**: `sd_1.5_turbo` (para `SD Turbo`) o `sd_1.5` (para cualquier modelo `.safetensors` en `models/checkpoints`, como `dreamshaper_8.safetensors`).
   - **Width / Height**: `512 x 512`.
   - **Sampling steps (Batch size)**: `1` o `2`.
   - **Add Acceleration Lora**: `None` (para SD Turbo) o `LCM` / `HyperSD` (para modelos SD 1.5 estándar).
4. Hacé clic en **`Make engine (Compilar TensorRT)`** y esperá entre 3 y 8 minutos hasta que diga `Status: Ready`.
5. En TouchDesigner, hacé clic en **`Refresh engines`**, seleccioná el nuevo motor `[TensorRT]` y pulsá **`Load engine`**.

---

## 🤖 Prompt de Auto-Instalación para Agentes LLM (Claude Code, Antigravity, Cursor)

Si usás un agente de IA con terminal, copiá y pegale este bloque para que instale y verifique todo automáticamente:

```markdown
Necesito que instales, configures y verifiques el proyecto TouchDiffusion Standalone dentro de la carpeta `TouchDiffusion_Standalone`. Seguí estos pasos hasta que el autodiagnóstico devuelva `"all_ok": true`:

1. Ejecutá `python TouchDiffusion_Standalone/setup_standalone.py --detect-json` para verificar mi GPU NVIDIA y la instalación de TouchDesigner.
2. Ejecutá `python TouchDiffusion_Standalone/setup_standalone.py --auto` para crear el `.venv` con Python 3.11, instalar PyTorch 2.1.0+cu118, TensorRT 9.0.1, Polygraphy, ONNX-GraphSurgeon y pre-descargar TinyVAE (`madebyollin/taesd`) y `stabilityai/sd-turbo`.
3. Ejecutá `python TouchDiffusion_Standalone/setup_standalone.py --selftest` para correr una prueba real de inferencia en GPU y verificar que devuelva `"all_ok": true` junto con los FPS medidos.
```

---

## 📁 Estructura de la Carpeta `TouchDiffusion_Standalone`

```text
TouchDiffusion_Standalone/
├── 1_Instalar_TouchDiffusion.bat   # Instalador en 1 clic (Python 3.11 + PyTorch + TensorRT + Modelos)
├── 2_Compilar_Motor_WebUI.bat      # Abre la WebUI en http://127.0.0.1:7860 para compilar motores .engine
├── 3_Abrir_Proyecto_TD.bat         # Abre TD/TouchDiffusion_Clase_IA.toe con rutas y DLLs listas
├── webui.bat                       # Alias compatible con el tutorial original de Oleg Chomp
├── setup_standalone.py             # Instalador inteligente, descargador HF y suite --selftest
├── TouchDiffusion.tox              # Componente TouchDesigner mejorado (auto-ruta + DLLs + PyTorch Direct)
├── TouchDiffusionExt.py            # Extensión Python de TouchDiffusion sincronizada con el .tox
├── TD/
│   └── TouchDiffusion_Clase_IA.toe # Proyecto TouchDesigner listo para abrir y usar en clase
├── StreamDiffusion/                # Fork de StreamDiffusion con fixes de TensorRT, HyperSD y rutas
├── models/
│   ├── checkpoints/                # Modelos SD 1.5 (.safetensors) y cache de stabilityai/sd-turbo
│   ├── acceleration_loras/         # Cache de LCM-LoRA y ByteDance Hyper-SD LoRA
│   └── vae/                        # Cache de TinyVAE (madebyollin/taesd)
└── engines/                        # Motores TensorRT compilados (unet.engine, vae_encoder, vae_decoder)
```

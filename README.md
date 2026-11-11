# 📘 Suite Modular TouchDesigner ↔ ComfyUI (Windows & macOS)
## 7 Módulos Autónomos: Imágenes HQ + Realtime + 3D + Video + Alfa & Inpaint + Parallax 2.5D + Poster Studio
### 📖 Incluye Sistema de Guías y Tutoriales In-App en los 7 Módulos (`/tutorial.html`)

Arquitectura **modular, standalone y multiplataforma (Windows & macOS)** diseñada para cátedras y talleres de Inteligencia Artificial Generativa. Separa el sistema en **siete módulos independientes y conectados** con **ComfyUI (`127.0.0.1:8188`)** y **TouchDesigner (`TD/v1.toe` & `TD/ComfyUI_StreamDiffusion_Studio.toe`)**:

1. **🎨 Módulo 1 — Generador de Imágenes de Alta Calidad (`bridge/` + `webapp/` • Puerto `8080`)**:
   - Dedicado a **Text-to-Image (`txt2img`)** e **Image-to-Image (`img2img`)** de máxima fidelidad visual.
   - **Soporte Nativo MiniMax H3 Image & P-H3-Image**: Integración nativa oficial de ComfyUI ([blog.comfy.org](https://blog.comfy.org/p/minimax-h3-image-is-now-available)). Checkpoint unificado para **Text-to-Image** y **edición multi-referencia** (hasta 16 imágenes con sintaxis `<Picture 1>`) a resoluciones de **4MP hasta 16–20 Megapixels en solo 6 pasos** (CFG=1.0) y compatible con GPUs desde 6 GB de VRAM con `VAEDecodeTiled`.
   - **Auto-detección de Hardware Potente**: Al detectar GPUs de gama alta (ej. **NVIDIA GeForce RTX 4090 24 GB VRAM**), activa automáticamente el nivel **`AVANZADO`** con **Flux.2 Klein 4B / Flux.2 Dev / Qwen Image 2.1** a `1024×1024` y decodificador **Full VAE** (`flux2-vae.safetensors`).
   - **Galería + Lightbox Pantalla Completa (`⛶`) + Envío Directo a 3D (`🧊 a 3D`) y Video (`🎬 a Video`)**.
   - **📖 Guía Integrada In-App**: Botón permanente en la barra superior y chip contextual para consultar parámetros, resoluciones y conexión TouchDesigner sin salir de la página.

2. **🎬 Módulo 2 — StreamDiffusion Realtime Studio + Pizarra Pro (`streamdiffusion_studio/` • Puerto `8081`)**:
   - Módulo en [`streamdiffusion_studio/`](streamdiffusion_studio/) para generación en tiempo real con latencia out-of-process sub-segundo:
     - **[ComfyUI_StreamDiffusion (`jesenzhang/ComfyUI_StreamDiffusion`)](https://github.com/jesenzhang/ComfyUI_StreamDiffusion)**: Nodos nativos `StreamDiffusion_Loader` y `StreamDiffusion_Sampler` (700 ms – 1.2 s en SD1.5/DreamShaper 8 en RTX 4090 con cero cuelgues de GPU).
     - **🎨 Pizarra Pro Multicapa**: Herramientas profesionales de pluma, pincel suave, marcador, spray, borrador, bote de pintura, gotero, selector de tamaño/opacidad, capas reordenables, plantillas de cielo/paisaje y sol, y herramienta mover capa (`M`).
     - **🪟 Captura Directa de Ventana / Aplicación (WebRTC)**: Captura con 1 clic la ventana de **TouchDesigner**, Photoshop o navegador.
     - **📖 Botón de Guía M2**: Acceso directo al tutorial interactivo sobre latencia, parámetros y configuración TouchDesigner.

3. **🧊 Módulo 3 — TRELLIS v2 + Hunyuan 3D Local Studio (`trellis3d_studio/` • Puerto `8082`)**:
   - Convierte cualquier imagen en un **objeto 3D local (`.GLB` PBR y `.OBJ`)**:
     - **Nivel Avanzado**: TRELLIS v2 Local (Microsoft / Comfy-Org 4B Sparse Voxel + PBR Texture Stage).
     - **Nivel Medio**: Hunyuan 3D v2.1 Local DiT.
     - **Nivel Universal Instantáneo**: Escultor PBR 3D Instantáneo (<1.5s sin requerir VRAM extra).
   - Incluye **Visor 3D Orbital WebGL integrado** y **botón de Guía M3** con explicaciones de formatos PBR y `fileinSOP`.

4. **🎬 Módulo 4 — AI Video Studio T2V & I2V (`video_studio/` • Puerto `8083`)**:
   - Generación de animación temporal e Image-to-Video conectada a TouchDesigner:
     - **Motor Wan 2.2 / 2.1 14B High Noise + 4-Step Lightning LoRA**: Inferencia cinemática HD en solo 4 pasos.
     - **Motor Wan 2.2 14B Q4_K_M GGUF**: Optimizado para GPUs de 8 a 12 GB VRAM.
     - **Camera Motion Synth 3D & Parallax**: Animación instantánea (<2s) a 60 FPS con zoom, pan, órbita 3D y vórtice (0 VRAM extra).
   - Sincroniza automáticamente `outputs/latest_video.mp4` para reproducción continua en loop con `moviefileinTOP` en TouchDesigner.
   - **📖 Botón de Guía M4**: Documentación sobre Wan 2.2, prompts de cámara y sincronización de frames.

5. **✂️ Módulo 5 — Segmentación Alfa & Inpainting Studio (`segmentation_studio/` • Puerto `8084`)**:
   - Aislamiento de sujetos y reconstrucción de fondos:
     - **Modo 1-Click Auto**: Detección y extracción de silueta principal.
     - **Modo Puntos Interactivos (SAM 2)**: Clic izquierdo (verde +) para incluir y clic derecho (rojo -) para excluir.
     - **Desoclusión e Inpainting de Fondo**: Rellena el fondo detrás del sujeto para colocar textos o partículas intermedias.
   - Entrega `latest_subject_alpha.png` (RGBA transparente) y `latest_background_inpainted.png` listos para `overTOP` en TD.
   - **📖 Botón de Guía M5**: Explicación detallada de segmentación SAM 2 y composición en capas en TouchDesigner.

6. **🌌 Módulo 6 — 2.5D Parallax & Depth Studio (`parallax25d_studio/` • Puerto `8085`)**:
   - Convierte imágenes 2D en escenarios tridimensionales interactivos:
     - **Estimación de Profundidad**: Depth maps suaves y continuos de 8/16-bit.
     - **Visor 3D WebGL con Parallax**: Mové el mouse en el navegador para ver la perspectiva en tiempo real.
     - **Malla 3D Desplazada (`.OBJ` y `.GLB`)**: Cuadrícula deformable con UVs lista para `fileinSOP` o `displaceTOP` a 60 FPS en TouchDesigner.
   - **📖 Botón de Guía M6**: Guía de integración de mapas de profundidad y shaders de desplazamiento.

7. **🖼️ Módulo 7 — Poster Studio · MOSS-Image & MiniMax H3 (`poster_studio/` • Puerto `8086`)**:
   - Pósters con **texto legible y ubicación exacta**: escribís los textos, los arrastrás en un canvas y el sistema genera la pieza.
   - **Modo Híbrido (default)**: MOSS-Image genera el arte y la tipografía se compone con **fuentes reales del sistema** (Pillow). Texto exacto, editable sin regenerar, control de cuerpo, tracking, interlínea y alineación.
   - **Modo IA pura**: el modelo dibuja también el texto usando **layout JSON por cajas** (`MOSSImageLayoutPrompt`). Efecto demo, sin control de fuente: revisar letra por letra.
   - **Generador de Arte Dual**: Soporta tanto **MOSS-Image** (instancia 8189) como el nuevo **MiniMax H3 Image Nativo** (`profile="minimax_h3"` en solo 6 pasos) compatible directamente con el ComfyUI estándar de la suite (8188).
   - Entrega `outputs/latest_poster.png` (y `latest.png`) para levantarlo en TD con `moviefileinTOP`.
   - **📖 Botón de Guía M7**: Manual de tipografía, layouts JSON, cajas y exportación.

Repositorio oficial: **[https://github.com/tolchx/comfyui_td_webapp](https://github.com/tolchx/comfyui_td_webapp)**

---

## ✨ Características Principales

1. **⚡ Modo StreamDiffusion Realtime Integrado (1-Step + TAESD TinyVAE + 60 FPS GPU Blend)**:
   - Inspirado en [StreamDiffusion (`cumulo-autumn/streamdiffusion`)](https://github.com/cumulo-autumn/streamdiffusion) y [TouchDiffusion (`olegchomp/TouchDiffusion`)](https://github.com/olegchomp/TouchDiffusion), pero modernizado para funcionar **sin compilar motores TensorRT** y compatible con **PCs de bajos recursos (desde 3 GB VRAM) y Macs con Apple Silicon**.
   - Combina **Hyper-SD 1-Step LoRA** (`steps=1, cfg=1.0`), **TAESD / TAEF2 Tiny AutoEncoder** (5 MB), **Stochastic Similarity Filter** y un **Interpolador Temporal en GPU a 60 FPS (`feedbackTOP` + `crossTOP`)** dentro del nodo de TouchDesigner.
2. **Detección Automática de Hardware y Diagnóstico Inteligente**:
   - Detecta automáticamente placas **NVIDIA CUDA** (Windows/Linux), chips **Apple Silicon M1/M2/M3/M4 con Metal (MPS)** en macOS, placas **AMD Radeon** o **GPUs integradas / CPU**.
   - Evalúa en vivo si el modelo elegido es compatible con la memoria VRAM disponible y **muestra en la WebApp alertas y soluciones paso a paso** cuando detecta cuellos de botella o falta de memoria.
3. **Instalación por Niveles (Realtime / Básico / Medio / Avanzado)**:
   - Descarga automáticamente el modelo generador óptimo desde HuggingFace con barra de progreso y soporte para reanudar descargas interrumpidas.
4. **Modo Práctica Procedural Integrado (Cero VRAM / Sin ComfyUI)**:
   - Si un alumno está en clase con una notebook sin GPU dedicada o mientras descarga ComfyUI, el servidor puente genera arte procedural reactivo al Prompt y al TOP de entrada (`img2img`) para que pueda practicar todo el flujo en TouchDesigner a **60 FPS** desde el primer minuto.
5. **Sincronización Bidireccional TouchDesigner ↔ WebApp**:
   - Cualquier generación disparada desde la WebApp actualiza automáticamente la textura `SALIDA_COMFYUI` dentro de TouchDesigner.
   - Cualquier generación disparada desde TouchDesigner aparece en vivo en el monitor y en el historial de la WebApp.

---

## ⚡ Arquitectura StreamDiffusion Realtime: ¿Por qué esta implementación es más óptima?

Hace unos años surgieron proyectos pioneros como **[StreamDiffusion](https://github.com/cumulo-autumn/streamdiffusion)** (2023) y **[TouchDiffusion](https://github.com/olegchomp/TouchDiffusion)** (2024). Si bien demostraron la generación en tiempo real, presentaban grandes barreras para alumnos en una cursada:
- Requerían **compilar motores TensorRT (`.engine`)** específicos para cada placa NVIDIA (lo cual demora 15–30 minutos, consume >12 GB de RAM/VRAM al compilar y falla al cambiar la resolución).
- Estaban atados a versiones antiguas de Python/PyTorch/Diffusers y **no funcionaban bien en macOS (Apple Silicon) ni en placas de gama de entrada** sin TensorRT.
- No soportaban modelos modernos como **Hyper-SD**, **SDXL Turbo** o **Flux.2 Klein** dentro de un mismo flujo con ComfyUI.

### Nuestra Solución Moderna (`StreamDiffusion-RT` sobre ComfyUI)
En lugar de depender de un motor TensorRT rígido, implementamos las optimizaciones matemáticas clave de StreamDiffusion directamente sobre el ecosistema moderno de ComfyUI y TouchDesigner:

| Componente | Cómo funcionaba antes (SD Clásico) | Cómo funciona nuestro modo `StreamDiffusion-RT` | Ganancia en Tiempo Real |
| :--- | :--- | :--- | :--- |
| **Pasos de Inferencia (UNET)** | 20 a 30 pasos con `CFG = 7.5` (40–60 pasadas de red por imagen) | **1 solo paso (`steps = 1`)** con **ByteDance Hyper-SD 1-Step LoRA** y **`CFG = 1.0`** (1 única pasada de red sin duplicación CFG) | **20x a 40x más rápido** (corre hasta en 3 GB VRAM) |
| **Codificación / Decodificación Latente (VAE)** | VAE estándar de 335 MB (consume el 50% del tiempo total en 1-step) | **TAESD / TAEF2 (`madebyollin/taesd`)**: Tiny AutoEncoder de **5 MB** para codificar el TOP de TD y decodificar el latente | **10x más rápido** y ahorra ~1.2 GB de VRAM |
| **Semilla Latente Coherente** | Semilla aleatoria en cada frame (provoca parpadeo / flicker severo) | **Latent Seed Locking (`seed = 424242`)** + `denoise = 0.55–0.60` sobre el TOP en movimiento de TouchDesigner | Coherencia temporal fluida entre frames consecutivos |
| **Stochastic Similarity Filter** | Re-ejecuta la GPU aunque la imagen de entrada no haya cambiado | Calcula una firma perceptual `16x16` del frame entrante y omite la inferencia si el cambio es menor al umbral | Ahorra GPU y temperatura cuando la escena está quieta |
| **Interpolación Temporal en TD** | Salto brusco entre cada frame recibido desde Python | **GPU Temporal Feedback Blend (`fb_smooth` + `cross_smooth`)** dentro de `/project1/comfy_studio` | Salida visual continua a **60.0 FPS** constantes en TouchDesigner |

---

## 🖥️ Requisitos y Niveles de Modelos según tu Hardware

| Nivel | Hardware Recomendado (Windows / Mac) | Modelo Instalado Automáticamente | Peso | Flags Automáticos |
| :--- | :--- | :--- | :--- | :--- |
| **0. ⚡ StreamDiffusion RT** | **NVIDIA 3 GB+ VRAM** (GTX 1650, RTX 2060/3050/4060/4090)<br>**Mac Apple Silicon 8 GB+ RAM** (M1/M2/M3/M4) | **DreamShaper 8 + Hyper-SD 1-Step LoRA + TAESD / TAEF2 TinyVAE** | ~2.1 GB | `<6GB`: `--lowvram`<br>`>=14GB`: `--highvram`<br>Mac: `--force-fp16` |
| **1. Básico** | **NVIDIA 4–6 GB VRAM** (GTX 1650, RTX 3050)<br>**Mac Apple Silicon 8–16 GB RAM** (M1/M2/M3) | **DreamShaper 8 (SD 1.5) + LCM LoRA + TAESD** | ~2.1 GB | Windows `<6GB`: `--lowvram`<br>Mac: `--force-fp16` |
| **2. Medio** | **NVIDIA 8–10 GB VRAM** (RTX 3060, 3070, 4060)<br>**Mac Apple Silicon 16–24 GB RAM** (M1 Pro/Max, M2/M3/M4) | **SDXL Turbo FP16** | ~6.6 GB | Windows: Normal VRAM<br>Mac: `--force-fp16` |
| **3. Avanzado** | **NVIDIA 12 GB+ VRAM** (RTX 3080, 4070, 4080, 4090)<br>**Mac Apple Silicon 32 GB+ RAM** | **Flux.1 Schnell FP8** *(+ soporte nativo para Flux.2 Klein 4B / Dev con TAEF2 y Qwen Image 2.1)* | ~16.4 GB | Windows `>=14GB`: `--highvram`<br>Mac: `--force-fp16` |
| **4. 🚀 P-H3-Image** | **NVIDIA 6 GB+ VRAM** (RTX 2060, 3050, 3060, 4060, 4090)<br>**Mac Apple Silicon 16 GB+ RAM** | **P-H3-Image INT8 ConvRot** (8 Megapixels en 8 pasos + 4MP Edición Multi-Ref) | ~29.4 GB | `--lowvram`<br>`VAEDecodeTiled` (512px) |

---

## 🚀 MiniMax-H3-Image Nativo ComfyUI & P-H3-Image: Hasta 16-20 Megapixels en 6 Pasos

Basado en la integración oficial nativa de **ComfyUI** ([ComfyUI Blog Announcement](https://blog.comfy.org/p/minimax-h3-image-is-now-available)), **MiniMax-H3-Image** representa un salto generacional en síntesis y edición fotográfica de ultra-alta resolución:

### 🌟 Capacidades Clave
1. **Checkpoint Unificado (T2I & Image Edit)**: Un solo modelo maneja tanto la generación desde texto como la edición con múltiples imágenes de referencia.
2. **Muestreo Distilled Ultrarrápido en 6 Pasos (`CFG = 1.0`)**: No requiere negative prompt ni sintonización compleja; evalúa un solo forward pass por paso (`euler / simple`), reduciendo el consumo de cómputo y memoria a la mitad.
3. **Resolución Nativa de 4MP a 16–20 Megapixels en Una Pasada**:
   - `1:1`: 4096×4096 (~16.8 MP) y 2048×2048 (4.2 MP)
   - `16:9`: 5472×3072 (~16.8 MP) y 3872×2176 (8.4 MP)
   - `4:3`: 4736×3552 (~16.8 MP)
   - `21:9`: 6272×2688 (~16.8 MP Ultra-Panorámico)
4. **Edición Contextual Multi-Referencia (hasta 16 imágenes)**: Permite conectar referencias visuales y referenciarlas directamente en el prompt como `<Picture 1>`, `<Picture 2>`, logrando transformaciones estilísticas y consistencia de personajes.

### 🔬 ¿Por qué la versión INT8 ConvRot corre en solo 6 GB de VRAM?
El modelo original en `BF16` pesa más de **140 GB** combinados y exige estaciones de trabajo de 80 GB VRAM. La versión **`INT8 ConvRot`** resuelve este problema mediante tres pilares:
1. **Cuantización INT8 con Rotación Hadamard (`ConvRot`)**: Almacena escalas por fila y rota las matrices de pesos para dispersar los valores atípicos (*activation outliers*). El UNET se reduce de 40.5 GB a **21.2 GB** con fidelidad indistinguible de BF16 (superando a FP8 en rango dinámico y velocidad).
2. **Descarga Dinámica de Bloques de ComfyUI (`--lowvram`)**: ComfyUI mantiene en RAM del sistema los bloques no activos y transfiere a la GPU únicamente las capas que se están calculando en cada instante, permitiendo ejecutar el modelo completo en tarjetas con **6 GB de VRAM** (RTX 2060, RTX 3050, RTX 3060, Apple Silicon).
3. **Decodificación por Mosaicos (`VAEDecodeTiled` a 512px)**: Decodificar una imagen de 8 Megapixels en un solo bloque VAE consumiría >14 GB de VRAM contigua de activación, disparando un error CUDA OOM. Con `VAEDecodeTiled` (`tile_size = 512`, `overlap = 64`), la decodificación se procesa en pequeños fragmentos con solapamiento suave, manteniendo el consumo pico de VRAM por debajo de los **2 GB** durante todo el renderizado.

### 📥 Cómo Descargar P-H3-Image
Hacé doble clic en **`Windows/6_Descargar_P-H3-Image.bat`** (o en **`Mac/6_Descargar_P-H3-Image_Mac.command`** en macOS):
- Descargará con barra de progreso y reanudación automática (`.part`):
  - `models/diffusion_models/p-h3-image_int8_convrot.safetensors` (~21.2 GB)
  - `models/text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors` (~5.1 GB)
  - `models/vae/p-h3-image_vae_int8_convrot.safetensors` (~3.1 GB)
- *(Opcional)*: También incluye la opción para descargar la variante ultra-comprimida **W4A8** (~13.4 GB).

---

## 🚀 Instalación y Uso en Windows

### Paso 1: Configurar Hardware y Descargar Modelo
Hacé doble clic en **`Windows/1_Instalar_Modelos.bat`**.
- El asistente detectará tu GPU y VRAM, te mostrará el diagnóstico de compatibilidad y te permitirá elegir el nivel (**`[1] BASICO`**, **`[2] MEDIO`** o **`[3] AVANZADO`** para GPUs potentes como RTX 3080/4080/4090).
- Si no tenés ComfyUI instalado, elegí la opción **`[1]`** para que descargue e instale **ComfyUI Portable** automáticamente dentro de esta carpeta (o la opción **`[2]`** para vincular un ComfyUI que ya tengas en tu PC).

### Paso 2: Iniciar Módulo 1 (Generador de Imágenes Alta Calidad • Puerto 8080)
Hacé doble clic en **`Windows/2_Iniciar_Sistema_TD_Comfy.bat`**.
- Levantará **ComfyUI en segundo plano** (`http://127.0.0.1:8188`).
- Iniciará el **Módulo 1 (Generador de Imágenes HQ)** en `http://127.0.0.1:8080` y también levantará en segundo plano el **Módulo 2 (StreamDiffusionV2 Studio)** en `http://127.0.0.1:8081`.

### Paso 3: Iniciar Módulo 2 Independiente (StreamDiffusionV2 + ScreenDiffusion Studio • Puerto 8081)
Si querés abrir directamente la WebApp de tiempo real (**StreamDiffusionV2 + ComfyUI_StreamDiffusion + ScreenDiffusion**), hacé doble clic en **`Windows/3_Iniciar_StreamDiffusion_V2_Studio.bat`** (o en **`Mac/3_Iniciar_StreamDiffusion_Mac.command`** en macOS), o abrí `http://127.0.0.1:8081`.

---

## 🍎 Instalación y Uso en macOS (Apple Silicon M1 / M2 / M3 / M4 e Intel)

Sí, **el sistema y el modo StreamDiffusion Realtime son 100% compatibles con macOS**. En Macs con chip Apple Silicon (`M1`, `M2`, `M3`, `M4`), ComfyUI utiliza aceleración por hardware **Apple Metal (`MPS`)** y Memoria Unificada con el flag `--force-fp16` configurado automáticamente por el servidor puente.

### Paso 1: Dar permisos de ejecución (solo la primera vez)
Abrí la aplicación **Terminal** en tu Mac, arrastrá la carpeta del proyecto o entrá en ella y ejecutá:

```bash
chmod +x Mac/*.command
```

### Paso 2: Instalar ComfyUI y el Modelo en Mac
Hacé doble clic en **`Mac/1_Instalar_Mac.command`** (o ejecutalo con `./Mac/1_Instalar_Mac.command` en Terminal).
- El instalador detectará tu chip Apple Silicon y cuánta Memoria Unificada tenés disponible.
- Si no tenés ComfyUI instalado (ni la versión Desktop en `/Applications/ComfyUI.app`), elegí **`[1]`**: el script clonará ComfyUI, creará el entorno `.venv` e instalará **PyTorch con soporte Metal (MPS)** automáticamente.
- **Recomendación para usuarios de Mac**: Elegí el **Nivel 0 (`REALTIME — StreamDiffusion-RT: Hyper-SD 1-Step + TAESD`)** o el **Nivel 1 (`Básico — DreamShaper 8 + LCM`)** para obtener la máxima fluidez en Metal.

### Paso 3: Iniciar el Sistema en Mac
Hacé doble clic en **`Mac/2_Iniciar_Sistema_Mac.command`** (o ejecutá `./Mac/2_Iniciar_Sistema_Mac.command`).
- Se iniciará ComfyUI en segundo plano con `--force-fp16` y se abrirá la WebApp en `http://127.0.0.1:8080`.

---

## 🤖 Guía de Auto-Instalación y Auto-Reparación con Agentes LLM (Claude Code, Antigravity, Freebuff, Cursor)

Si sos alumno y usás un asistente de IA con acceso a terminal (**Claude Code / Cloud**, **Google Antigravity**, **Freebuff**, **Cursor**, **Windsurf** o **OpenCode**), podés pedirle que **instale, configure, pruebe y repare automáticamente todo el sistema** en tu computadora (Windows o Mac).

### 📋 Copiá y pegale este Prompt exacto a tu Agente LLM:

```markdown
Necesito que instales, configures, verifiques y dejes funcionando al 100% este repositorio (Puente TouchDesigner <-> ComfyUI + StreamDiffusion Realtime + WebApp) en mi máquina. Seguí este protocolo autónomo sin detenerte hasta que el autodiagnóstico pase con `all_ok: true`:

1. **Inspección de Hardware y Entorno**:
   - Ejecutá `python bridge/installer.py --detect-json` (en Mac usá `python3`).
   - Leé el JSON devuelto: verificá el sistema operativo (`os`), el tipo de GPU (`vendor`: `nvidia`, `apple_mps`, `amd` o `cpu_integrated`), la VRAM (`vram_gb`), el nivel recomendado (`recommended_tier`) y el bloque `compatibility`.

2. **Dependencias Python del Puente**:
   - Asegurate de que `Pillow` esté instalado ejecutando `python -m pip install pillow`.

3. **Verificación o Instalación de ComfyUI**:
   - Si en el paso 1 `comfy_installation` dio `null`:
     - Buscá primero si ya tengo una carpeta `ComfyUI` o `ComfyUI_windows_portable` en mi usuario (`~/Documents`, `C:\`, `D:\`, `/Applications/ComfyUI.app`). Si la encontrás, agregala al inicio de `comfy_installations` en `bridge/config.json`.
     - Si no tengo ComfyUI en ningún lado, ejecutá `python bridge/installer.py --install-comfy` para instalarlo automáticamente.

4. **Descarga del Modelo StreamDiffusion Realtime y Nivel Recomendado**:
   - Ejecutá `python bridge/installer.py --download-tier realtime` para instalar el paquete ligero de tiempo real (`DreamShaper 8 + Hyper-SD 1-Step LoRA + TAESD TinyVAE`).

5. **Ejecución del Autodiagnóstico End-to-End y Auto-Reparación**:
   - Ejecutá `python bridge/selftest.py`.
   - Este script verifica archivos, levanta el servidor puente en `http://127.0.0.1:8080`, inicia ComfyUI en segundo plano y dispara una generación real verificando que `bridge/outputs/latest.png` sea una imagen PNG válida.
   - **Si cualquier paso de `bridge/selftest.py` falla (`all_ok: false`)**:
     - Leé el traceback o mensaje de error en los logs (`http://127.0.0.1:8080/api/status`), diagnosticá la causa (puerto ocupado, ruta de Python incorrecta en `bridge/config.json`, enlace simbólico roto en `models/loras`, falta de VRAM que requiere cambiar a `--lowvram` o `--cpu`, o dependencia faltante en el venv de ComfyUI), corregilo en el código o configuración y volvé a correr `python bridge/selftest.py` en bucle hasta que devuelva `"all_ok": true`.

6. **Reporte Final**:
   - Cuando `"all_ok": true` esté confirmado, mostrame un resumen con mi GPU detectada, el modelo activo y confirmame que `http://127.0.0.1:8080` y `TD/v1.toe` están listos para usar.
```

---

## 🛠️ Diagnóstico de Compatibilidad en la WebApp y Soluciones Frecuentes

La WebApp (`http://127.0.0.1:8080`) incluye en su panel izquierdo la tarjeta **"Diagnóstico de Hardware y Soluciones"**, que evalúa automáticamente tu computadora:

- **Caso A: Placa NVIDIA con 3–4 GB de VRAM (ej. GTX 1650 / RTX 3050 Laptop)**
  - **Diagnóstico**: Si elegís un modelo pesado (SDXL o Flux), ocurrirá un error *CUDA Out of Memory (OOM)*.
  - **Solución automática**: Seleccioná el nivel **`⚡ StreamDiffusion RT`** (`DreamShaper 8 + Hyper-SD 1-Step + TAESD`), resolución `512x512` (o `384x384`) y `1 paso`. El puente activa `--lowvram` sólo en placas de menos de 6.5 GB y usa el TinyVAE de 5 MB para evitar picos de memoria.
- **Caso B: Usuarios de Mac (Apple Silicon M1/M2/M3/M4)**
  - **Diagnóstico**: Los modelos en formato `FP8` (como `flux1-schnell-fp8`) están diseñados para núcleos Tensor de NVIDIA serie 40 y pueden dar error en Metal (`MPS`).
  - **Solución automática**: Usá el nivel **`⚡ StreamDiffusion RT`** o **Nivel Básico / Medio (FP16)**. El puente detecta `darwin` y pasa automáticamente `--force-fp16` a ComfyUI.
- **Caso C: Placas AMD Radeon o Notebooks con GPU Integrada Intel**
  - **Diagnóstico**: No disponen de NVIDIA CUDA.
  - **Solución automática**: El servidor puente activa el **Modo Práctica Procedural** cuando no hay backend GPU disponible o permite correr ComfyUI con `--cpu` / `--directml`, garantizando que el alumno siempre pueda usar TouchDesigner en clase a 60 FPS sin bloqueos.

---

## 🎨 Cómo usar StreamDiffusion Realtime en TouchDesigner (`TD/v1.toe`) y en la WebApp

### En TouchDesigner (`TD/v1.toe` / `TD/ComfyUI_Studio_Bridge.tox`)
1. Abrí [`TD/v1.toe`](./TD/v1.toe) en TouchDesigner (Windows o macOS) o arrastrá [`TD/ComfyUI_Studio_Bridge.tox`](./TD/ComfyUI_Studio_Bridge.tox) a tu propio proyecto.
2. Conectá cualquier operador visual en movimiento (cámara web `videodevinTOP`, ruido `noiseTOP`, partículas o geometría 3D) a la primera entrada de **`/project1/comfy_studio`** (por defecto ya viene conectado `/project1/entrada_img2img`).
3. En los parámetros de **`/project1/comfy_studio`** (página **`ComfyUI Control`**):
   - **Modo Generación (`Mode`)**: Seleccioná **`StreamDiffusion Realtime (1-2 Steps + TAESD)`**.
   - **Nivel Hardware (`Tier`)**: Seleccioná **`StreamDiffusion RT (3GB+ / Mac - HyperSD + TAESD)`**.
   - **Turbo TAESD + SimFilter (`Realtimemode`)**: Activo (`On`).
   - **Suavizado Temporal 60FPS (`Smooth60fps`)**: Activo (`On`) — interpola en GPU cada frame generado por la IA para obtener movimiento continuo a **60 FPS** en `/project1/SALIDA_COMFYUI` y `/project1/comparador_img2img`.
   - **Auto-Stream Continuo (`Autostream`)**: Activo (`On`) — en cuanto termina un frame (~180–240 ms), TouchDesigner dispara el siguiente en el mismo milisegundo sin esperas de polling.
   - **Pasos (`Steps`)**: `1` (con Hyper-SD 1-Step LoRA) o `2`.
   - **Denoise (`Denoise`)**: `0.50` a `0.65` (controla el balance entre la estructura del TOP de entrada en vivo y la transformación creativa del Prompt).

### En la WebApp (`http://127.0.0.1:8080`)
1. Hacé clic en la tarjeta **`⚡ StreamDiffusion RT`** en el panel izquierdo.
2. En **Modo**, elegí **`⚡ StreamDiffusion Realtime (1-2 Steps + TAESD TinyVAE + TD Input)`**.
3. Hacé clic en el botón **`🔴 Iniciar Loop StreamDiffusion Realtime (Continuo)`** para controlar o disparar el flujo continuo desde el navegador mientras ves la entrada de TouchDesigner y la salida IA actualizándose en vivo.

---

## 📁 Estructura del Repositorio

```text
comfyui_td_webapp/
├── Windows/                         # Lanzadores y scripts para Windows (.bat)
│   ├── 1_Instalar_Modelos.bat       # Instalador interactivo para Windows (incluye opción [0] StreamDiffusion RT)
│   ├── 2_Iniciar_Sistema_TD_Comfy.bat # Lanzador en 1 clic para Módulo 1 (Imágenes HQ en 8080)
│   ├── 3_Iniciar_StreamDiffusion_V2_Studio.bat # Lanzador en 1 clic para Módulo 2 (StreamDiffusionV2 en 8081)
│   ├── 4_Iniciar_Trellis3D_Studio.bat # Lanzador en 1 clic para Módulo 3 (Trellis v2 + Hunyuan 3D en 8082)
│   ├── 5_Descargar_Modelos_3D.bat   # Descargador oficial de pesos 3D (Hunyuan 3D v2.1 y TRELLIS.2)
│   ├── 6_Descargar_P-H3-Image.bat   # Descargador oficial P-H3-Image INT8 ConvRot (8MP / 8 Pasos)
│   ├── 7_Iniciar_Video_Studio.bat   # Lanzador en 1 clic para Módulo 4 (Video Studio en 8083)
│   ├── 8_Iniciar_Segmentacion_Studio.bat # Lanzador en 1 clic para Módulo 5 (Segmentación Alfa & Inpaint en 8084)
│   ├── 9_Iniciar_Parallax_Studio.bat # Lanzador en 1 clic para Módulo 6 (Parallax 2.5D en 8085)
│   ├── 10_Iniciar_Poster_Studio.bat # Lanzador en 1 clic para Módulo 7 (Poster Studio MOSS-Image en 8086)
│   └── Iniciar_Suite_Completa.bat   # Inicia los 6 módulos principales simultáneamente
├── Mac/                             # Lanzadores y scripts para macOS (.command)
│   ├── 1_Instalar_Mac.command       # Instalador interactivo para macOS (Apple Silicon / Intel)
│   ├── 2_Iniciar_Sistema_Mac.command # Lanzador Módulo 1 para macOS
│   ├── 3_Iniciar_StreamDiffusion_Mac.command # Lanzador Módulo 2 para macOS
│   ├── 4_Iniciar_Trellis3D_Mac.command # Lanzador Módulo 3 para macOS
│   ├── 5_Descargar_Modelos_3D_Mac.command # Descargador de pesos 3D para macOS
│   ├── 6_Descargar_P-H3-Image_Mac.command # Descargador P-H3-Image para macOS
│   ├── 7_Iniciar_Video_Mac.command  # Lanzador Módulo 4 para macOS
│   ├── 8_Iniciar_Segmentacion_Mac.command # Lanzador Módulo 5 para macOS
│   ├── 9_Iniciar_Parallax_Mac.command # Lanzador Módulo 6 para macOS
│   ├── 10_Iniciar_Poster_Mac.command # Lanzador Módulo 7 para macOS
│   └── Iniciar_Suite_Completa_Mac.command # Inicia la suite completa en macOS
├── README.md                        # Documentación completa + Arquitectura StreamDiffusion + Guía LLM
├── TD/
│   ├── ComfyUI_StreamDiffusion_Studio.toe # Proyecto TouchDesigner optimizado para ComfyUI + StreamDiffusion
│   ├── v1.toe                       # Proyecto TouchDesigner configurado a 60 FPS con SOP_ASSET_3D y StreamDiffusion RT
│   └── ComfyUI_Studio_Bridge.tox    # Componente .tox portable con interpolador GPU 60 FPS
├── bridge/                          # Módulo 1: Generador de Imágenes HQ + P-H3-Image INT8 (8080)
│   ├── download_ph3_image.py        # Descargador CLI oficial de P-H3-Image INT8 ConvRot / W4A8
│   ├── workflows.py                 # Constructores de flujos ComfyUI (P-H3-Image, Flux.2, SDXL, SD1.5)
│   └── server.py                    # Servidor API Módulo 1 con auto-detección y endpoints TouchDesigner
├── streamdiffusion_studio/          # Módulo 2: StreamDiffusion Realtime + Pizarra Pro + Captura Ventana (8081)
├── trellis3d_studio/                # Módulo 3: TRELLIS v2 + Hunyuan 3D Local Studio (8082)
├── video_studio/                    # Módulo 4: AI Video Studio T2V & I2V Wan 2.2 / Parallax (8083)
├── segmentation_studio/             # Módulo 5: Segmentación Alfa & Inpainting Studio (8084)
├── parallax25d_studio/              # Módulo 6: 2.5D Parallax & Depth Maps para TouchDesigner (8085)
├── poster_studio/                   # Módulo 7: Poster Studio MOSS-Image Layout & Typography (8086)
└── webapp/                          # WebApp Módulo 1 con galería, lightbox y selector P-H3-Image 8MP
    ├── index.html                   # Interfaz Web principal con modal interactivo del tutorial
    └── tutorial.html                # Centro de documentación completa, parámetros y repositorios
```

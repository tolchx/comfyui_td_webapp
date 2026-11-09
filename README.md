# 📘 Sistema Modular TouchDesigner ↔ ComfyUI (Windows & macOS)
## Módulo 1: Generador de Imágenes Alta Calidad (`8080`) + Módulo 2: StreamDiffusionV2 & ScreenDiffusion Studio (`8081`)

Arquitectura **modular, standalone y multiplataforma (Windows & macOS)** diseñada para cátedras y talleres de Inteligencia Artificial Generativa. Separa el sistema en **dos módulos independientes y complementarios** conectados a **ComfyUI** y **TouchDesigner**:

1. **🎨 Módulo 1 — Generador de Imágenes de Alta Calidad (`bridge/` + `webapp/` • Puerto `8080`)**:
   - Dedicado exclusivamente a **Text-to-Image (`txt2img`)** e **Image-to-Image (`img2img`)** de máxima fidelidad visual.
   - **Auto-detección de Hardware Potente**: Al detectar GPUs de gama alta (ej. **NVIDIA GeForce RTX 4090 24 GB VRAM**), activa automáticamente el nivel **`AVANZADO`** con **Flux.2 Klein 4B / Flux.2 Dev / Qwen Image 2.1** a `1024×1024` y decodificador **Full VAE** (`flux2-vae.safetensors`), evitando los artefactos y deformaciones de los modos de 1 paso.
   - **Galería e Historial Limpio + Lightbox Pantalla Completa**: El historial almacena únicamente generaciones `txt2img` e `img2img`, e incluye un visor **Lightbox (`⛶`)** con flechas de navegación (`❮` / `❯` y teclado `←` / `→`) y un **carrusel inferior de miniaturas**.

2. **🎬 Módulo 2 — StreamDiffusionV2 + ComfyUI_StreamDiffusion + ScreenDiffusion Studio (`streamdiffusion_studio/` • Puerto `8081`)**:
   - Módulo separado en la carpeta [`streamdiffusion_studio/`](streamdiffusion_studio/) que fusiona tres tecnologías de tiempo real conectadas directamente a **ComfyUI (`127.0.0.1:8188`)** y **TouchDesigner**:
     - **[StreamDiffusionV2 (`chenfengxu714/StreamDiffusionV2`)](https://github.com/chenfengxu714/StreamDiffusionV2)**: Escalado dinámico según GPU (`Flux.2 Klein 4B` en 4 pasos con Full VAE a ~610 ms en RTX 4090, o `DreamShaper 8 + Hyper-SD` multi-paso) + **Motion-Aware Temporal Controller** (analiza el delta de movimiento entre fotogramas para estabilizar rostros y detalles sin deformación ni parpadeo).
     - **[ComfyUI_StreamDiffusion (`jesenzhang/ComfyUI_StreamDiffusion`)](https://github.com/jesenzhang/ComfyUI_StreamDiffusion)**: Integración nativa con los nodos personalizados `StreamDiffusion_Loader` y `StreamDiffusion_Sampler` en `ComfyUI/custom_nodes/ComfyUI_StreamDiffusion`.
     - **[ScreenDiffusion (`rudyaa-sd/ScreenDiffusion`)](https://github.com/rudyaa-sd/ScreenDiffusion)**: Captura en vivo de cualquier región de tu pantalla o ventana del sistema operativo (`PIL.ImageGrab` nativo + `getDisplayMedia` WebRTC), además de **Webcam en vivo**, **Pizarra de Boceto Interactiva** y entrada desde **TouchDesigner TOP**.

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

---

## 🚀 Instalación y Uso en Windows

### Paso 1: Configurar Hardware y Descargar Modelo
Hacé doble clic en **`1_Instalar_Modelos.bat`**.
- El asistente detectará tu GPU y VRAM, te mostrará el diagnóstico de compatibilidad y te permitirá elegir el nivel (**`[1] BASICO`**, **`[2] MEDIO`** o **`[3] AVANZADO`** para GPUs potentes como RTX 3080/4080/4090).
- Si no tenés ComfyUI instalado, elegí la opción **`[1]`** para que descargue e instale **ComfyUI Portable** automáticamente dentro de esta carpeta (o la opción **`[2]`** para vincular un ComfyUI que ya tengas en tu PC).

### Paso 2: Iniciar Módulo 1 (Generador de Imágenes Alta Calidad • Puerto 8080)
Hacé doble clic en **`2_Iniciar_Sistema_TD_Comfy.bat`**.
- Levantará **ComfyUI en segundo plano** (`http://127.0.0.1:8188`).
- Iniciará el **Módulo 1 (Generador de Imágenes HQ)** en `http://127.0.0.1:8080` y también levantará en segundo plano el **Módulo 2 (StreamDiffusionV2 Studio)** en `http://127.0.0.1:8081`.

### Paso 3: Iniciar Módulo 2 Independiente (StreamDiffusionV2 + ScreenDiffusion Studio • Puerto 8081)
Si querés abrir directamente la WebApp de tiempo real (**StreamDiffusionV2 + ComfyUI_StreamDiffusion + ScreenDiffusion**), hacé doble clic en **`3_Iniciar_StreamDiffusion_V2_Studio.bat`** (o en **`3_Iniciar_StreamDiffusion_Mac.command`** en macOS), o abrí `http://127.0.0.1:8081`.

---

## 🍎 Instalación y Uso en macOS (Apple Silicon M1 / M2 / M3 / M4 e Intel)

Sí, **el sistema y el modo StreamDiffusion Realtime son 100% compatibles con macOS**. En Macs con chip Apple Silicon (`M1`, `M2`, `M3`, `M4`), ComfyUI utiliza aceleración por hardware **Apple Metal (`MPS`)** y Memoria Unificada con el flag `--force-fp16` configurado automáticamente por el servidor puente.

### Paso 1: Dar permisos de ejecución (solo la primera vez)
Abrí la aplicación **Terminal** en tu Mac, arrastrá la carpeta del proyecto o entrá en ella y ejecutá:

```bash
chmod +x 1_Instalar_Mac.command 2_Iniciar_Sistema_Mac.command
```

### Paso 2: Instalar ComfyUI y el Modelo en Mac
Hacé doble clic en **`1_Instalar_Mac.command`** (o ejecutalo con `./1_Instalar_Mac.command` en Terminal).
- El instalador detectará tu chip Apple Silicon y cuánta Memoria Unificada tenés disponible.
- Si no tenés ComfyUI instalado (ni la versión Desktop en `/Applications/ComfyUI.app`), elegí **`[1]`**: el script clonará ComfyUI, creará el entorno `.venv` e instalará **PyTorch con soporte Metal (MPS)** automáticamente.
- **Recomendación para usuarios de Mac**: Elegí el **Nivel 0 (`REALTIME — StreamDiffusion-RT: Hyper-SD 1-Step + TAESD`)** o el **Nivel 1 (`Básico — DreamShaper 8 + LCM`)** para obtener la máxima fluidez en Metal.

### Paso 3: Iniciar el Sistema en Mac
Hacé doble clic en **`2_Iniciar_Sistema_Mac.command`** (o ejecutá `./2_Iniciar_Sistema_Mac.command`).
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
├── 1_Instalar_Modelos.bat           # Instalador interactivo para Windows (incluye opción [0] StreamDiffusion RT)
├── 2_Iniciar_Sistema_TD_Comfy.bat   # Lanzador en 1 clic para Windows
├── 1_Instalar_Mac.command           # Instalador interactivo para macOS (Apple Silicon / Intel)
├── 2_Iniciar_Sistema_Mac.command    # Lanzador en 1 clic para macOS
├── README.md                        # Documentación completa + Arquitectura StreamDiffusion + Guía LLM
├── TD/
│   ├── v1.toe                       # Proyecto TouchDesigner configurado a 60 FPS con StreamDiffusion RT activo
│   └── ComfyUI_Studio_Bridge.tox    # Componente .tox portable con interpolador GPU 60 FPS
├── bridge/
│   ├── config.json                  # Configuración de niveles (Realtime, Básico, Medio, Avanzado) y rutas
│   ├── installer.py                 # Detector de GPU/VRAM/Mac MPS, diagnóstico y descargador HF + TAESD
│   ├── workflows.py                 # Generador dinámico de workflows API (StreamDiffusion 1-Step + TAESD, SDXL, Flux, Qwen)
│   ├── server.py                    # Servidor HTTP puente de baja latencia + filtro de similitud StreamDiffusion
│   ├── selftest.py                  # Suite de autodiagnóstico y auto-reparación para agentes LLM
│   └── build_td_network.py          # Script constructor de la red TouchDesigner vía TDMCP
└── webapp/
    └── index.html                   # WebApp con modo StreamDiffusion RT, diagnóstico de hardware, visor dual e historial
```

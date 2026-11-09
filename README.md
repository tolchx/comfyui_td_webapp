# 📘 Puente TouchDesigner ↔ ComfyUI + WebApp (Windows & macOS)

Sistema **standalone y multiplataforma (Windows & macOS)** diseñado para cátedras y talleres de Inteligencia Artificial Generativa. Permite conectar **TouchDesigner** con **ComfyUI** en tiempo real (**Text-to-Image**, **Image-to-Image** desde cualquier TOP y **Auto-Stream continuo**), integrando una **WebApp de monitoreo de hardware, diagnóstico de compatibilidad, historial y control interactivo**.

Repositorio oficial: **[https://github.com/tolchx/comfyui_td_webapp](https://github.com/tolchx/comfyui_td_webapp)**

---

## ✨ Características Principales

1. **Detección Automática de Hardware y Diagnóstico Inteligente**:
   - Detecta automáticamente placas **NVIDIA CUDA** (Windows/Linux), chips **Apple Silicon M1/M2/M3/M4 con Metal (MPS)** en macOS, placas **AMD Radeon** o **GPUs integradas / CPU**.
   - Evalúa en vivo si el modelo elegido es compatible con la memoria VRAM disponible y **muestra en la WebApp alertas y soluciones paso a paso** cuando detecta cuellos de botella o falta de memoria.
2. **Instalación por Niveles (Básico / Medio / Avanzado)**:
   - Descarga automáticamente el modelo generador óptimo desde HuggingFace con barra de progreso y soporte para reanudar descargas interrumpidas.
3. **Modo Práctica Procedural Integrado (Cero VRAM / Sin ComfyUI)**:
   - Si un alumno está en clase con una notebook sin GPU dedicada o mientras descarga ComfyUI, el servidor puente genera arte procedural reactivo al Prompt y al TOP de entrada (`img2img`) para que pueda practicar todo el flujo en TouchDesigner a **60 FPS** desde el primer minuto.
4. **Sincronización Bidireccional TouchDesigner ↔ WebApp**:
   - Cualquier generación disparada desde la WebApp actualiza automáticamente la textura `SALIDA_COMFYUI` dentro de TouchDesigner.
   - Cualquier generación disparada desde TouchDesigner aparece en vivo en el monitor y en el historial de la WebApp.

---

## 🖥️ Requisitos y Niveles de Modelos según tu Hardware

| Nivel | Hardware Recomendado (Windows / Mac) | Modelo Instalado Automáticamente | Peso | Flags Automáticos |
| :--- | :--- | :--- | :--- | :--- |
| **1. Básico** | **NVIDIA 4–6 GB VRAM** (GTX 1650, RTX 3050)<br>**Mac Apple Silicon 8–16 GB RAM** (M1/M2/M3) | **DreamShaper 8 (SD 1.5) + LCM LoRA** | ~2.1 GB | Windows: `--lowvram`<br>Mac: `--force-fp16` |
| **2. Medio** | **NVIDIA 8–10 GB VRAM** (RTX 3060, 3070, 4060)<br>**Mac Apple Silicon 16–24 GB RAM** (M1 Pro/Max, M2/M3/M4) | **SDXL Turbo FP16** | ~6.6 GB | Windows: `--normalvram`<br>Mac: `--force-fp16` |
| **3. Avanzado** | **NVIDIA 12 GB+ VRAM** (RTX 3080, 4070, 4080, 4090)<br>**Mac Apple Silicon 32 GB+ RAM** | **Flux.1 Schnell FP8** *(+ soporte nativo para Flux.2 Klein 4B / Dev y Qwen Image 2.1)* | ~16.4 GB | Windows: `--normalvram`<br>Mac: `--force-fp16` |

---

## 🚀 Instalación y Uso en Windows (En 2 clics)

### Paso 1: Configurar Hardware y Descargar Modelo
Hacé doble clic en **`1_Instalar_Modelos.bat`**.
- El asistente detectará tu GPU y VRAM, te mostrará el diagnóstico de compatibilidad y te sugerirá el nivel (Básico, Medio o Avanzado).
- Si no tenés ComfyUI instalado, elegí la opción **`[1]`** para que descargue e instale **ComfyUI Portable** automáticamente dentro de esta carpeta (o la opción **`[2]`** para vincular un ComfyUI que ya tengas en tu PC).

### Paso 2: Iniciar el Sistema
Hacé doble clic en **`2_Iniciar_Sistema_TD_Comfy.bat`**.
- Levantará **ComfyUI en segundo plano** (`http://127.0.0.1:8188`).
- Iniciará el **Servidor Puente** y abrirá automáticamente la **WebApp** en tu navegador (`http://127.0.0.1:8080`).

---

## 🍎 Instalación y Uso en macOS (Apple Silicon M1 / M2 / M3 / M4 e Intel)

Sí, **el sistema es 100% compatible con macOS**. En Macs con chip Apple Silicon (`M1`, `M2`, `M3`, `M4`), ComfyUI utiliza aceleración por hardware **Apple Metal (`MPS`)** y Memoria Unificada con el flag `--force-fp16` configurado automáticamente por el servidor puente.

### Paso 1: Dar permisos de ejecución (solo la primera vez)
Abrí la aplicación **Terminal** en tu Mac, arrastrá la carpeta del proyecto o entrá en ella y ejecutá:

```bash
chmod +x 1_Instalar_Mac.command 2_Iniciar_Sistema_Mac.command
```

### Paso 2: Instalar ComfyUI y el Modelo en Mac
Hacé doble clic en **`1_Instalar_Mac.command`** (o ejecutalo con `./1_Instalar_Mac.command` en Terminal).
- El instalador detectará tu chip Apple Silicon y cuánta Memoria Unificada tenés disponible.
- Si no tenés ComfyUI instalado (ni la versión Desktop en `/Applications/ComfyUI.app`), elegí **`[1]`**: el script clonará ComfyUI, creará el entorno `.venv` e instalará **PyTorch con soporte Metal (MPS)** automáticamente.
- **Recomendación para usuarios de Mac**: Elegí el **Nivel 1 (Básico — DreamShaper 8 + LCM)** si tu Mac tiene 8 GB o 16 GB de RAM unificada, o el **Nivel 2 (Medio — SDXL Turbo FP16)** si tenés 16 GB a 32 GB+. Evitá modelos cuantizados en `FP8` en Mac ya que FP8 es específico de arquitecturas NVIDIA Ada/Hopper.

### Paso 3: Iniciar el Sistema en Mac
Hacé doble clic en **`2_Iniciar_Sistema_Mac.command`** (o ejecutá `./2_Iniciar_Sistema_Mac.command`).
- Se iniciará ComfyUI en segundo plano con `--force-fp16` y se abrirá la WebApp en `http://127.0.0.1:8080`.

---

## 🤖 Guía de Auto-Instalación y Auto-Reparación con Agentes LLM (Claude Code, Antigravity, Freebuff, Cursor)

Si sos alumno y usás un asistente de IA con acceso a terminal (**Claude Code / Cloud**, **Google Antigravity**, **Freebuff**, **Cursor**, **Windsurf** o **OpenCode**), podés pedirle que **instale, configure, pruebe y repare automáticamente todo el sistema** en tu computadora (Windows o Mac).

### 📋 Copiá y pegale este Prompt exacto a tu Agente LLM:

```markdown
Necesito que instales, configures, verifiques y dejes funcionando al 100% este repositorio (Puente TouchDesigner <-> ComfyUI + WebApp) en mi máquina. Seguí este protocolo autónomo sin detenerte hasta que el autodiagnóstico pase con `all_ok: true`:

1. **Inspección de Hardware y Entorno**:
   - Ejecutá `python bridge/installer.py --detect-json` (en Mac usá `python3`).
   - Leé el JSON devuelto: verificá el sistema operativo (`os`), el tipo de GPU (`vendor`: `nvidia`, `apple_mps`, `amd` o `cpu_integrated`), la VRAM (`vram_gb`), el nivel recomendado (`recommended_tier`) y el bloque `compatibility`.

2. **Dependencias Python del Puente**:
   - Asegurate de que `Pillow` esté instalado ejecutando `python -m pip install pillow`.

3. **Verificación o Instalación de ComfyUI**:
   - Si en el paso 1 `comfy_installation` dio `null`:
     - Buscá primero si ya tengo una carpeta `ComfyUI` o `ComfyUI_windows_portable` en mi usuario (`~/Documents`, `C:\`, `D:\`, `/Applications/ComfyUI.app`). Si la encontrás, agregala al inicio de `comfy_installations` en `bridge/config.json`.
     - Si no tengo ComfyUI en ningún lado, ejecutá `python bridge/installer.py --install-comfy` para instalarlo automáticamente.

4. **Descarga del Modelo según mi Hardware**:
   - Si `installed_checkpoints` está vacío (`[]`), descargá automáticamente el modelo del nivel recomendado ejecutando:
     `python bridge/installer.py --download-tier <recommended_tier>`
     (donde `<recommended_tier>` es `basico`, `medio` o `avanzado` según lo detectado en el paso 1).

5. **Ejecución del Autodiagnóstico End-to-End y Auto-Reparación**:
   - Ejecutá `python bridge/selftest.py`.
   - Este script verifica archivos, levanta el servidor puente en `http://127.0.0.1:8080`, inicia ComfyUI en segundo plano y dispara una generación real verificando que `bridge/outputs/latest.png` sea una imagen PNG válida.
   - **Si cualquier paso de `bridge/selftest.py` falla (`all_ok: false`)**:
     - Leé el traceback o mensaje de error en los logs (`http://127.0.0.1:8080/api/status`), diagnosticá la causa (puerto ocupado, ruta de Python incorrecta en `bridge/config.json`, falta de VRAM que requiere cambiar a `--lowvram` o `--cpu`, o dependencia faltante en el venv de ComfyUI), corregilo en el código o configuración y volvé a correr `python bridge/selftest.py` en bucle hasta que devuelva `"all_ok": true`.

6. **Reporte Final**:
   - Cuando `"all_ok": true` esté confirmado, mostrame un resumen con mi GPU detectada, el modelo activo y confirmame que `http://127.0.0.1:8080` y `TD/v1.toe` están listos para usar.
```

---

## 🛠️ Diagnóstico de Compatibilidad en la WebApp y Soluciones Frecuentes

La WebApp (`http://127.0.0.1:8080`) incluye en su panel izquierdo la tarjeta **"Diagnóstico de Hardware y Soluciones"**, que evalúa automáticamente tu computadora:

- **Caso A: Placa NVIDIA con 4 GB o menos de VRAM (ej. GTX 1650 / RTX 3050 Laptop)**
  - **Diagnóstico**: Si elegís un modelo pesado (SDXL o Flux), ocurrirá un error *CUDA Out of Memory (OOM)*.
  - **Solución automática**: Seleccioná el **Nivel Básico** (`DreamShaper 8 + LCM LoRA`), resolución `512x512` y `4 a 6 pasos`. El puente lanza ComfyUI con `--lowvram` automáticamente.
- **Caso B: Usuarios de Mac (Apple Silicon M1/M2/M3/M4)**
  - **Diagnóstico**: Los modelos en formato `FP8` (como `flux1-schnell-fp8`) están diseñados para núcleos Tensor de NVIDIA serie 40 y pueden dar error en Metal (`MPS`).
  - **Solución automática**: Usá el **Nivel Básico (SD 1.5 LCM)** o **Nivel Medio (SDXL Turbo FP16)**. El puente detecta `darwin` y pasa automáticamente `--force-fp16` a ComfyUI.
- **Caso C: Placas AMD Radeon o Notebooks con GPU Integrada Intel**
  - **Diagnóstico**: No disponen de NVIDIA CUDA.
  - **Solución automática**: El servidor puente activa el **Modo Práctica Procedural** cuando no hay backend GPU disponible o permite correr ComfyUI con `--cpu` / `--directml`, garantizando que el alumno siempre pueda usar TouchDesigner en clase a 60 FPS sin bloqueos.

---

## 🎨 Cómo usar el parche en TouchDesigner (`TD/v1.toe`)

1. Abrí [`TD/v1.toe`](./TD/v1.toe) en TouchDesigner (Windows o macOS) o arrastrá [`TD/ComfyUI_Studio_Bridge.tox`](./TD/ComfyUI_Studio_Bridge.tox) a tu propio proyecto.
2. Seleccioná el contenedor **`/project1/comfy_studio`** y abrí su ventana de parámetros en la página **`ComfyUI Control`**:
   - **Modo (`Mode`)**: Elegí entre `Text-to-Image (Solo Prompt)` o `Image-to-Image (Usa TOP Entrada)`.
   - **Nivel Hardware (`Tier`)**: `Basico`, `Medio` o `Avanzado`.
   - **Prompt Positivo / Negativo**: Escribí lo que querés generar.
   - **Denoise (Img2Img)**: Controla cuánto respeta la forma del TOP conectado a la entrada (`0.3` = muy fiel al TOP original, `0.85` = transforma casi todo).
   - **Generar Ahora (`Generate`)**: Pulsa para crear una nueva imagen.
   - **Enviar TOP + Generar (`Sendframe`)**: Captura el frame actual de `/project1/entrada_img2img`, lo manda a ComfyUI y devuelve la imagen generada en `/project1/SALIDA_COMFYUI`.
   - **Auto-Stream Continuo (`Autostream`)**: Cuando está activo, apenas ComfyUI termina de renderizar una imagen, TouchDesigner dispara automáticamente la siguiente generación en bucle continuo manteniendo **60 FPS** en la red.

---

## 📁 Estructura del Repositorio

```text
comfyui_td_webapp/
├── 1_Instalar_Modelos.bat           # Instalador interactivo para Windows
├── 2_Iniciar_Sistema_TD_Comfy.bat   # Lanzador en 1 clic para Windows
├── 1_Instalar_Mac.command           # Instalador interactivo para macOS (Apple Silicon / Intel)
├── 2_Iniciar_Sistema_Mac.command    # Lanzador en 1 clic para macOS
├── README.md                        # Documentación completa + Guía para Agentes LLM
├── TD/
│   ├── v1.toe                       # Proyecto TouchDesigner configurado a 60 FPS
│   └── ComfyUI_Studio_Bridge.tox    # Componente .tox portable para Windows y Mac
├── bridge/
│   ├── config.json                  # Configuración de niveles (Básico, Medio, Avanzado) y rutas
│   ├── installer.py                 # Detector de GPU/VRAM/Mac MPS, diagnóstico y descargador HF
│   ├── workflows.py                 # Generador dinámico de workflows API (SD1.5, SDXL, Flux, Qwen)
│   ├── server.py                    # Servidor HTTP puente + gestor de ComfyUI en segundo plano
│   ├── selftest.py                  # Suite de autodiagnóstico y auto-reparación para agentes LLM
│   └── build_td_network.py          # Script constructor de la red TouchDesigner vía TDMCP
└── webapp/
    └── index.html                   # WebApp con diagnóstico de hardware, visor dual e historial
```

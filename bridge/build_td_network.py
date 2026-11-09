#!/usr/bin/env python3
"""
Constructor de la Red TouchDesigner `/project1/comfy_studio` en `v1.toe`.
Se conecta al servidor TDMCP activo en 127.0.0.1:13316 y construye:
  1. Generador de entrada en vivo en `/project1` (`noise_source` -> `edge_stylize` -> `null_to_comfy`)
  2. Componente `/project1/comfy_studio` (`baseCOMP`) con Custom Parameters completos:
     - Bridge URL, Prompt, Negative Prompt, Mode (txt2img / img2img), Hardware Tier (basico / medio / avanzado)
     - Width, Height, Steps, CFG, Denoise, Seed
     - Generate (Pulse), Send Frame & Generate (Pulse), Auto Stream (Toggle), Stream Interval (s), Open WebApp (Pulse)
  3. Red interna de `/project1/comfy_studio`:
     - `in_top` -> `input_frame` (nullTOP)
     - `api_generate` (webclientDAT) y `api_upload` (webclientDAT) y `api_poll` (webclientDAT)
     - `poll_timer` (timerCHOP a 0.5s) para sincronizar automáticamente cuando la WebApp o ComfyUI terminan una imagen
     - `ai_result` (moviefileinTOP apuntando a `bridge/outputs/latest.png`)
     - `side_by_side` (layoutTOP comparando Entrada vs Salida IA)
     - `out_ai` (outTOP)
  4. Anotaciones didácticas en español para los alumnos de la cursada.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Tolch\Documents\AI_Code\TDMCP\tolchx-TDMCP\knowledge")
import live

TD_BUILD_CODE = r'''
import json
import td

root = op('/project1')
latest_png_path = r"C:/Users/Tolch/Documents/AI_Code/ic_AI_Modulo/bridge/outputs/latest.png"

# Limpiar operadores viejos de prueba si existen (excepto TDMCP)
for child in list(root.children):
    if child.name not in ('TDMCP',):
        child.destroy()

# -------------------------------------------------------------------------
# 1. FUENTE VISUAL EN VIVO (PARA IMAGE-TO-IMAGE EN TOUCHDESIGNER)
# -------------------------------------------------------------------------
src_noise = root.create(td.noiseTOP, 'fuente_procedural')
src_noise.nodeX, src_noise.nodeY = -520, 120
src_noise.par.outputresolution = 'custom'
src_noise.par.resolutionw = 512
src_noise.par.resolutionh = 512
src_noise.par.type = 'simplex3d'
src_noise.par.period = 1.8
src_noise.par.harmon = 3
src_noise.par.amp = 0.72
src_noise.par.mono = False
src_noise.par.tz.expr = 'absTime.seconds * 0.85'

in_null = root.create(td.nullTOP, 'entrada_img2img')
in_null.nodeX, in_null.nodeY = -300, 120
in_null.inputConnectors[0].connect(src_noise)
in_null.viewer = True

# -------------------------------------------------------------------------
# 2. COMPONENTE PUENTE: /project1/comfy_studio
# -------------------------------------------------------------------------
comp = root.create(td.baseCOMP, 'comfy_studio')
comp.nodeX, comp.nodeY = -40, 120
comp.par.opviewer = './out_ai'
comp.viewer = True

# Custom Parameters
for p in list(comp.customPages):
    p.destroy()

page = comp.appendCustomPage('ComfyUI Control')

p_url = page.appendStr('Bridgeurl', label='Bridge URL (Mod 1: 8080 / Mod 2: 8081)')[0]
p_url.val = 'http://127.0.0.1:8080'

p_mode = page.appendMenu('Mode', label='Modo Generacion')[0]
p_mode.menuNames = ['txt2img', 'img2img']
p_mode.menuLabels = [
    'Text-to-Image Alta Calidad (Solo Prompt)',
    'Image-to-Image Alta Calidad (Usa TOP Entrada)'
]
p_mode.val = 'txt2img'

p_tier = page.appendMenu('Tier', label='Nivel Hardware (Tier)')[0]
p_tier.menuNames = ['basico', 'medio', 'avanzado']
p_tier.menuLabels = [
    'Basico (4-6GB SD1.5 DreamShaper 8 HQ)',
    'Medio (8-10GB SDXL / Flux Schnell)',
    'Avanzado (12-24GB+ RTX 4090 - Flux.2 Klein / Dev / Qwen)'
]
p_tier.val = 'avanzado'

p_rt = page.appendToggle('Realtimemode', label='Enviar a StreamDiffusionV2 Studio (Puerto 8081)')[0]
p_rt.val = False

p_smooth = page.appendToggle('Smooth60fps', label='Suavizado Temporal 60FPS (GPU Blend)')[0]
p_smooth.val = False

p_prompt = page.appendStr('Prompt', label='Prompt Positivo')[0]
p_prompt.val = 'Anime art of Akira, detailed scene, red, masterpiece, crisp lineart, perfect face, volumetric lighting, 8k'

p_neg = page.appendStr('Negativeprompt', label='Prompt Negativo')[0]
p_neg.val = 'low quality, blurry, watermark, text, deformed, bad anatomy'

p_w = page.appendInt('Width', label='Ancho (px)')[0]
p_w.val = 1024
p_w.min, p_w.max = 256, 1536
p_w.clampMin, p_w.clampMax = True, True

p_h = page.appendInt('Height', label='Alto (px)')[0]
p_h.val = 1024
p_h.min, p_h.max = 256, 1536
p_h.clampMin, p_h.clampMax = True, True

p_steps = page.appendInt('Steps', label='Pasos (Steps)')[0]
p_steps.val = 6
p_steps.min, p_steps.max = 1, 40
p_steps.clampMin, p_steps.clampMax = True, True

p_cfg = page.appendFloat('Cfg', label='CFG Scale')[0]
p_cfg.val = 1.0
p_cfg.min, p_cfg.max = 1.0, 15.0

p_denoise = page.appendFloat('Denoise', label='Denoise (Img2Img)')[0]
p_denoise.val = 0.65
p_denoise.min, p_denoise.max = 0.05, 1.0
p_denoise.clampMin, p_denoise.clampMax = True, True

p_seed = page.appendInt('Seed', label='Semilla (-1 = Random)')[0]
p_seed.val = -1

p_gen = page.appendPulse('Generate', label='Generar Imagen HQ Ahora (Pulse)')[0]
p_send = page.appendPulse('Sendframe', label='Enviar TOP + Generar (Img2Img HQ)')[0]
p_send_rt = page.appendPulse('Sendrtframe', label='Enviar Frame a StreamDiffusionV2 (8081)')[0]
p_auto = page.appendToggle('Autostream', label='Auto-Stream Continuo (Hacia Modulo Activo)')[0]
p_auto.val = False

p_interval = page.appendFloat('Streaminterval', label='Intervalo Auto-Stream (s)')[0]
p_interval.val = 4.0
p_interval.min, p_interval.max = 1.0, 30.0

p_web = page.appendPulse('Openwebapp', label='Abrir WebApp Generador HQ (8080)')[0]
p_web_rt = page.appendPulse('Openstreamwebapp', label='Abrir WebApp StreamDiffusionV2 (8081)')[0]
p_web_3d = page.appendPulse('Open3dwebapp', label='Abrir WebApp Trellis v2 / Hunyuan 3D (8082)')[0]

page_stat = comp.appendCustomPage('Estado & Telemetria')
p_st = page_stat.appendStr('Status', label='Estado Servidor')[0]
p_st.val = 'READY'
p_st.readOnly = True

p_genid = page_stat.appendInt('Generationid', label='ID Ultima Generacion')[0]
p_genid.val = 0
p_genid.readOnly = True

p_lat = page_stat.appendFloat('Latencyms', label='Ultima Latencia (ms)')[0]
p_lat.val = 0.0
p_lat.readOnly = True

# -------------------------------------------------------------------------
# 3. RED INTERNA DE /project1/comfy_studio
# -------------------------------------------------------------------------
in1 = comp.create(td.inTOP, 'in1')
in1.nodeX, in1.nodeY = -500, 150

input_frame = comp.create(td.nullTOP, 'input_frame')
input_frame.nodeX, input_frame.nodeY = -300, 150
input_frame.inputConnectors[0].connect(in1)

root = op('/project1')

# Ruta relativa al proyecto TouchDesigner (compatible con Windows y macOS en cualquier carpeta)
ai_movie = comp.create(td.moviefileinTOP, 'ai_result')
ai_movie.nodeX, ai_movie.nodeY = -280, -50
ai_movie.par.file.expr = "project.folder + '/../bridge/outputs/latest.png'"
ai_movie.par.play = False

# Interpolador Temporal GPU a 60 FPS (Feedback + Cross Blend) para fluidez visual continua en StreamDiffusion
fb_smooth = comp.create(td.feedbackTOP, 'fb_smooth')
fb_smooth.nodeX, fb_smooth.nodeY = -100, 60
fb_smooth.inputConnectors[0].connect(ai_movie)

cross_smooth = comp.create(td.crossTOP, 'cross_smooth')
cross_smooth.nodeX, cross_smooth.nodeY = 80, 60
cross_smooth.par.cross = 0.22
cross_smooth.inputConnectors[0].connect(fb_smooth)
cross_smooth.inputConnectors[1].connect(ai_movie)

fb_target = comp.create(td.nullTOP, 'fb_target')
fb_target.nodeX, fb_target.nodeY = 240, 60
fb_target.inputConnectors[0].connect(cross_smooth)
fb_smooth.par.top = 'fb_target'

out_switch = comp.create(td.switchTOP, 'out_switch')
out_switch.nodeX, out_switch.nodeY = 240, -50
out_switch.par.index.expr = "parent().par.Smooth60fps"
out_switch.inputConnectors[0].connect(ai_movie)
out_switch.inputConnectors[1].connect(fb_target)

out_ai = comp.create(td.outTOP, 'out_ai')
out_ai.nodeX, out_ai.nodeY = 420, -50
out_ai.inputConnectors[0].connect(out_switch)

# Telemetria CHOP
metrics = comp.create(td.constantCHOP, 'stream_metrics')
metrics.nodeX, metrics.nodeY = -300, -240
metrics.par.name0 = 'comfy_online'
metrics.par.value0 = 1.0
metrics.par.name1 = 'generating'
metrics.par.value1 = 0.0
metrics.par.name2 = 'latency_ms'
metrics.par.value2 = 0.0
metrics.par.name3 = 'generation_id'
metrics.par.value3 = 0.0

out_chop = comp.create(td.outCHOP, 'out_metrics')
out_chop.nodeX, out_chop.nodeY = -80, -240
out_chop.inputConnectors[0].connect(metrics)

# Tabla de Historial Local en TD
hist_tab = comp.create(td.tableDAT, 'generation_log')
hist_tab.nodeX, hist_tab.nodeY = 180, -240
hist_tab.clear()
hist_tab.appendRow(['id', 'prompt', 'mode', 'tier', 'latency_ms'])

# WebClients: 1 para disparar generaciones y 1 para hacer polling liviano
wc_gen = comp.create(td.webclientDAT, 'api_generate')
wc_gen.nodeX, wc_gen.nodeY = -500, -450

wc_poll = comp.create(td.webclientDAT, 'api_poll')
wc_poll.nodeX, wc_poll.nodeY = -180, -450

# Script controlador central (funciones de envio y recepcion)
ctrl_dat = comp.create(td.textDAT, 'controller_ext')
ctrl_dat.nodeX, ctrl_dat.nodeY = -500, -640
ctrl_dat.text = """import json
import base64
import webbrowser

def trigger_generation(force_img2img=False, force_rt=False):
    comp = parent()
    rt_toggle = bool(comp.par.Realtimemode.eval()) or force_rt
    base_url = 'http://127.0.0.1:8081' if rt_toggle else comp.par.Bridgeurl.eval().rstrip('/')
    raw_mode = comp.par.Mode.eval()

    if rt_toggle:
        mode = 'img2img'
    elif force_img2img and raw_mode == 'txt2img':
        mode = 'img2img'
    else:
        mode = raw_mode

    steps_val = int(comp.par.Steps.eval())
    cfg_val = float(comp.par.Cfg.eval())
    denoise_val = float(comp.par.Denoise.eval())

    payload = {
        'prompt': comp.par.Prompt.eval(),
        'negative_prompt': comp.par.Negativeprompt.eval(),
        'mode': mode,
        'tier': comp.par.Tier.eval(),
        'realtime_mode': False,
        'width': int(comp.par.Width.eval()),
        'height': int(comp.par.Height.eval()),
        'steps': steps_val,
        'cfg': cfg_val,
        'denoise': denoise_val,
        'seed': int(comp.par.Seed.eval()),
        'source': 'touchdesigner',
    }

    if mode == 'img2img' or rt_toggle:
        top_in = op('input_frame')
        if top_in and top_in.width > 0:
            ext = '.jpg' if rt_toggle else '.png'
            img_bytes = bytes(top_in.saveByteArray(ext))
            payload['image_b64'] = base64.b64encode(img_bytes).decode('ascii')

    endpoint = '/api/stream_frame' if rt_toggle else '/api/generate'
    raw_body = json.dumps(payload).encode('utf-8')
    op('api_generate').request(
        base_url + endpoint,
        'POST',
        header={'Content-Type': 'application/json'},
        data=raw_body
    )
    comp.par.Status = 'GENERATING'
    op('stream_metrics').par.value1 = 1.0

def poll_bridge():
    comp = parent()
    rt_toggle = bool(comp.par.Realtimemode.eval())
    base_url = 'http://127.0.0.1:8081' if rt_toggle else comp.par.Bridgeurl.eval().rstrip('/')
    op('api_poll').request(base_url + '/api/td_poll', 'GET')

def open_webapp(stream_studio=False):
    comp = parent()
    if stream_studio:
        webbrowser.open('http://127.0.0.1:8081')
    else:
        webbrowser.open(comp.par.Bridgeurl.eval().rstrip('/'))
"""

# Callbacks compartidos para api_generate y api_poll (actualiza textura en <1ms al terminar cada frame y relanza el loop Realtime)
cb_poll = op(wc_poll.par.callbacks.eval()) if wc_poll.par.callbacks.eval() else comp.create(td.textDAT, 'api_poll_callbacks')
cb_poll.nodeX, cb_poll.nodeY = 40, -450
wc_poll.par.callbacks = cb_poll.name
wc_gen.par.callbacks = cb_poll.name
cb_poll.text = """import json

def onConnect(webClientDAT, id):
    return

def onDisconnect(webClientDAT, id):
    return

def onResponse(webClientDAT, statusCode, headerDict, data, id):
    if statusCode.get('code') != 200:
        return
    try:
        msg = json.loads(data.decode('utf-8'))
    except Exception:
        return

    comp = parent()
    metrics = op('stream_metrics')
    state = msg.get('state', msg.get('status', 'IDLE'))
    gen_id = int(msg.get('generation_id', 0))
    lat = float(msg.get('latency_ms', 0.0))
    online = float(msg.get('comfy_online', 1))

    comp.par.Status = state
    if lat > 0:
        comp.par.Latencyms = lat
        metrics.par.value2 = lat
    metrics.par.value0 = online
    metrics.par.value1 = 1.0 if state in ('GENERATING', 'QUEUED') else 0.0

    prev_id = int(comp.par.Generationid.eval())
    if gen_id > 0 and gen_id != prev_id:
        comp.par.Generationid = gen_id
        metrics.par.value3 = gen_id
        movie = op('ai_result')
        if msg.get('latest_file'):
            movie.par.file = msg['latest_file']
        else:
            movie.par.file = project.folder + '/../bridge/outputs/latest.png'
        movie.par.reloadpulse.pulse()

        log = op('generation_log')
        if log.numRows > 60:
            log.deleteRow(1)
        log.appendRow([str(gen_id), msg.get('latest_prompt', '')[:60], comp.par.Mode.eval(), msg.get('tier', ''), str(lat)])

    # Si Auto-Stream esta activo y el servidor ya esta libre (READY o IDLE), disparar el siguiente frame inmediatamente
    if comp.par.Autostream.eval() and state in ('READY', 'IDLE'):
        mod('controller_ext').trigger_generation()
"""

# Parameter Execute DAT para reaccionar a los botones Pulse del usuario en TouchDesigner
par_exec = comp.create(td.parameterexecuteDAT, 'par_callbacks')
par_exec.nodeX, par_exec.nodeY = -180, -640
par_exec.par.op = '..'
par_exec.par.pars = '*'
par_exec.par.onpulse = True
par_exec.par.valuechange = True
par_exec.text = """def onPulse(par):
    if par.name == 'Generate':
        mod('controller_ext').trigger_generation(force_img2img=False, force_rt=False)
    elif par.name == 'Sendframe':
        mod('controller_ext').trigger_generation(force_img2img=True, force_rt=False)
    elif par.name == 'Sendrtframe':
        mod('controller_ext').trigger_generation(force_img2img=True, force_rt=True)
    elif par.name == 'Openwebapp':
        mod('controller_ext').open_webapp(stream_studio=False)
    elif par.name == 'Openstreamwebapp':
        mod('controller_ext').open_webapp(stream_studio=True)
    elif par.name == 'Open3dwebapp':
        import webbrowser
        webbrowser.open('http://127.0.0.1:8082')

def onValueChange(par, prev):
    if par.name == 'Autostream' and par.eval():
        mod('controller_ext').trigger_generation()

def onValuesChanged(changes):
    return

def onExpressionChange(par, val, prev):
    return

def onExportChange(par, val, prev):
    return

def onEnableChange(par, val, prev):
    return

def onModeChange(par, val, prev):
    return
"""

# LFO + ChopExecute para hacer polling cada ~0.25 segundos sin bloquear TouchDesigner
poll_lfo = comp.create(td.lfoCHOP, 'poll_clock')
poll_lfo.nodeX, poll_lfo.nodeY = 180, -640
poll_lfo.par.wavetype = 'square'
poll_lfo.par.frequency = 4.0

poll_exec = comp.create(td.chopexecuteDAT, 'poll_runner')
poll_exec.nodeX, poll_exec.nodeY = 380, -640
poll_exec.par.chop = 'poll_clock'
poll_exec.par.offtoon = True
poll_exec.par.whileon = False
poll_exec.par.ontooff = False
poll_exec.par.whileoff = False
poll_exec.par.valuechange = False
poll_exec.text = """def onOffToOn(channel, sampleIndex, val, prev):
    try:
        mod('controller_ext').poll_bridge()
    except Exception:
        pass

def onValueChange(channel, sampleIndex, val, prev):
    return
"""

# -------------------------------------------------------------------------
# 4. CONEXIONES EXTERNAS EN /project1 Y SALIDA FINAL
# -------------------------------------------------------------------------
comp.inputConnectors[0].connect(in_null)

out_final = root.create(td.nullTOP, 'SALIDA_COMFYUI')
out_final.nodeX, out_final.nodeY = 240, 120
out_final.inputConnectors[0].connect(comp.outputConnectors[0])
out_final.viewer = True
out_final.display = True

# Comparador Lado a Lado (Entrada vs Salida IA)
comp_layout = root.create(td.layoutTOP, 'comparador_img2img')
comp_layout.nodeX, comp_layout.nodeY = 240, -80
comp_layout.par.resolutionw = 1024
comp_layout.par.resolutionh = 512
comp_layout.par.align = 'horiz'
comp_layout.inputConnectors[0].connect(in_null)
comp_layout.inputConnectors[1].connect(comp.outputConnectors[0])
comp_layout.viewer = True

# Lector de Malla 3D generada por el Modulo 3 (Trellis v2 / Hunyuan 3D -> .OBJ)
sop_3d = root.create(td.fileinSOP, 'SOP_ASSET_3D')
sop_3d.nodeX, sop_3d.nodeY = 500, 120
sop_3d.par.file.expr = "project.folder + '/../trellis3d_studio/outputs/latest_model.obj'"
sop_3d.viewer = True

# Configurar valores iniciales en Nivel Avanzado (RTX 4090 Alta Calidad) sin Auto-Stream forzado
comp.par.Mode = 'txt2img'
comp.par.Tier = 'avanzado'
comp.par.Realtimemode = 0
comp.par.Smooth60fps = 0
comp.par.Width = 1024
comp.par.Height = 1024
comp.par.Steps = 6
comp.par.Cfg = 1.0
comp.par.Denoise = 0.65
comp.par.Autostream = 0

# Guardar componente .tox y el proyecto v1.toe
comp.save(project.folder + '/ComfyUI_Studio_Bridge.tox')
project.save(project.folder + '/v1.toe')

print('<<JSON>>' + json.dumps({
    'ok': True,
    'comp': comp.path,
    'out_top': out_final.path,
    'saved': project.name
}))
'''

ok, res = live.exec_code(TD_BUILD_CODE, timeout=20.0)
print(json.dumps(res, indent=2))

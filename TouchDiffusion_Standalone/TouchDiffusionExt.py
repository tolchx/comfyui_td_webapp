from TDStoreTools import StorageManager
import TDFunctions as TDF
import numpy as np
import os
import sys
import webbrowser
import json
from datetime import datetime

# Safe fallback for CUDAMemoryShape outside of TouchDesigner
try:
    _CUDAMemoryShapeClass = CUDAMemoryShape
except NameError:
    try:
        import td
        _CUDAMemoryShapeClass = td.CUDAMemoryShape
    except Exception:
        class _CUDAMemoryShapeClass:
            def __init__(self):
                self.width = 0
                self.height = 0
                self.numComps = 0
                self.dataType = None


def _get_debug_log_path(base_dir=None):
    try:
        if not base_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "touchdiffusion_debug.log")
    except Exception:
        return None


def _write_debug_log(msg, base_dir=None):
    try:
        log_path = _get_debug_log_path(base_dir)
        if log_path:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"[{timestamp}] {msg}\n")
    except Exception:
        pass


def _resolve_and_inject_touchdiffusion_env(comp=None):
    """
    Detecta automáticamente la carpeta TouchDiffusion_Standalone y registra
    .venv/Lib/site-packages + DLLs de CUDA/TensorRT en caliente sin requerir
    reiniciar TouchDesigner.
    """
    # 1. Limpiar sys.path de librerías de otras versiones de Python (ej. Python 3.13 / 3.12 / 3.10)
    # que causan colisiones de C-extensions en Windows
    cleaned = [
        p for p in sys.path
        if not any(bad in p.replace('\\', '/') for bad in ['/Python313/', '/Python312/', '/Python310/', '/Python39/', '/Python38/'])
    ]
    sys.path[:] = cleaned

    candidates = []
    if comp is not None:
        try:
            user_val = comp.par.Venvpath.eval().strip()
            if user_val:
                candidates.append(os.path.abspath(user_val))
        except Exception:
            pass

    try:
        proj_dir = os.path.abspath(project.folder)
        candidates.extend([
            proj_dir,
            os.path.abspath(os.path.join(proj_dir, "..")),
            os.path.abspath(os.path.join(proj_dir, "TouchDiffusion_Standalone")),
            os.path.abspath(os.path.join(proj_dir, "..", "TouchDiffusion_Standalone")),
        ])
    except Exception:
        pass

    try:
        cur_file_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(cur_file_dir)
        candidates.append(os.path.abspath(os.path.join(cur_file_dir, "..")))
    except Exception:
        pass

    resolved = None
    for cand in candidates:
        if os.path.isdir(cand) and (
            os.path.isdir(os.path.join(cand, "StreamDiffusion"))
            or os.path.isdir(os.path.join(cand, ".venv"))
        ):
            resolved = cand
            break

    if not resolved and candidates:
        resolved = candidates[0]

    if resolved:
        if comp is not None:
            try:
                if comp.par.Venvpath.eval() != resolved:
                    comp.par.Venvpath = resolved
            except Exception:
                pass

        # Registrar DLLs de CUDA 11.8, TensorRT y cuDNN en Windows
        if os.name == "nt":
            dll_subdirs = [
                os.path.join(".venv", "Lib", "site-packages", "torch", "lib"),
                os.path.join(".venv", "Lib", "site-packages", "tensorrt_libs"),
                os.path.join(".venv", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
                os.path.join(".venv", "Lib", "site-packages", "nvidia", "cublas", "bin"),
                os.path.join(".venv", "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin"),
                os.path.join("embedded_python", "Lib", "site-packages", "torch", "lib"),
                os.path.join("embedded_python", "Lib", "site-packages", "tensorrt_libs"),
                os.path.join("embedded_python", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
            ]
            for dll_rel in dll_subdirs:
                dll_dir = os.path.join(resolved, dll_rel)
                if os.path.isdir(dll_dir):
                    if dll_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(dll_dir)
                        except Exception:
                            pass

        # Agregar site-packages y StreamDiffusion a sys.path en las primeras posiciones
        for sub in [
            os.path.join(resolved, ".venv", "Lib", "site-packages"),
            os.path.join(resolved, "embedded_python", "Lib", "site-packages"),
            os.path.join(resolved, "StreamDiffusion", "src"),
            os.path.join(resolved, "StreamDiffusion"),
            resolved,
        ]:
            if os.path.isdir(sub):
                while sub in sys.path:
                    sys.path.remove(sub)
                sys.path.insert(0, sub)

    # Evitar conflicto de scipy
    try:
        import diffusers.utils.import_utils as du
        du._scipy_available = False
    except Exception:
        pass

    return resolved


class TouchDiffusionExt:
    """
    TouchDiffusion Extension (Standalone Edición Clase IA)
    Soporta tanto motores TensorRT (.engine) como Modo PyTorch Directo (sin compilar).
    Diseñado para máxima estabilidad y prevención total de crashes en TouchDesigner.
    """

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.folder = _resolve_and_inject_touchdiffusion_env(ownerComp)
        _write_debug_log(f"TouchDiffusionExt init in {self.folder}", self.folder)

        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source = op("null1")
        self.stream_toparray = torch.cuda.current_stream(device=self.device) if self.device == "cuda" else None
        self.to_tensor = TopArrayInterface(self.source)

        self.stream = None
        self.rgba_tensor = None
        self.output_interface = None
        self.update_size()

        try:
            self.update_engines()
        except Exception as e:
            _write_debug_log(f"update_engines warning on init: {e}", self.folder)

    def activate_stream(self):
        self.folder = _resolve_and_inject_touchdiffusion_env(parent())
        _write_debug_log("=== activate_stream initiated ===", self.folder)
        self.fifolog("Status", "Iniciando motor de difusion...")

        # Liberar motor anterior si existía para prevenir fugas de VRAM
        if self.stream is not None:
            try:
                _write_debug_log("Liberando motor previo...", self.folder)
                del self.stream
                self.stream = None
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception as e:
                _write_debug_log(f"Error liberando motor previo: {e}", self.folder)

        try:
            from StreamDiffusion.utils.wrapper import StreamDiffusionWrapper
        except Exception as e:
            err_msg = f"No se pudo importar StreamDiffusionWrapper: {e}"
            _write_debug_log(err_msg, self.folder)
            self.fifolog("Error", err_msg)
            return

        # Si aún no hay motor seleccionado, refrescar lista automáticamente
        eng_val = parent().par.Enginelist.eval() if hasattr(parent().par, "Enginelist") else ""
        if not eng_val or "--" not in str(eng_val):
            self.update_engines()
            eng_val = parent().par.Enginelist.eval() if hasattr(parent().par, "Enginelist") else ""

        if not eng_val or "--" not in str(eng_val):
            err_msg = "No se encontro un motor configurado. Selecciona un motor en Enginelist."
            _write_debug_log(err_msg, self.folder)
            self.fifolog("Error", err_msg)
            return

        # Parsear directamente del string del motor seleccionado para evitar desfase de DATs
        vals = str(eng_val).split("--")
        checkpoint = vals[0] if len(vals) > 0 and vals[0] else "sd-turbo"
        checkpoint_type = vals[1] if len(vals) > 1 and vals[1] else "sd_1.5_turbo"
        try:
            width = int(vals[2]) if len(vals) > 2 else 512
        except Exception:
            width = 512
        try:
            height = int(vals[3]) if len(vals) > 3 else 512
        except Exception:
            height = 512

        accel_lora = vals[4] if len(vals) > 4 else "None"
        try:
            batchsize = int(vals[5]) if len(vals) > 5 else 1
        except Exception:
            batchsize = 1

        ckpt_mode = vals[7] if len(vals) > 7 and vals[7] != "None" else "img2img"
        lora_list = vals[8] if len(vals) > 8 and vals[8] != "None" else None
        ctrl_tag = vals[9] if len(vals) > 9 else "None"

        use_lcm_lora = accel_lora == "LCM"
        use_hyper_lora = accel_lora == "HyperSD"
        accel_backend = "none" if ctrl_tag == "PyTorchDirect" else "tensorrt"

        # Sincronizar parámetros visibles del COMP
        try:
            parent().par.Checkpoint = checkpoint
            parent().par.Checkpointtype = checkpoint_type
            parent().par.Sizex = width
            parent().par.Sizey = height
            parent().par.Accelerationlora = accel_lora
            parent().par.Batchsizex = batchsize
            parent().par.Batchsizey = batchsize
            parent().par.Checkpointmode = ckpt_mode
            parent().par.Loralist = lora_list or ""
            parent().par.Controlnet = ctrl_tag
        except Exception as e:
            _write_debug_log(f"Advertencia sincronizando parms: {e}", self.folder)

        # Configurar dimensiones y bufer de salida
        self.update_size(target_w=width, target_h=height)

        try:
            engine_dir = os.path.join(self.folder, "engines") if self.folder else "engines"
            _write_debug_log(
                f"Creando StreamDiffusionWrapper(checkpoint={checkpoint}, type={checkpoint_type}, "
                f"size={width}x{height}, accel={accel_backend}, engine_dir={engine_dir})...",
                self.folder,
            )

            seed_val = 50
            try:
                seed_val = int(parent().par.Seed.eval())
            except Exception:
                pass

            t_index_init = list(range(max(1, batchsize)))

            stream = StreamDiffusionWrapper(
                model_id_or_path=checkpoint,
                lora_dict=lora_list,
                t_index_list=t_index_init,
                frame_buffer_size=1,
                width=width,
                height=height,
                warmup=0,
                acceleration=accel_backend,
                mode=ckpt_mode,
                use_denoising_batch=True,
                cfg_type="self",
                seed=seed_val,
                use_lcm_lora=use_lcm_lora,
                use_hyper_lora=use_hyper_lora,
                output_type="pt",
                model_type=checkpoint_type,
                touchdiffusion=True,
                engine_dir=engine_dir,
            )

            prompt_val = parent().par.Prompt.eval() if hasattr(parent().par, "Prompt") else "cat"
            neg_val = parent().par.Negprompt.eval() if hasattr(parent().par, "Negprompt") else ""
            cfg_val = float(parent().par.Cfgscale.eval()) if hasattr(parent().par, "Cfgscale") else 1.0
            delta_val = float(parent().par.Deltamult.eval()) if hasattr(parent().par, "Deltamult") else 1.0

            # Calcular t_index_list según denoise
            denoise_val = float(parent().par.Denoise.eval()) if hasattr(parent().par, "Denoise") else 0.98
            mode_val = "linear"
            if hasattr(parent().par, "Denoisemode"):
                mode_val = str(parent().par.Denoisemode.eval())

            try:
                t_idx = stream.touchdiffusion_generate_t_index_list(denoise_val, mode_val)
            except Exception:
                t_idx = [0]

            _write_debug_log(f"Preparando pipeline (prompt='{prompt_val}', t_idx={t_idx})...", self.folder)
            stream.prepare(
                prompt=prompt_val,
                negative_prompt=neg_val,
                guidance_scale=cfg_val,
                delta=delta_val,
                t_index_list=t_idx,
            )

            # Asignar solo tras inicialización completa
            self.stream = stream

            try:
                menuindex = parent().par.Enginelist.menuIndex
                menulabel = parent().par.Enginelist.menuLabels[menuindex]
                parent().par.Activeengine = menulabel
            except Exception:
                parent().par.Activeengine = "Active"

            _write_debug_log("=== Motor cargado y listo exitosamente ===", self.folder)
            self.fifolog("Status", "Motor activado y listo para difusion en tiempo real")
        except Exception as e:
            parent().par.Activeengine = "None"
            self.stream = None
            err_msg = f"Error al cargar motor: {e}"
            _write_debug_log(err_msg, self.folder)
            self.fifolog("Error", err_msg)

    def generate(self, scriptOp):
        stream = self.stream
        if stream is None or not hasattr(stream, "batch_size"):
            return

        if self.output_interface is None:
            self.update_size()

        target_w = self.output_interface.mem_shape.width
        target_h = self.output_interface.mem_shape.height

        # Capturar o construir el tensor de entrada GPU
        image_tensor = None
        try:
            if self.to_tensor is not None and self.stream_toparray is not None:
                self.to_tensor.update(self.stream_toparray.cuda_stream)
                if self.to_tensor.is_valid():
                    raw_image = self.torch.as_tensor(self.to_tensor, device=self.device)
                    image_tensor = self.preprocess_image(raw_image, target_w, target_h)
        except Exception as e:
            pass

        # Fallback ultra-seguro si la entrada aún no tiene señal o no ha cocinado
        if image_tensor is None:
            image_tensor = self.torch.zeros((1, 3, target_h, target_w), dtype=self.torch.float32, device=self.device)

        try:
            last_element = 1 if stream.batch_size != 1 else 0
            for _ in range(max(1, stream.batch_size - last_element)):
                output_image = stream(image=image_tensor)

            output_tensor = self.postprocess_image(output_image)

            # Copiar a la textura de script1 mediante CUDA
            if output_tensor is not None:
                scriptOp.copyCUDAMemory(
                    output_tensor.data_ptr(),
                    self.output_interface.size,
                    self.output_interface.mem_shape,
                )
        except Exception as e:
            self.fifolog("Error", f"generate: {e}")

    def update_size(self, target_w=None, target_h=None):
        if target_w is None:
            try:
                target_w = max(64, int(parent().par.Sizex.eval() or 512))
            except Exception:
                target_w = 512
        if target_h is None:
            try:
                target_h = max(64, int(parent().par.Sizey.eval() or 512))
            except Exception:
                target_h = 512

        self.rgba_tensor = self.torch.zeros((target_h, target_w, 4), dtype=self.torch.float32, device=self.device)
        self.rgba_tensor[..., 3] = 1.0  # Canal Alpha en 1.0 completamente visible
        self.output_interface = TopCUDAInterface(target_w, target_h, 4, np.float32)

        s1 = op("script1")
        if s1:
            try:
                s1.par.outputresolution = "custom"
                s1.par.resolutionw = target_w
                s1.par.resolutionh = target_h
                s1.par.format = "rgba32float"
            except Exception:
                pass

    def preprocess_image(self, image, target_w, target_h):
        # image shape es (C, H, W)
        image = self.torch.flip(image, [1])
        image = self.torch.clamp(image, 0.0, 1.0)
        image = image[:3, :, :]
        image = image.unsqueeze(0).to(dtype=self.torch.float32)
        if image.shape[-2] != target_h or image.shape[-1] != target_w:
            image = self.torch.nn.functional.interpolate(
                image,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        return image

    def postprocess_image(self, image):
        if image.dim() == 4:
            image = image[0]
        image = self.torch.flip(image, [1])
        image = image.permute(1, 2, 0).to(dtype=self.torch.float32)
        self.rgba_tensor[..., :3] = image
        self.rgba_tensor[..., 3] = 1.0
        return self.rgba_tensor

    def update_engines(self):
        folder = _resolve_and_inject_touchdiffusion_env(parent())
        menuNames = []
        menuLabels = []

        engines_root = os.path.join(folder, "engines") if folder else "engines"
        if os.path.isdir(engines_root):
            for root, dirs, files in os.walk(engines_root):
                if "unet.engine" in files:
                    folder_name = os.path.basename(root)
                    split_folder_name = folder_name.split("--")
                    if len(split_folder_name) >= 10:
                        name = "-".join([
                            split_folder_name[0],
                            split_folder_name[2],
                            split_folder_name[3],
                            split_folder_name[5],
                        ])
                        menuLabels.append(f"[TensorRT] {name}")
                        menuNames.append(folder_name)

        # Agregar siempre modo PyTorch Directo (sin necesidad de compilar .engine)
        direct_turbo = "sd-turbo--sd_1.5_turbo--512--512--None--1--1--img2img--None--PyTorchDirect"
        menuNames.append(direct_turbo)
        menuLabels.append("[PyTorch-Direct] sd-turbo-512-512-1 (Sin compilar TensorRT)")

        parent().par.Enginelist.menuNames = menuNames
        parent().par.Enginelist.menuLabels = menuLabels
        if menuNames and (not parent().par.Enginelist.eval() or parent().par.Enginelist.eval() not in menuNames):
            parent().par.Enginelist = menuNames[0]
        self.update_selected_engine()

    def update_selected_engine(self):
        try:
            eng_str = parent().par.Enginelist.eval()
            if not eng_str or "--" not in str(eng_str):
                return
            vals = str(eng_str).split("--")
            parent().par.Checkpoint = vals[0]
            parent().par.Checkpointtype = vals[1]
            parent().par.Sizex = int(vals[2])
            parent().par.Sizey = int(vals[3])
            parent().par.Accelerationlora = vals[4]
            parent().par.Batchsizex = int(vals[5])
            parent().par.Batchsizey = int(vals[6])
            parent().par.Checkpointmode = vals[7]
            parent().par.Loralist = vals[8] if vals[8] != "None" else ""
            parent().par.Controlnet = vals[9]
        except Exception:
            pass

    def update_prompt(self):
        if self.stream is not None and hasattr(self.stream, "touchdiffusion_prompt"):
            prompt = parent().par.Prompt.eval()
            self.stream.touchdiffusion_prompt(prompt)

    def update_denoising_strength(self):
        if self.stream is not None and hasattr(self.stream, "touchdiffusion_generate_t_index_list"):
            amount = float(parent().par.Denoise.eval())
            mode = "linear"
            if hasattr(parent().par, "Denoisemode"):
                mode = str(parent().par.Denoisemode.eval())
            return self.stream.touchdiffusion_generate_t_index_list(amount, mode)
        return [0]

    def update_cfg_setting(self):
        if self.stream is not None and hasattr(self.stream, "touchdiffusion_update_cfg_setting"):
            guidance_scale = float(parent().par.Cfgscale.eval())
            delta = float(parent().par.Deltamult.eval())
            self.stream.touchdiffusion_update_cfg_setting(guidance_scale=guidance_scale, delta=delta)

    def update_noise(self):
        if self.stream is not None and hasattr(self.stream, "touchdiffusion_update_noise"):
            seed = int(parent().par.Seed.eval())
            self.stream.touchdiffusion_update_noise(seed=seed)

    def parexec_onValueChange(self, par, prev):
        if self.stream is not None and hasattr(self.stream, "batch_size"):
            if par.name == "Prompt":
                self.update_prompt()
            elif par.name == "Denoise":
                self.update_denoising_strength()
            elif par.name == "Cfgscale" or par.name == "Deltamult":
                self.update_cfg_setting()
            elif par.name == "Seed":
                self.update_noise()
        if par.name == "Enginelist":
            self.update_selected_engine()

    def parexec_onPulse(self, par):
        if par.name == "Loadengine":
            self.activate_stream()
        elif par.name == "Refreshenginelist":
            self.update_engines()
        elif par.name.startswith("Url"):
            self.about(par.name)

    def fifolog(self, status, message):
        try:
            current_time = datetime.now()
            formated_time = current_time.strftime("%H:%M:%S")
            fifo = op("fifo1")
            if fifo:
                fifo.appendRow([formated_time, str(status), str(message)])
        except Exception:
            pass

    def about(self, endpoint):
        urls = {
            "Urlg": "https://github.com/olegchomp/TouchDiffusion",
            "Urld": "https://discord.gg/wNW8xkEjrf",
            "Urlt": "https://www.youtube.com/vjschool",
            "Urla": "https://olegcho.mp/",
            "Urldonate": "https://boosty.to/vjschool/",
        }
        if endpoint in urls:
            webbrowser.open(urls[endpoint], new=2)


class TopCUDAInterface:
    def __init__(self, width, height, num_comps, dtype):
        self.mem_shape = _CUDAMemoryShapeClass()
        self.mem_shape.width = width
        self.mem_shape.height = height
        self.mem_shape.numComps = num_comps
        self.mem_shape.dataType = dtype
        self.bytes_per_comp = np.dtype(dtype).itemsize
        self.size = width * height * num_comps * self.bytes_per_comp


class TopArrayInterface:
    def __init__(self, top, stream=0):
        self.top = top
        self._cuda_dict = None
        self.update(stream)

    def is_valid(self):
        try:
            return (
                self._cuda_dict is not None
                and self._cuda_dict.get("data", (0,))[0] not in (0, None)
                and self._cuda_dict.get("shape", (0, 0, 0))[1] > 0
                and self._cuda_dict.get("shape", (0, 0, 0))[2] > 0
            )
        except Exception:
            return False

    @property
    def __cuda_array_interface__(self):
        return self._cuda_dict

    def update(self, stream=0):
        try:
            if not self.top or getattr(self.top, "width", 0) <= 0 or getattr(self.top, "height", 0) <= 0:
                self._cuda_dict = None
                return
            mem = self.top.cudaMemory(stream=stream)
            if not mem or not getattr(mem, "ptr", None) or mem.shape.width <= 0 or mem.shape.height <= 0:
                self._cuda_dict = None
                return

            w = mem.shape.width
            h = mem.shape.height
            num_comps = mem.shape.numComps
            num_bytes = 4  # float32
            num_bytes_px = num_bytes * num_comps
            shape = (num_comps, h, w)
            strides = (num_bytes, num_bytes_px * w, num_bytes_px)

            self._cuda_dict = {
                "version": 3,
                "shape": shape,
                "typestr": "<f4",
                "descr": [("", "<f4")],
                "stream": stream,
                "strides": strides,
                "data": (mem.ptr, False),
            }
        except Exception:
            self._cuda_dict = None

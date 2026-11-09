from TDStoreTools import StorageManager
import TDFunctions as TDF
import numpy as np
import os
import sys
import webbrowser
import json
from datetime import datetime


def _resolve_and_inject_touchdiffusion_env(comp):
    """
    Detecta automáticamente la carpeta TouchDiffusion_Standalone y registra
    .venv/Lib/site-packages + DLLs de CUDA/TensorRT en caliente sin requerir
    reiniciar TouchDesigner.
    """
    candidates = []
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
        try:
            if comp.par.Venvpath.eval() != resolved:
                comp.par.Venvpath = resolved
        except Exception:
            pass

        # Registrar DLLs de CUDA 11.8 y TensorRT en Windows
        if os.name == "nt":
            for dll_rel in [
                os.path.join(".venv", "Lib", "site-packages", "torch", "lib"),
                os.path.join(".venv", "Lib", "site-packages", "tensorrt_libs"),
                os.path.join("embedded_python", "Lib", "site-packages", "torch", "lib"),
                os.path.join("embedded_python", "Lib", "site-packages", "tensorrt_libs"),
            ]:
                dll_dir = os.path.join(resolved, dll_rel)
                if os.path.isdir(dll_dir):
                    if dll_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(dll_dir)
                        except Exception:
                            pass

        # Agregar site-packages y StreamDiffusion a sys.path
        for sub in [
            os.path.join(resolved, ".venv", "Lib", "site-packages"),
            os.path.join(resolved, "embedded_python", "Lib", "site-packages"),
            os.path.join(resolved, "StreamDiffusion", "src"),
            os.path.join(resolved, "StreamDiffusion"),
            resolved,
        ]:
            if os.path.isdir(sub) and sub not in sys.path:
                sys.path.insert(0, sub)

    return resolved


class TouchDiffusionExt:
    """
    TouchDiffusion Extension (Standalone Edición Clase IA)
    Soporta tanto motores TensorRT (.engine) como Modo PyTorch Directo (sin compilar).
    """

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        _resolve_and_inject_touchdiffusion_env(ownerComp)

        import torch
        self.torch = torch
        self.source = op("null1")
        self.device = "cuda"
        self.to_tensor = TopArrayInterface(self.source)
        self.stream_toparray = torch.cuda.current_stream(device=self.device)
        self.rgba_tensor = torch.zeros((512, 512, 4), dtype=torch.float32).to(self.device)
        self.rgba_tensor[..., 3] = 0
        self.output_interface = TopCUDAInterface(512, 512, 4, np.float32)
        self.stream = None
        try:
            self.update_engines()
        except Exception:
            pass

    def activate_stream(self):
        folder = _resolve_and_inject_touchdiffusion_env(parent())
        try:
            from StreamDiffusion.utils.wrapper import StreamDiffusionWrapper
        except Exception as e:
            self.fifolog("Error", f"No se pudo importar StreamDiffusionWrapper: {e}")
            return

        # Si aún no hay motor seleccionado, refrescar lista automáticamente
        if not parent().par.Checkpoint.eval():
            self.update_engines()

        self.update_size()

        acceleration_lora = op("parameter1")["Accelerationlora", 1].val
        use_lcm_lora = acceleration_lora == "LCM"
        use_hyper_lora = acceleration_lora == "HyperSD"

        ctrl_tag = op("parameter1")["Controlnet", 1].val
        accel_backend = "none" if ctrl_tag == "PyTorchDirect" else "tensorrt"

        try:
            width = max(64, int(op("parameter1")["Sizex", 1] or 512))
            height = max(64, int(op("parameter1")["Sizey", 1] or 512))
            engine_dir = os.path.join(folder, "engines") if folder else "engines"

            self.stream = StreamDiffusionWrapper(
                model_id_or_path=f"{op('parameter1')['Checkpoint', 1].val}",
                lora_dict=op("parameter1")["Loralist", 1].val,
                t_index_list=self.generate_t_index_list(),
                frame_buffer_size=1,
                width=width,
                height=height,
                warmup=0,
                acceleration=accel_backend,
                mode=op("parameter1")["Checkpointmode", 1].val or "img2img",
                use_denoising_batch=True,
                cfg_type="self",
                seed=int(op("parameter1")["Seed", 1] or 50),
                use_lcm_lora=use_lcm_lora,
                use_hyper_lora=use_hyper_lora,
                output_type="pt",
                model_type=op("parameter1")["Checkpointtype", 1].val or "sd_1.5_turbo",
                touchdiffusion=True,
                engine_dir=engine_dir,
            )

            self.stream.prepare(
                prompt=parent().par.Prompt.eval(),
                negative_prompt=parent().par.Negprompt.eval(),
                guidance_scale=parent().par.Cfgscale.eval(),
                delta=parent().par.Deltamult.eval(),
                t_index_list=self.update_denoising_strength(),
            )

            try:
                menuindex = parent().par.Enginelist.menuIndex
                menulabel = parent().par.Enginelist.menuLabels[menuindex]
                parent().par.Activeengine = menulabel
            except Exception:
                parent().par.Activeengine = "Active"
            self.fifolog("Status", "Engine activated")
        except Exception as e:
            parent().par.Activeengine = "None"
            self.fifolog("Error", str(e))

    def generate(self, scriptOp):
        stream = self.stream
        if stream is None or not hasattr(stream, "batch_size"):
            return
        self.to_tensor.update(self.stream_toparray.cuda_stream)
        image = self.torch.as_tensor(self.to_tensor, device=self.device)
        image_tensor = self.preprocess_image(image)

        last_element = 1 if stream.batch_size != 1 else 0
        for _ in range(max(1, stream.batch_size - last_element)):
            output_image = stream(image=image_tensor)

        output_tensor = self.postprocess_image(output_image)
        scriptOp.copyCUDAMemory(
            output_tensor.data_ptr(),
            self.output_interface.size,
            self.output_interface.mem_shape,
        )

    def update_size(self):
        width = max(64, int(op("parameter1")["Sizex", 1] or 512))
        height = max(64, int(op("parameter1")["Sizey", 1] or 512))
        self.rgba_tensor = self.torch.zeros((height, width, 4), dtype=self.torch.float32).to(self.device)
        self.rgba_tensor[..., 3] = 0
        self.output_interface = TopCUDAInterface(width, height, 4, np.float32)

    def preprocess_image(self, image):
        image = self.torch.flip(image, [1])
        image = self.torch.clamp(image, 0, 1)
        image = image[:3, :, :]
        image = image.unsqueeze(0)
        return image

    def postprocess_image(self, image):
        image = self.torch.flip(image, [1])
        image = image.permute(1, 2, 0)
        self.rgba_tensor[..., :3] = image
        return self.rgba_tensor

    def acceleration_mode(self):
        turbo = False
        lcm = False
        acceleration_mode = parent().par.Acceleration.val
        if acceleration_mode == "LCM":
            lcm = True
        if acceleration_mode == "sd_turbo":
            turbo = True
        return lcm, turbo

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
            vals = parent().par.Enginelist.eval().split("--")
            parent().par.Checkpoint = vals[0]
            parent().par.Checkpointtype = vals[1]
            parent().par.Accelerationlora = vals[4]
            parent().par.Checkpointmode = vals[7]
            parent().par.Controlnet = vals[9]
            parent().par.Loralist = vals[8]
            parent().par.Sizex = int(vals[2])
            parent().par.Sizey = int(vals[3])
            parent().par.Batchsizex = int(vals[5])
            parent().par.Batchsizey = int(vals[6])
        except Exception:
            parent().par.Checkpoint, parent().par.Checkpointtype, parent().par.Accelerationlora = "", "", ""
            parent().par.Checkpointmode, parent().par.Controlnet, parent().par.Loralist = "", "", ""
            parent().par.Sizex, parent().par.Sizey, parent().par.Batchsizex, parent().par.Batchsizey = 512, 512, 1, 1

    def update_prompt(self):
        prompt = parent().par.Prompt.eval()
        self.stream.touchdiffusion_prompt(prompt)

    def prompt_to_str(self):
        prompt_list = []
        seq = parent().seq.Promptblock
        enable_weights = parent().par.Enableweight
        for block in seq.blocks:
            if block.par.Weight.val > 0:
                if enable_weights:
                    prompt_with_weight = f"({block.par.Prompt.val}){block.par.Weight.val}"
                else:
                    prompt_with_weight = block.par.Prompt.val
                prompt_list.append(prompt_with_weight)
        return ", ".join(prompt_list)

    def update_scheduler(self):
        t_index_list = []
        seq = parent().seq.Schedulerblock
        for block in seq.blocks:
            t_index_list.append(block.par.Step)
        self.stream.touchdiffusion_scheduler(t_index_list)

    def update_denoising_strength(self):
        amount = parent().par.Denoise.eval()
        mode = parent().par.Denoisemode.eval()
        t_index_list = self.stream.touchdiffusion_generate_t_index_list(amount, mode)
        return t_index_list

    def generate_t_index_list(self):
        batchsize = max(1, int(op("parameter1")["Batchsizex", 1] or 1))
        return list(range(batchsize))

    def update_cfg_setting(self):
        guidance_scale = parent().par.Cfgscale.eval()
        delta = parent().par.Deltamult.eval()
        self.stream.touchdiffusion_update_cfg_setting(guidance_scale=guidance_scale, delta=delta)

    def update_noise(self):
        seed = parent().par.Seed.eval()
        self.stream.touchdiffusion_update_noise(seed=seed)

    def parexec_onValueChange(self, par, prev):
        if hasattr(self.stream, "batch_size"):
            if par.name == "Prompt":
                self.update_prompt()
            elif par.name == "Denoise":
                self.update_denoising_strength()
            elif par.name == "Cfgscale":
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
        elif par.name[0:3] == "Url":
            self.about(par.name)

    def fifolog(self, status, message):
        current_time = datetime.now()
        formated_time = current_time.strftime("%H:%M:%S")
        op("fifo1").appendRow([formated_time, str(status), str(message)])

    def about(self, endpoint):
        if endpoint == "Urlg":
            webbrowser.open("https://github.com/olegchomp/TouchDiffusion", new=2)
        if endpoint == "Urld":
            webbrowser.open("https://discord.gg/wNW8xkEjrf", new=2)
        if endpoint == "Urlt":
            webbrowser.open("https://www.youtube.com/vjschool", new=2)
        if endpoint == "Urla":
            webbrowser.open("https://olegcho.mp/", new=2)
        if endpoint == "Urldonate":
            webbrowser.open("https://boosty.to/vjschool/", new=2)


class TopCUDAInterface:
    def __init__(self, width, height, num_comps, dtype):
        self.mem_shape = CUDAMemoryShape()
        self.mem_shape.width = width
        self.mem_shape.height = height
        self.mem_shape.numComps = num_comps
        self.mem_shape.dataType = dtype
        self.bytes_per_comp = np.dtype(dtype).itemsize
        self.size = width * height * num_comps * self.bytes_per_comp


class TopArrayInterface:
    def __init__(self, top, stream=0):
        self.top = top
        mem = top.cudaMemory(stream=stream)
        self.w, self.h = mem.shape.width, mem.shape.height
        self.num_comps = mem.shape.numComps
        self.dtype = mem.shape.dataType
        shape = (mem.shape.numComps, self.h, self.w)
        dtype_info = {"descr": [("", "<f4")], "num_bytes": 4}
        dtype_descr = dtype_info["descr"]
        num_bytes = dtype_info["num_bytes"]
        num_bytes_px = num_bytes * mem.shape.numComps

        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": dtype_descr[0][1],
            "descr": dtype_descr,
            "stream": stream,
            "strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
            "data": (mem.ptr, False),
        }

    def update(self, stream=0):
        mem = self.top.cudaMemory(stream=stream)
        self.__cuda_array_interface__["stream"] = stream
        self.__cuda_array_interface__["data"] = (mem.ptr, False)
        return

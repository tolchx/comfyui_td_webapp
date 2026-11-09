import gc
import os
import sys
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

module_path = os.path.abspath(__file__)
touchdiffusion_path = os.path.abspath(os.path.join(module_path, "../../../"))

# Registrar automáticamente los directorios de DLLs de CUDA y TensorRT del .venv
# para que los alumnos NO necesiten instalar CUDA Toolkit 11.8 en Windows ni reiniciar la PC.
if os.name == "nt":
    _dll_candidates = [
        os.path.join(touchdiffusion_path, ".venv", "Lib", "site-packages", "torch", "lib"),
        os.path.join(touchdiffusion_path, ".venv", "Lib", "site-packages", "tensorrt_libs"),
        os.path.join(touchdiffusion_path, "embedded_python", "Lib", "site-packages", "torch", "lib"),
        os.path.join(touchdiffusion_path, "embedded_python", "Lib", "site-packages", "tensorrt_libs"),
    ]
    for _dll_dir in _dll_candidates:
        if os.path.isdir(_dll_dir):
            if _dll_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_dll_dir)
                except Exception:
                    pass

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pt",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = False,
        use_hyper_lora: bool = False,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = os.path.join(touchdiffusion_path, "engines"),
        touchdiffusion: bool = False,
        model_type: str = 'None'
    ):
        self.sd_turbo = False

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker
        self.local_files_only = bool(touchdiffusion)
        self.model_type = model_type

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_hyper_lora=use_hyper_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            touchdiffusion=touchdiffusion,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def touchdiffusion_prompt(self, prompt):
        self.stream.update_prompt(prompt)

    def touchdiffusion_scheduler(self, t_index_list):
        self.stream.update_scheduler(t_index_list=t_index_list)

    def touchdiffusion_generate_t_index_list(self, noise_strength, mode):
        t_index_list = self.stream.generate_t_index_list(noise_strength=float(noise_strength), mode=str(mode))
        self.stream.update_scheduler(t_index_list=t_index_list)
        return t_index_list

    def touchdiffusion_update_cfg_setting(self, guidance_scale, delta):
        self.stream.update_cfg_setting(guidance_scale=float(guidance_scale), delta=float(delta))

    def touchdiffusion_update_noise(self, seed):
        self.stream.init_generator(int(seed))
        self.stream.update_noise()

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        t_index_list: Optional[List[int]] = None,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            t_index_list=t_index_list,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)
        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        return image

    def preprocess_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return self.stream.image_processor.preprocess(
                image, self.height, self.width
            ).to(device=self.device, dtype=self.dtype)
        else:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB").resize((self.width, self.height))
            if isinstance(image, Image.Image):
                image = image.convert("RGB").resize((self.width, self.height))

            return self.stream.image_processor.preprocess(
                image, self.height, self.width
            ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pt"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor, output_type=output_type)
        else:
            return postprocess_image(image_tensor, output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = False,
        use_hyper_lora: bool = False,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        touchdiffusion: bool = False,
    ) -> StreamDiffusion:
        pipe = None
        ckpt_dir = os.path.join(touchdiffusion_path, "models", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        if self.model_type == "sd_1.5":
            ckpt_file = os.path.join(ckpt_dir, f"{model_id_or_path}.safetensors")
            for local_only in ([True, False] if self.local_files_only else [False]):
                try:
                    pipe = StableDiffusionPipeline.from_single_file(
                        ckpt_file,
                        cache_dir=ckpt_dir,
                        use_safetensors=True,
                        local_files_only=local_only,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        add_watermarker=False,
                        safety_checker=None,
                    ).to(device=self.device, dtype=self.dtype)
                    break
                except Exception as e:
                    last_err = e
        else:
            # Default a sd_1.5_turbo ('stabilityai/sd-turbo')
            for local_only in ([True, False] if self.local_files_only else [False]):
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "stabilityai/sd-turbo",
                        cache_dir=ckpt_dir,
                        use_safetensors=True,
                        local_files_only=local_only,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        add_watermarker=False,
                        safety_checker=None,
                    ).to(device=self.device, dtype=self.dtype)
                    break
                except Exception as e:
                    last_err = e

        if pipe is None:
            raise RuntimeError(f"Could not load model '{model_id_or_path}' ({self.model_type}): {last_err}")

        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        if not self.sd_turbo:
            lora_cache_dir = os.path.join(touchdiffusion_path, "models", "acceleration_loras")
            os.makedirs(lora_cache_dir, exist_ok=True)
            if use_lcm_lora:
                target_lcm = lcm_lora_id or "latent-consistency/lcm-lora-sdv1-5"
                for local_only in ([True, False] if self.local_files_only else [False]):
                    try:
                        stream.load_lcm_lora(
                            pretrained_model_name_or_path_or_dict=target_lcm,
                            cache_dir=lora_cache_dir,
                            local_files_only=local_only,
                        )
                        break
                    except Exception:
                        pass
                stream.fuse_lora()
            elif use_hyper_lora:
                for local_only in ([True, False] if self.local_files_only else [False]):
                    try:
                        stream.load_HyperSD_lora(
                            cache_dir=lora_cache_dir,
                            local_files_only=local_only,
                        )
                        break
                    except Exception:
                        pass
                stream.fuse_lora()

            if isinstance(lora_dict, dict):
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)

        if use_tiny_vae:
            vae_cache_dir = os.path.join(touchdiffusion_path, "models", "vae")
            os.makedirs(vae_cache_dir, exist_ok=True)
            target_vae = vae_id or "madebyollin/taesd"
            loaded_vae = None
            for local_only in ([True, False] if self.local_files_only else [False]):
                try:
                    loaded_vae = AutoencoderTiny.from_pretrained(
                        target_vae,
                        cache_dir=vae_cache_dir,
                        local_files_only=local_only,
                    ).to(device=pipe.device, dtype=pipe.dtype)
                    break
                except Exception:
                    pass
            if loaded_vae is not None:
                stream.vae = loaded_vae

        if acceleration == "tensorrt":
            def create_prefix(
                model_id_or_path: str,
                max_batch_size: int,
                min_batch_size: int,
            ):
                if use_lcm_lora:
                    acceleration_mode = "LCM"
                elif use_hyper_lora:
                    acceleration_mode = "HyperSD"
                else:
                    acceleration_mode = "None"

                maybe_path = Path(os.path.join(touchdiffusion_path, model_id_or_path))
                stem = maybe_path.stem if maybe_path.exists() else model_id_or_path
                return f"{stem}--{self.model_type}--{self.width}--{self.height}--{acceleration_mode}--{max_batch_size}--{min_batch_size}--{self.mode}--None--None"

            if not os.path.isabs(str(engine_dir)):
                engine_dir = os.path.join(touchdiffusion_path, str(engine_dir))
            engine_dir = Path(engine_dir)

            prefix_unet = create_prefix(
                model_id_or_path=model_id_or_path,
                max_batch_size=stream.trt_unet_batch_size,
                min_batch_size=stream.trt_unet_batch_size,
            )
            prefix_vae = create_prefix(
                model_id_or_path=model_id_or_path,
                max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
            )
            unet_path = os.path.join(engine_dir, prefix_unet, "unet.engine")
            vae_encoder_path = os.path.join(engine_dir, prefix_vae, "vae_encoder.engine")
            vae_decoder_path = os.path.join(engine_dir, prefix_vae, "vae_decoder.engine")

            if touchdiffusion and not os.path.exists(unet_path):
                print(f"[TouchDiffusion] Engine '{unet_path}' not found; running in PyTorch Direct FP16 + TinyVAE mode.")
            else:
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                if not touchdiffusion:
                    if not os.path.exists(unet_path):
                        os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                        unet_model = UNet(
                            fp16=True,
                            device=stream.device,
                            max_batch_size=stream.trt_unet_batch_size,
                            min_batch_size=stream.trt_unet_batch_size,
                            embedding_dim=stream.text_encoder.config.hidden_size,
                            unet_dim=stream.unet.config.in_channels,
                        )
                        compile_unet(
                            stream.unet,
                            unet_model,
                            unet_path + ".onnx",
                            unet_path + ".opt.onnx",
                            unet_path,
                            self.height,
                            self.width,
                            opt_batch_size=stream.trt_unet_batch_size,
                        )

                    if not os.path.exists(vae_decoder_path):
                        os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                        stream.vae.forward = stream.vae.decode
                        vae_decoder_model = VAE(
                            device=stream.device,
                            max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                            min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        )
                        compile_vae_decoder(
                            stream.vae,
                            vae_decoder_model,
                            vae_decoder_path + ".onnx",
                            vae_decoder_path + ".opt.onnx",
                            vae_decoder_path,
                            self.height,
                            self.width,
                            opt_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        )
                        delattr(stream.vae, "forward")

                    if not os.path.exists(vae_encoder_path):
                        os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                        vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                        vae_encoder_model = VAEEncoder(
                            device=stream.device,
                            max_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                            min_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        )
                        compile_vae_encoder(
                            vae_encoder,
                            vae_encoder_model,
                            vae_encoder_path + ".onnx",
                            vae_encoder_path + ".opt.onnx",
                            vae_encoder_path,
                            self.height,
                            self.width,
                            opt_batch_size=self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                        )

                cuda_stream = cuda.Stream()
                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_stream, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()
                print("TensorRT acceleration enabled.")
        else:
            print("PyTorch Direct FP16 + TinyVAE acceleration enabled (No TensorRT engine build needed).")

        if seed < 0:
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.Generator(),
            seed=seed,
        )

        return stream

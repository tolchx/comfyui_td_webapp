import os
import sys
import time
import subprocess
import platform
from pathlib import Path
import requests
import json
import webbrowser
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
touchdiffusion_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(current_dir, "src")

for p in [current_dir, src_dir, touchdiffusion_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Registrar DLLs de CUDA 11.8 y TensorRT incluidas en .venv
if os.name == "nt":
    for dll_sub in [
        os.path.join(touchdiffusion_root, ".venv", "Lib", "site-packages", "torch", "lib"),
        os.path.join(touchdiffusion_root, ".venv", "Lib", "site-packages", "tensorrt_libs"),
    ]:
        if os.path.isdir(dll_sub):
            if dll_sub not in os.environ.get("PATH", ""):
                os.environ["PATH"] = dll_sub + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dll_sub)
                except Exception:
                    pass

import gradio as gr

try:
    from utils.wrapper import StreamDiffusionWrapper
except ImportError as e:
    print(f"Dependencies not installed yet: {e}")


def ensure_directories():
    directories = [
        "models",
        "models/acceleration_loras",
        "models/checkpoints",
        "models/loras",
        "models/vae",
        "engines",
    ]
    for directory in directories:
        full_dir = os.path.join(touchdiffusion_root, directory)
        os.makedirs(full_dir, exist_ok=True)


def list_files_in_folder(folder_rel_path):
    ensure_directories()
    folder_path = os.path.join(touchdiffusion_root, folder_rel_path)
    if not os.path.isdir(folder_path):
        return []
    return [
        file
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file)) and file.lower().endswith((".safetensors", ".ckpt"))
    ]


def stream_engine(width, height, steps, acceleration, model_id_or_path, selected_model_type):
    if not selected_model_type:
        selected_model_type = "sd_1.5_turbo"
    if not model_id_or_path:
        model_id_or_path = "SD Turbo" if selected_model_type == "sd_1.5_turbo" else ""

    use_lcm_lora = acceleration == "LCM"
    use_hyper_lora = acceleration == "HyperSD"

    t_index_list = list(range(int(steps)))
    engine_dir = os.path.join(touchdiffusion_root, "engines")
    model_id_or_path = os.path.splitext(str(model_id_or_path))[0]

    if model_id_or_path == "SD Turbo" or selected_model_type == "sd_1.5_turbo":
        model_id_or_path = "sd-turbo"

    print(f"[TouchDiffusion WebUI] Building TensorRT Engine: model={model_id_or_path} ({selected_model_type}), {int(width)}x{int(height)}, steps={int(steps)}, lora={acceleration}")

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=None,
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=int(width),
        height=int(height),
        warmup=0,
        acceleration="tensorrt",
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="self",
        seed=2,
        use_lcm_lora=use_lcm_lora,
        use_hyper_lora=use_hyper_lora,
        touchdiffusion=False,
        engine_dir=engine_dir,
        model_type=selected_model_type,
    )

    stream.prepare(
        prompt="1girl with brown dog hair, thick glasses, smiling",
        negative_prompt="low quality, bad quality, blurry, low resolution",
        num_inference_steps=50,
        guidance_scale=1.2,
        delta=0.5,
        t_index_list=t_index_list,
    )

    input_img = os.path.join(current_dir, "images", "inputs", "input.png")
    image_tensor = stream.preprocess_image(input_img)
    fps = 0
    for _ in range(10):
        start_time = time.time()
        last_element = 1 if stream.batch_size != 1 else 0
        for _ in range(stream.batch_size - last_element):
            stream(image=image_tensor)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time

    return (
        f"Model: {model_id_or_path}\n"
        f"WxH: {int(width)}x{int(height)}\n"
        f"Batch size: {int(steps)}\n"
        f"Expected: {int(fps)} FPS\n"
        f"Status: Ready (Podés hacer clic en 'Refresh engines' y 'Load engine' dentro de TouchDesigner)"
    )


def is_installed(package_name):
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def update_acceleration_lora(amount_steps, selected_model_type):
    if selected_model_type != "sd_1.5_turbo" and selected_model_type is not None:
        if int(amount_steps) in [1, 2, 4, 8]:
            return ["None", "LCM", "HyperSD"]
        else:
            return ["None", "LCM"]
    return ["None"]


def update_interactivity(selected_value, amount_steps):
    current_models = list_files_in_folder("models/checkpoints")
    if selected_value == "sd_1.5_turbo":
        return (
            gr.Dropdown(choices=["SD Turbo"], value="SD Turbo", allow_custom_value=True, interactive=False),
            gr.Radio(["None"], value="None", label="Acceleration Lora not available for this model"),
        )
    else:
        value = current_models[0] if len(current_models) > 0 else ""
        lora_list = update_acceleration_lora(amount_steps, selected_value)
        return (
            gr.Dropdown(choices=current_models, value=value, allow_custom_value=True, interactive=True),
            gr.Radio(lora_list, value="None", label="Add Acceleration Lora"),
        )


def update_hypersd(amount_steps, selected_model_type):
    lora_list = update_acceleration_lora(amount_steps, selected_model_type)
    return gr.Radio(lora_list, value="None", label="Add Acceleration Lora")


def inst_upd():
    setup_script = os.path.join(touchdiffusion_root, "setup_standalone.py")
    if os.path.exists(setup_script):
        try:
            subprocess.check_call([sys.executable, setup_script, "--install-deps-only"])
            return "All packages installed & verified successfully!"
        except Exception as e:
            return f"Error during update: {e}"
    return "setup_standalone.py not found."


def fix_pop():
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "nvidia-cudnn-cu11"], check=False)
        return "Done! Conflicting nvidia-cudnn-cu11 removed."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def check_version():
    return "TouchDiffusion Standalone v0.0.3 (Edición Clase IA)"


def open_link(link):
    webbrowser.open_new_tab(link)


models = list_files_in_folder("models/checkpoints")
model_types = ["sd_1.5_turbo", "sd_1.5"]

button_info = [
    ("Github", "https://github.com/olegchomp/TouchDiffusion"),
    ("Discord", "https://discord.gg/wNW8xkEjrf"),
    ("Youtube", "https://www.youtube.com/vjschool"),
    ("Telegram", "https://t.me/vjschool"),
    ("Author", "https://olegcho.mp/"),
    ("Buy me a coffee", "https://boosty.to/vjschool/"),
]

with gr.Blocks(title="TouchDiffusion — Generador de Motores TensorRT") as demo:
    gr.Markdown("## ⚡ TouchDiffusion — Compilador de Motores TensorRT (Edición Standalone Clase IA)")
    with gr.Tab("Engine"):
        with gr.Row():
            with gr.Column(scale=1):
                model_type_dd = gr.Dropdown(
                    model_types, value="sd_1.5_turbo", label="Select model type"
                )
                model_dropdown = gr.Dropdown(
                    ["SD Turbo"] + models,
                    value="SD Turbo",
                    allow_custom_value=True,
                    interactive=False,
                    label="Select model",
                )
                width_slider = gr.Slider(256, 1024, value=512, step=8, label="Width", interactive=True)
                height_slider = gr.Slider(256, 1024, value=512, step=8, label="Height", interactive=True)
                sampling_steps_slider = gr.Slider(
                    1, 20, value=1, step=1, label="Sampling steps (Batch size)", interactive=True
                )
                acceleration_radio = gr.Radio(
                    ["None"], value="None", label="Acceleration Lora not available for this model"
                )
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Output",
                    lines=6,
                    value="Listo para compilar motor TensorRT (o podés usar directamente '[PyTorch-Direct] sd-turbo' en TouchDesigner sin esperar).",
                )
                make_engine = gr.Button("Make engine (Compilar TensorRT)", variant="primary")

            make_engine.click(
                fn=stream_engine,
                inputs=[
                    width_slider,
                    height_slider,
                    sampling_steps_slider,
                    acceleration_radio,
                    model_dropdown,
                    model_type_dd,
                ],
                outputs=output,
            )
            model_type_dd.change(
                fn=update_interactivity,
                inputs=[model_type_dd, sampling_steps_slider],
                outputs=[model_dropdown, acceleration_radio],
            )
            sampling_steps_slider.change(
                fn=update_hypersd,
                inputs=[sampling_steps_slider, model_type_dd],
                outputs=[acceleration_radio],
            )

    with gr.Tab("Install & Update"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Text("Verifica o repara las dependencias del entorno .venv.", label="Description")
                install_update = gr.Button("Update / Verify dependencies", variant="primary")
                gr.Text(
                    "Si aparece un cartel de error de .dll en Windows, hacé clic en 'Fix pop up'.",
                    label="Additional step",
                )
                fix_popup = gr.Button("Fix pop up", variant="secondary")
            with gr.Column(scale=1):
                output_inst = gr.Textbox(label="Output")

            install_update.click(fn=inst_upd, inputs=[], outputs=output_inst)
            fix_popup.click(fn=fix_pop, inputs=[], outputs=output_inst)

    with gr.Tab("About"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Text("TouchDiffusion-v.0.0.3 (Standalone Clase IA)", label="Your version", interactive=False)
                gr.Text(check_version(), label="Status", interactive=False)
            with gr.Column(scale=1):
                with gr.Row():
                    for label, url in button_info:
                        button = gr.Button(label, variant="secondary")
                        button.click(lambda url=url: open_link(url))

if __name__ == "__main__":
    ensure_directories()
    demo.launch(inbrowser=False, server_name="127.0.0.1", server_port=7860)

"""Gradio app to test prompting and segmenting images with different encoders.

Make sure you have gradio installed `pip install gradio`

The app uses the rayfronts hydra configs to initialize the encoder.

Typical Usage:
  python scripts/encoder_semseg_app.py encoder=naradio encoder.model_version=radio_v2.5-b
"""

import sys
import os
import logging
import base64
from io import BytesIO
import colorsys
from functools import partial
from dataclasses import dataclass

import gradio as gr
from PIL import Image
import numpy as np
import torch
from matplotlib import cm
import hydra
from hydra.core.config_store import ConfigStore

sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import rayfronts.utils as utils

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
  # Chunk size to compute cos similarity. Reduce if getting OOM error.
  chunk_size: int = 10000

cs = ConfigStore.instance()
cs.store(name="extras", node=AppConfig)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Store prompts and colors
prompt_list = []
color_list = []

def apply_colormap(image: np.ndarray, cmap_name='viridis') -> np.ndarray:
  """Apply a colormap to a grayscale image and return an RGB uint8 image."""
  # Ensure image is normalized to [0, 1]
  if image.dtype != np.float16 and image.dtype != np.float32 and image.dtype != np.float64:
      image = image.astype(np.float32) / 255.0
  image = np.clip(image, 0, 1)
  cmap = cm.get_cmap(cmap_name)
  colored = cmap(image)[:, :, :3]  # Drop alpha channel
  return (colored * 255).astype(np.uint8)

def numpy_to_base64(img_array):
  """Convert a NumPy array image to base64 string."""
  if img_array.dtype != np.uint8:
      img_array = (img_array * 255).astype(np.uint8)  # normalize if needed
  pil_img = Image.fromarray(img_array)
  buffered = BytesIO()
  pil_img.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode()

def make_grid_output(images, labels):
  html = """
  <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;'>
  """
  for img_array, label in zip(images, labels):
    img_str = numpy_to_base64(img_array)
    html += f"""
    <div style='text-align: center;'>
      <div style='font-weight: bold; margin-bottom: 5px;'>{label}</div>
      <img src='data:image/png;base64,{img_str}' style='width: 100%; height: auto; border: 1px solid #ccc;' />
    </div>
    """
  html += "</div>"
  return html

def generate_distinct_color(index):
  """Generate visually distinct colors using HSV color space."""
  hue = (index * 0.61803398875) % 1  # golden ratio for spacing hues
  r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
  return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def add_prompt(prompts):
  for prompt in prompts.split("\n"):
    if not prompt.strip() or prompt in prompt_list:
      continue
    color = generate_distinct_color(len(prompt_list))
    prompt_list.append(prompt)
    color_list.append(color)

  # Format prompt display
  colored_prompts = [
    f"<span style='background-color:{color}; color:#FFFFFF'>{p}</span>" 
    for p, color in zip(prompt_list, color_list)
  ]
  return gr.update(value=""), gr.update(value="<br>".join(colored_prompts))

def clear_prompts():
  prompt_list.clear()
  color_list.clear()
  return gr.update(value=""), gr.update(value="")

def on_page_load():
  prompt_list.clear()
  color_list.clear()
  return gr.update(value="")

@torch.inference_mode()
def process_all(input_image, use_templates, softmax, resolution,
                model, chunk_size):
  N = len(prompt_list)
  resolution = (resolution, resolution)
  if N == 0:
    raise gr.Error("You must add some prompts", duration=5)
  elif softmax and N == 1:
    raise gr.Error("With softmax enabled, you need at least two prompts", duration=5)

  if hasattr(model, "input_resolution"):
    model.input_resolution = resolution

  logger.info("Prompts submitted: %s", str(prompt_list))
  m = "Computing feature map.."
  logger.info(m)
  yield m
  tensor_image = torch.from_numpy(input_image).permute(2, 0, 1)
  tensor_image = tensor_image.to(device).float() / 255.0
  tensor_image = torch.nn.functional.interpolate(
    tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)
  feat_map = model.encode_image_to_feat_map(tensor_image)
  feat_map = model.align_spatial_features_with_language(feat_map)
  feat_map = torch.nn.functional.interpolate(
    feat_map, resolution, mode="bilinear", antialias=True)
  feat_map = feat_map.squeeze(0).permute(1, 2, 0)

  m = "Computing prompt embeddings.."
  logger.info(m)
  yield m

  if use_templates:
    prompt_embeddings = model.encode_labels(prompt_list)
  else:
    prompt_embeddings = model.encode_prompts(prompt_list)

  m = "Computing cosine similarity.."
  logger.info(m)
  yield m
  H,W,C = feat_map.shape
  feat_map = feat_map.reshape(-1, C)
  num_chunks = int(np.ceil(feat_map.shape[0] / chunk_size))
  cos_sim = list()
  for c in range(num_chunks):
    cos_sim.append(utils.compute_cos_sim(
      prompt_embeddings, feat_map[c*chunk_size: (c+1)*chunk_size],
      softmax=softmax))
  cos_sim = torch.cat(cos_sim, dim=0)
  cos_sim = cos_sim.reshape(H,W,N)
  m = "Visualizing.."
  logger.info(m)
  yield m
  if not softmax:
    cos_sim = utils.norm_img_01(cos_sim.permute(2, 0, 1).unsqueeze(0))
    cos_sim = cos_sim.squeeze(0).permute(1, 2, 0)

  yield make_grid_output(
    [apply_colormap(x) for x in cos_sim.permute(2, 0, 1).cpu().numpy()],
    prompt_list)


@hydra.main(version_base="1.2",
            config_path="../rayfronts/configs",
            config_name="default")
@torch.inference_mode()
def main(cfg=None):
  encoder_kwargs = dict()
  if "NARadioEncoder" in cfg.encoder._target_:
    encoder_kwargs["input_resolution"] = [224, 224]
    encoder_kwargs["compile"] = False # Compiling will make resolution changes slow

  encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
  step = 16

  with gr.Blocks() as demo:
    desc = gr.HTML(
    """
    <p align="center"><img src="/gradio_api/file=assets/logo.gif" width="400" alt="RayFronts"/></p>
    <h1 align="center">Test the RayFronts encoders in 2D !</h1>
    <h3 align="center"><a href="https://arxiv.org/abs/2504.06994">Paper</a> | <a href="https://RayFronts.github.io/">Project Page</a> | <a href="https://www.youtube.com/watch?v=fFSKUBHx5gA">Video</a></h3>
    <p align="center">Note that results may be noisy in 2D and get smoothed out as you aggregate features in 3D giving RayFronts its robust 3D open-vocabulary semantic segmentation performance.</p>
    """)
    with gr.Row():
      with gr.Column(scale=1):
        input_image = gr.Image(label="Input Image", type="numpy")

        with gr.Row():
          use_templates = gr.Checkbox(label="Use templates", value=True)
          softmax = gr.Checkbox(label="Use softmax", value=True)
          res_slider = gr.Slider(224, 1024, 224, step=step, show_reset_button=False, label="Resolution", )

        with gr.Row(equal_height=True):
          prompt = gr.Textbox(label="Prompt", placeholder="Type a prompt...",
                              scale=15)
          add_button = gr.Button("+", scale=1)

        prompt_display = gr.HTML()  # To show added prompts
        with gr.Row():
          clear_button = gr.Button("Clear")
          process_button = gr.Button("Run")

      with gr.Column(scale=2):
        # output_image = gr.Image(label="Output Image", type="numpy")
        output_image = gr.HTML()

      add_button.click(
        fn=add_prompt,
        inputs=prompt,
        outputs=[prompt, prompt_display])
      process_button.click(
        fn=partial(process_all, model=encoder, chunk_size=cfg.chunk_size),
        inputs=[input_image, use_templates, softmax, res_slider],
        outputs=output_image)
      clear_button.click(
        fn=clear_prompts,
        inputs=None,
        outputs=[prompt, prompt_display])

    examples = gr.Examples(
      examples=[
        ["assets/example1.jpg", True, True, "Pothole\nRoad\nSky\nCar\nWater", 224],
        ["assets/example2.jpg", False, True, "Person\nShoes\nGrey Jacket\nRed overalls\nRoad\nCrosswalk\nCar", 512],
        ["assets/example3.jpg", True, True, "Paved ground\nFlood lights\nRed Container\nBuilding\nTanker\nSky\nClouds\nTreeline", 768]
      ],
      inputs=[input_image, use_templates, softmax, prompt, res_slider],
    )

    demo.load(fn=on_page_load, inputs=None, outputs=prompt_display)
  demo.launch(allowed_paths=["assets/logo.gif"])


if __name__ == "__main__":
  main()
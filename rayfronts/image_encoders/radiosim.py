"""Includes the RayFronts Encoder based on NACLIP + RADIO models.

The module only relies on the base.py and prompt_templates.py files in
image_encoders. Encoder can be copied with those files to your own project.

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0),size=(512, 512))

  labels = ["car", "person"]

  enc = NARadioEncoder(model_version="radio_v2.5-b", lang_model="siglip",
                       input_resolution=[512,512])
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

from typing_extensions import override, List, Tuple

import torch

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder

import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import use_fused_attn
import math


class SimilarityAttn(nn.Module):
  """Encompases the NACLIP attention mechanism."""

  def __init__(
    self,
    orig_attn,
    device,
    chosen_cls_id: int,
    dim: int,
    qk_norm: bool = False,
    num_prefix_tokens: int = 8,
    sim_scale: int = 10,
    prefix_scale: int = 2
  ) -> None:
    super().__init__()
    num_heads = orig_attn.num_heads
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.fused_attn = use_fused_attn()
    self.chosen_cls_id = chosen_cls_id

    self.qkv = orig_attn.qkv
    self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
    self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
    self.attn_drop = orig_attn.attn_drop
    self.proj = orig_attn.proj
    self.proj_drop = orig_attn.proj_drop
    self.device = device
    self.num_prefix_tokens = num_prefix_tokens
    self.sim_scale = sim_scale
    self.prefix_scale = prefix_scale

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    B, N, C = x.shape
    x_out = self.custom_attn(x.permute(1, 0, 2))
    x_out = x_out.permute(1, 0, 2)
    return x_out

  
  def custom_attn(self, x):
    num_heads = self.num_heads
    num_tokens, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    # kk.T vs kq.T has the most impact
    attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    # Similarity Attention 
    prefix_weights = attn_weights[:,:,:self.num_prefix_tokens]

    attn_output = attn_output.view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    sim_tokens =  F.normalize(attn_output,dim = -1)
    sim_matrix = torch.bmm(sim_tokens,sim_tokens.transpose(1,2))*self.sim_scale
    sim_matrix[sim_matrix < 0] = -torch.inf
    sim_matrix = F.softmax(sim_matrix,dim=-1)
    sim_matrix[:,:,:self.num_prefix_tokens] = prefix_weights*self.prefix_scale
    attn_output = torch.bmm(sim_matrix, v)    
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    # attn_output = torch.bmm(sim_matrix,attn_output).transpose(0, 1).contiguous().view(
    #   -1, bsz, embed_dim)
    
    return attn_output


class RadioSimilarityPrefixEncoder(LangSpatialGlobalImageEncoder):
  """The RayFronts Encoder based on NACLIP + RADIO models.

  The model modifies the attention of the last layer of RADIO following the
  example of NACLIP improving spatial structure. And uses the Summary CLS 
  projection to project the patch-wise tokens to SIGLIP or CLIP language aligned
  feature spaces. The model computes na-radio spatial or global features by
  default and exposes functions to project those features to Siglip, or CLIP
  feature spaces.
  """

  def __init__(self, device: str =None,
               model_version: str = "radio_v2.5-b",
               lang_model: str ="siglip",
               return_radio_features: bool = True,
               compile: bool = True,
               amp: bool = True,
               sim_scale: int = 10,
               prefix_scale: int = 2,
               **kwargs):
    """

    Args:
      device: "cpu" or "cuda", set to None to use CUDA if available.
      model_version: Choose from "radio_v2.5-x" where x can be b,l, or g.
        More models can be found on https://github.com/NVlabs/RADIO/
      lang_model: choose from ["siglip", "clip"]
      input_resolution: Tuple of ints (height, width) of the input images.
        Needed to initialize the guassian attention window.
      gauss_std: Standard deviation of the gaussian kernel.
      return_radio_features: Whether to return radio features which are not
        language aligned or whether to project them to the language aligned
        space directly. If True, then the user can always later use the
        functions `align_global_features_with_language` or 
        `align_spatial_features_with_language` to project the radio features
        to be language aligned.
      compile: Whether to compile the model or not. Compiling may be faster but may increase memory usage.
      amp: Whether to use automatic mixed percision or not.
    """

    super().__init__(device)

    self.compile = compile
    self.amp = amp
    self.model_version = model_version
    self.return_radio_features = return_radio_features
    adaptor_names = [lang_model, "sam","clip"]
    self.model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                version=model_version, progress=True,
                                skip_validation=True,
                                adaptor_names=adaptor_names)
    self.model.eval()
    self.model = self.model.to(self.device)
    self.model.make_preprocessor_external()
    # Steal adaptors from RADIO so it does not auto compute adaptor output.
    # We want to control when that happens.
    self.lang_adaptor = self.model.adaptors[lang_model]
    self.sam_adaptor = self.model.adaptors["sam"]
    self.clip_adaptor = self.model.adaptors["clip"]
    self.model.adaptors = None
    last_block = self.model.model.blocks[-1]
    last_block.attn = SimilarityAttn(
      last_block.attn,
      dim=self.model.model.embed_dim,
      chosen_cls_id=self.lang_adaptor.head_idx,
      device=self.device,
      num_prefix_tokens=self.model.num_summary_tokens,
      sim_scale=sim_scale,
      prefix_scale=prefix_scale)

    self.times = list()
    if self.compile:
      self.model.compile(fullgraph=True, options={"triton.cudagraphs":True})
      self.lang_adaptor.compile(fullgraph=True, options={"triton.cudagraphs":True})

  @property
  def input_resolution(self):
    return self.model.model.blocks[-1].attn.input_resolution

  @input_resolution.setter
  def input_resolution(self, value):
    if hasattr(value, "__len__") and len(value) == 2:
      if self.is_compatible_size(*value):
        self.model.model.blocks[-1].attn.update_input_resolution(value)
        if self.compile:
          self.model.compile(fullgraph=True, options={"triton.cudagraphs":True})
      else:
        raise ValueError(f"Incompatible input resolution {value}")
    else:
      raise ValueError("Input resolution must be a tuple of two ints")

  @override
  def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
    prompts_per_label = self.insert_labels_into_templates(labels)
    all_text_features = list()
    for i in range(len(labels)):
      text_features = self.encode_prompts(prompts_per_label[i])
      text_features = text_features.mean(dim=0, keepdim=True)
      all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features

  @override
  def encode_labels_clip(self, labels: List[str]) -> torch.FloatTensor:
    prompts_per_label = self.insert_labels_into_templates(labels)
    all_text_features = list()
    for i in range(len(labels)):
      text_features = self.encode_prompts_clip(prompts_per_label[i])
      text_features = text_features.mean(dim=0, keepdim=True)
      all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features, dim=0)
    return all_text_features


  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      text = self.lang_adaptor.tokenizer(prompts).to(self.device)
      text_features = self.lang_adaptor.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

  @override
  def encode_prompts_clip(self, prompts: List[str]) -> torch.FloatTensor:
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      text = self.clip_adaptor.tokenizer(prompts).to(self.device)
      text_features = self.clip_adaptor.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

  @override
  def encode_image_to_vector(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:

    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.model(rgb_image)
      C = out.summary.shape[-1] // 3
      i = self.lang_adaptor.head_idx
      out = out.summary[:, C*i: C*(i+1)]

      if not self.return_radio_features:
        out = self.lang_adaptor.head_mlp(out)

    return out

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.model(rgb_image).features
      if not self.return_radio_features:
        out = self.lang_adaptor.head_mlp(out)
    return out.permute(0, 2, 1).reshape(B, -1, H_, W_)

  @override
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
      -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.model(rgb_image)

      C = out.summary.shape[-1] // 3
      i = self.lang_adaptor.head_idx
      global_vector = out.summary[:, C*i: C*(i+1)]

      feat_map = out.features

      if not self.return_radio_features:
        global_vector = self.lang_adaptor.head_mlp(global_vector)
        feat_map = self.lang_adaptor.head_mlp(feat_map)

    return feat_map, global_vector

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features
    B,C = features.shape
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      return self.lang_adaptor.head_mlp(features)

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features
    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.lang_adaptor.head_mlp(features)
    return out.permute(0, 2, 1).reshape(B, -1, H, W)

  def get_sam_spatial_features(self, features: torch.FloatTensor):
    if self.sam_adaptor is None:
      raise ValueError("Cannot align to sam without a sam model")
  
    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)

    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.sam_adaptor.feat_mlp(features)

    return out.permute(0, 2, 1).reshape(B, -1, H, W)

  def get_sam_summary_features(self, features: torch.FloatTensor):
    if self.sam_adaptor is None:
      raise ValueError("Cannot align to sam without a sam model")

    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)

    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.sam_adaptor.head_mlp(features)
      
    return out.permute(0, 2, 1).reshape(B, -1, H, W)
  
  def get_clip_spatial_features(self, features: torch.FloatTensor):
    if self.clip_adaptor is None:
      raise ValueError("Cannot align to clip without a clip model")
  
    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)

    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.clip_adaptor.feat_mlp(features)

    return out.permute(0, 2, 1).reshape(B, -1, H, W)

  def get_clip_summary_features(self, features: torch.FloatTensor):
    if self.clip_adaptor is None:
      raise ValueError("Cannot align to clip without a clip model")

    B,C,H,W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(B, -1, C)

    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.clip_adaptor.head_mlp(features)
      
    return out.permute(0, 2, 1).reshape(B, -1, H, W)

  @override
  def is_compatible_size(self, h: int, w: int):
    hh, ww = self.get_nearest_size(h, w)
    return hh == h and ww == w

  @override
  def get_nearest_size(self, h, w):
    return self.model.get_nearest_supported_resolution(h, w)

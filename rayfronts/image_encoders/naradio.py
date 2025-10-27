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


class GaussKernelAttn(nn.Module):
  """Encompases the NACLIP attention mechanism."""

  def __init__(
    self,
    orig_attn,
    input_resolution: tuple,
    gauss_std: float,
    device,
    chosen_cls_id: int,
    dim: int,
    qk_norm: bool = False,
    num_prefix_tokens: int = 8,
  ) -> None:
    super().__init__()
    num_heads = orig_attn.num_heads
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.fused_attn = use_fused_attn()
    self.input_resolution = input_resolution

    h, w = input_resolution
    n_patches = (w // 16, h //16)
    window_size = [side * 2 - 1 for side in n_patches]
    window = GaussKernelAttn.gaussian_window(*window_size, std=gauss_std,
                                             device=device)
    self.attn_addition = GaussKernelAttn.get_attention_addition(
      *n_patches, window, num_prefix_tokens
    ).unsqueeze(0)

    self.chosen_cls_id = chosen_cls_id
    self.gauss_std = gauss_std

    self.qkv = orig_attn.qkv
    self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
    self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
    self.attn_drop = orig_attn.attn_drop
    self.proj = orig_attn.proj
    self.proj_drop = orig_attn.proj_drop
    self.device = device
    self.num_prefix_tokens = num_prefix_tokens

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    #import pdb; pdb.set_trace()
    B, N, C = x.shape
    x_out = self.custom_attn(x.permute(1, 0, 2))
    x_out = x_out.permute(1, 0, 2)
    return x_out

  @staticmethod
  def gaussian_window(dim1, dim2, std=5., device="cuda"):
    constant = 1 / (std * math.sqrt(2))
    start = -(dim1 - 1) / 2.0
    k1 = torch.linspace(start=start * constant,
                        end=(start + (dim1 - 1)) * constant,
                        steps=dim1,
                        dtype=torch.float, device=device)
    start = -(dim2 - 1) / 2.0
    k2 = torch.linspace(start=start * constant,
                        end=(start + (dim2 - 1)) * constant,
                        steps=dim2,
                        dtype=torch.float, device=device)
    dist_square_to_mu = (torch.stack(torch.meshgrid(
      k1, k2, indexing="ij")) ** 2).sum(0)

    return torch.exp(-dist_square_to_mu)

  @staticmethod
  def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
    d = window.device
    m = torch.einsum("ij,kl->ijkl",
                     torch.eye(dim1, device=d),
                     torch.eye(dim2, device=d))
    m = m.permute((0, 3, 1, 2)).contiguous()
    out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1),
                   window.unsqueeze(0).unsqueeze(1),
                   padding='same').squeeze(1)

    out = out.view(dim1 * dim2, dim1 * dim2)
    if num_prefix_tokens > 0:
      v_adjusted = torch.vstack(
        [torch.zeros((num_prefix_tokens, dim1 * dim2), device=d), out])
      out = torch.hstack([torch.zeros(
        (dim1 * dim2 + num_prefix_tokens, num_prefix_tokens), device=d),
        v_adjusted])

    return out

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
    attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

    # Gaussian attention seems to have minimal impact
    attn_weights += self.attn_addition
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    return attn_output

  def update_input_resolution(self, input_resolution):
    h, w = input_resolution
    n_patches = (w // 16, h //16)
    window_size = [side * 2 - 1 for side in n_patches]
    window = GaussKernelAttn.gaussian_window(*window_size, std=self.gauss_std,
                                             device=self.device)
    self.attn_addition = GaussKernelAttn.get_attention_addition(
      *n_patches, window, self.num_prefix_tokens
    ).unsqueeze(0)

class NARadioEncoder(LangSpatialGlobalImageEncoder):
  """The RayFronts Encoder based on NACLIP + RADIO models.

  The model modifies the attention of the last layer of RADIO following the
  example of NACLIP improving spatial structure. And uses the Summary CLS 
  projection to project the patch-wise tokens to SIGLIP or CLIP language aligned
  feature spaces. The model computes na-radio spatial or global features by
  default and exposes functions to project those features to Siglip, or CLIP
  feature spaces.
  """

  def __init__(self, device: str = None,
               model_version: str = "radio_v2.5-b",
               lang_model: str ="siglip",
               input_resolution: Tuple[int,int] = [512,512],
               gauss_std: float = 7.0,
               return_radio_features: bool = True,
               compile: bool = True,
               amp: bool = True,
               use_sliding_window: bool = False,
               slide_crop_size: int = 336,
               slide_stride: int = 112):
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
      use_sliding_window: Whether to use sliding window inference for large images.
      slide_crop_size: Size of the sliding window crop (default 336).
      slide_stride: Stride of the sliding window (default 112).
    """

    super().__init__(device)

    self.compile = compile
    self.amp = amp
    self.model_version = model_version
    self.return_radio_features = return_radio_features
    self.use_sliding_window = use_sliding_window
    self.slide_crop_size = slide_crop_size
    self.model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                version=model_version, progress=True,
                                skip_validation=True,
                                adaptor_names=[lang_model])
    self.model.eval()
    self.model = self.model.to(self.device)
    self.model.make_preprocessor_external()
    # Steal adaptors from RADIO so it does not auto compute adaptor output.
    # We want to control when that happens.
    self.lang_adaptor = self.model.adaptors[lang_model]
    self.model.adaptors = None
    last_block = self.model.model.blocks[-1]
    self.slide_stride = slide_stride
    
    # Determine the resolution for GaussKernelAttn initialization
    if self.use_sliding_window:
      init_resolution = [int(self.slide_crop_size), int(self.slide_crop_size)]
    else:
      init_resolution = input_resolution
    
    # Replace attention layer with GaussKernelAttn
    last_block.attn = GaussKernelAttn(
      last_block.attn,
      init_resolution,
      gauss_std,
      dim=self.model.model.embed_dim,
      chosen_cls_id=self.lang_adaptor.head_idx,
      device=self.device,
      num_prefix_tokens=self.model.num_summary_tokens)
    
    # Now set input_resolution property (after GaussKernelAttn is in place)
    self.input_resolution = init_resolution

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

  def _preprocess_image_for_sliding_window(self, image: torch.FloatTensor, 
                                           stride: int = 16, 
                                           slide_crop: int = 336) -> torch.FloatTensor:
    """Preprocess image to ensure dimensions are compatible with patch size and crop size.
    
    Args:
      image: Input image tensor of shape (B, C, H, W)
      stride: Patch size (default 16 for RADIO models)
      slide_crop: Crop size for sliding window
      
    Returns:
      Preprocessed image with dimensions aligned to stride
    """
    longer_side = max(image.shape[2:])
    h, w = image.shape[2:]
    
    # If already aligned, return as is
    if h % stride == 0 and w % stride == 0:
      return image
      
    # Calculate new dimensions
    if longer_side % stride != 0:
      dst_longer = (longer_side // stride + 1) * stride
    else:
      dst_longer = longer_side
      
    new_h = int(h * dst_longer / longer_side)
    new_w = int(w * dst_longer / longer_side)
    
    # Align to stride
    if new_h % stride != 0:
      new_h = (new_h // stride + 1) * stride
    if new_w % stride != 0:
      new_w = (new_w // stride + 1) * stride
      
    # Ensure dimensions are at least as large as crop size
    new_h, new_w = max(new_h, slide_crop), max(new_w, slide_crop)
    
    image = torch.nn.functional.interpolate(
      image, (new_h, new_w), mode='bilinear', align_corners=False)
    
    return image

  def _get_windowed_images(self, img: torch.FloatTensor, 
                          stride: int, 
                          crop_size: int, 
                          patch_size: int = 16):
    """Create windowed crops of the input image for sliding window inference.
    
    Args:
      img: Input image tensor of shape (B, C, H, W)
      stride: Stride for sliding window
      crop_size: Size of each crop
      patch_size: Patch size of the model (default 16)
      
    Returns:
      Tuple of (batched_imgs, patch_locs, (h_grids, w_grids))
        - batched_imgs: Concatenated crops of shape (N, C, crop_size, crop_size)
        - patch_locs: List of patch locations [y1, x1, y2, x2] in patch coordinates
        - (h_grids, w_grids): Number of grids in height and width
    """
    batch_size, _, h_img, w_img = img.shape
    h_grids = max(h_img - crop_size + stride - 1, 0) // stride + 1
    w_grids = max(w_img - crop_size + stride - 1, 0) // stride + 1
    
    crop_imgs, patch_locs = [], []
    
    for h_idx in range(h_grids):
      for w_idx in range(w_grids):
        y1 = h_idx * stride
        x1 = w_idx * stride
        y2 = min(y1 + crop_size, h_img)
        x2 = min(x1 + crop_size, w_img)
        y1 = max(y2 - crop_size, 0)
        x1 = max(x2 - crop_size, 0)
        
        crop_img = img[:, :, y1:y2, x1:x2]
        
        # Verify alignment with patch size
        assert y1 % patch_size == 0 and x1 % patch_size == 0, \
          f"Crop location ({y1}, {x1}) not aligned to patch size {patch_size}"
        assert y2 % patch_size == 0 and x2 % patch_size == 0, \
          f"Crop location ({y2}, {x2}) not aligned to patch size {patch_size}"
        
        # Store patch locations in patch coordinates
        patch_locs.append(torch.tensor([
          y1 // patch_size, x1 // patch_size, 
          y2 // patch_size, x2 // patch_size
        ]))
        
        crop_imgs.append(crop_img)
    
    batched_imgs = torch.cat(crop_imgs, dim=0)  # [n_patches, C, crop_size, crop_size]
    return batched_imgs, patch_locs, (h_grids, w_grids)

  def _encode_image_to_feat_map_sliding(self, 
                                       rgb_image: torch.FloatTensor,
                                       stride: int = None,
                                       crop_size: int = None) -> torch.FloatTensor:
    """Encode image to feature map using sliding window inference.
    
    Args:
      rgb_image: Input RGB image tensor of shape (B, C, H, W)
      stride: Stride for sliding window (uses self.slide_stride if None)
      crop_size: Crop size for sliding window (uses self.slide_crop_size if None)
      
    Returns:
      Feature map of shape (B, C', H', W') where H' and W' are downsampled
    """
    if stride is None:
      stride = self.slide_stride
    if crop_size is None:
      crop_size = self.slide_crop_size
      
    # Preprocess image to ensure compatibility
    img = self._preprocess_image_for_sliding_window(
      rgb_image, stride=self.model.patch_size, slide_crop=crop_size)
    
    # Get windowed images
    batched_imgs, patch_locs, (h_grids, w_grids) = self._get_windowed_images(
      img, stride=stride, crop_size=crop_size, patch_size=self.model.patch_size)
    
    batch_size = img.shape[0]
    _, _, h_img, w_img = img.shape
    
    # Process all crops through the model
    B, C, H, W = batched_imgs.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    #import pdb; pdb.set_trace()
    
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      image_feats = self.model(batched_imgs).features
      if not self.return_radio_features:
        image_feats = self.lang_adaptor.head_mlp(image_feats)
    
    # Reshape features: (n_patches, n_tokens, feat_dim) -> (n_patches, feat_dim, H_, W_)
    image_feats = image_feats.permute(0, 2, 1).reshape(
      batched_imgs.shape[0], -1, H_, W_).float()
    
    # Initialize output feature map
    feat_dim = image_feats.shape[1]
    dtype = image_feats.dtype
    device = image_feats.device
    h_feat = math.ceil(h_img / self.model.patch_size)
    w_feat = math.ceil(w_img / self.model.patch_size)
    
    feat_map = torch.zeros((batch_size, feat_dim, h_feat, w_feat), 
                           dtype=dtype, device=device)
    count_mat = torch.zeros((batch_size, 1, h_feat, w_feat), 
                            dtype=dtype, device=device)
    
    # Accumulate features from all patches
    for h_idx in range(h_grids):
      for w_idx in range(w_grids):
        coord = patch_locs[h_idx * w_grids + w_idx]
        img_feat = image_feats[h_idx * w_grids + w_idx]
        
        feat_map[:, :, coord[0]:coord[2], coord[1]:coord[3]] += img_feat
        count_mat[:, :, coord[0]:coord[2], coord[1]:coord[3]] += 1
    
    # Average overlapping regions
    feat_map = feat_map / count_mat
    
    return feat_map

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
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      text = self.lang_adaptor.tokenizer(prompts).to(self.device)
      text_features = self.lang_adaptor.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.float()

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
    return out.float()

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    # Use sliding window inference if enabled and image is large
    if self.use_sliding_window:
      return self._encode_image_to_feat_map_sliding(rgb_image)
    
    # Standard inference
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      out = self.model(rgb_image).features
      if not self.return_radio_features:
        out = self.lang_adaptor.head_mlp(out)
    return out.permute(0, 2, 1).reshape(B, -1, H_, W_).float()

  @override
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
      -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # Note: Sliding window only applied to feature map encoding
    # Global vector is computed from the original image
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

    feat_map = feat_map.permute(0, 2, 1).reshape(B, -1, H_, W_).float()
    
    # If sliding window is enabled, re-encode feat_map with sliding window
    if self.use_sliding_window:
      feat_map = self._encode_image_to_feat_map_sliding(rgb_image)
    
    return feat_map, global_vector.float()

  def encode_image_to_feat_map_with_sliding_window(
    self, rgb_image: torch.FloatTensor, 
    stride: int = None, 
    crop_size: int = None) -> torch.FloatTensor:
    """Public method to encode image to feature map with sliding window.
    
    This method can be used to override the use_sliding_window setting
    and explicitly use sliding window inference.
    
    Args:
      rgb_image: Input RGB image tensor of shape (B, C, H, W)
      stride: Stride for sliding window (uses self.slide_stride if None)
      crop_size: Crop size for sliding window (uses self.slide_crop_size if None)
      
    Returns:
      Feature map of shape (B, C', H', W')
    """
    return self._encode_image_to_feat_map_sliding(rgb_image, stride, crop_size)

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features
    B,C = features.shape
    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
      features = self.lang_adaptor.head_mlp(features).float()
    return features.float()

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
    return out.permute(0, 2, 1).reshape(B, -1, H, W).float()

  @override
  def is_compatible_size(self, h: int, w: int):
    hh, ww = self.get_nearest_size(h, w)
    return hh == h and ww == w

  @override
  def get_nearest_size(self, h, w):
    return self.model.get_nearest_supported_resolution(h, w)

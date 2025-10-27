"""Includes the Radio Encoder module https://github.com/NVlabs/RADIO."""

from typing_extensions import override, Tuple, List

import torch
import math

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder

class RadioEncoder(LangSpatialGlobalImageEncoder):
  """Radio global and spatial encoder.
  
  The model computes radio spatial or global features by default and exposes 
  functions to project those features to Siglip, or CLIP feature spaces.
  """

  def __init__(self, device: str = None,
               model_version: str = "radio_v2.5-b",
               lang_model: str = None,
               return_radio_features: bool = True,
               use_summ_proj_for_spatial: bool = False,
               use_sliding_window: bool = False,
               slide_crop_size: int = 336,
               slide_stride: int = 112):
    """

    Args:
      device: "cpu" or "cuda", set to None to use CUDA if available.
      model_version: Choose from "radio_v2.5-x" where x can be b,l, or g.
        More models can be found on https://github.com/NVlabs/RADIO/
      lang_model: choose from ["siglip", "clip"]
      return_radio_features: Whether to return radio features which are not
        language aligned or whether to project them to the language aligned
        space directly. If True, then the user can always use the functions
        `align_global_features_with_language` or 
        `align_spatial_features_with_language` to project the radio features.
      use_summ_proj_for_spatial: Whether to use the summary projection MLP
        to also project the spatial features. Use this for language alignment.
      use_sliding_window: Whether to use sliding window inference for large images.
      slide_crop_size: Size of the sliding window crop (default 336).
      slide_stride: Stride of the sliding window (default 112).
    """

    super().__init__(device)

    self.use_sliding_window = use_sliding_window
    self.slide_crop_size = slide_crop_size
    self.slide_stride = slide_stride

    if not return_radio_features and lang_model is None:
      raise ValueError("Cannot request language aligned features without "
                       "specifying a language model.")

    self.model_version = model_version
    self.return_radio_features = return_radio_features
    self.use_summ_proj_for_spatial = use_summ_proj_for_spatial
    adaptor_names = [lang_model] if lang_model is not None else None
    self.model = torch.hub.load("NVlabs/RADIO", "radio_model",
                                version=model_version, progress=True,
                                skip_validation=True,
                                adaptor_names=adaptor_names)
    self.model.eval()
    self.model = self.model.to(self.device)
    # Steal adaptors from RADIO so it does not auto compute adaptor output.
    # We want to control when that happens.
    if lang_model is not None:
      self.lang_adaptor = self.model.adaptors[lang_model]
      self.model.adaptors = None
    else:
      self.lang_adaptor = None


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
    return all_text_features.float()

  @override
  def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
    text = self.lang_adaptor.tokenizer(prompts).to(self.device)
    text_features = self.lang_adaptor.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.float()

  @override
  def encode_image_to_vector(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:

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
    if self.use_sliding_window:
      return self._encode_image_to_feat_map_sliding(rgb_image)

    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    out = self.model(rgb_image).features

    if not self.return_radio_features:
      if self.use_summ_proj_for_spatial:
        mlp = self.lang_adaptor.head_mlp
      else:
        mlp = self.lang_adaptor.feat_mlp
      out = mlp(out)

    return out.permute(0, 2, 1).reshape(B, -1, H_, W_).float()

  @override
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
      -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    out = self.model(rgb_image)

    C = out.summary.shape[-1] // 3
    i = self.lang_adaptor.head_idx
    global_vector = out.summary[:, C*i: C*(i+1)]

    feat_map = out.features

    if not self.return_radio_features:
      global_vector = self.lang_adaptor.head_mlp(global_vector)
      if self.use_summ_proj_for_spatial:
        mlp = self.lang_adaptor.head_mlp
      else:
        mlp = self.lang_adaptor.feat_mlp
      feat_map = mlp(feat_map)

    feat_map = feat_map.permute(0, 2, 1).reshape(B, -1, H_, W_).float()
    
    # If sliding window is enabled, re-encode feat_map with sliding window
    if self.use_sliding_window:
      feat_map = self._encode_image_to_feat_map_sliding(rgb_image)

    return feat_map, global_vector.float()

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features

    B,C = features.shape
    return self.lang_adaptor.head_mlp(features)

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor):
    if self.lang_adaptor is None:
      raise ValueError("Cannot align to language without a lang model")
    if not self.return_radio_features:
      return features
    B,C,H,W = features.shape
    if self.use_summ_proj_for_spatial:
      mlp = self.lang_adaptor.head_mlp
    else:
      mlp = self.lang_adaptor.feat_mlp
    out = mlp(features.permute(0, 2, 3, 1))
    return out.permute(0, 3, 1, 2)

  @override
  def is_compatible_size(self, h: int, w: int):
    hh, ww = self.get_nearest_size(h, w)
    return hh == h and ww == w

  @override
  def get_nearest_size(self, h, w):
    return self.model.get_nearest_supported_resolution(h, w)

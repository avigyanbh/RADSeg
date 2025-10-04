"""Includes the Radio Encoder module https://github.com/NVlabs/RADIO."""

from typing_extensions import override, Tuple, List

import torch

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
               use_summ_proj_for_spatial: bool = False):
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
    """

    super().__init__(device)

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
    text = self.lang_adaptor.tokenizer(prompts).to(self.device)
    text_features = self.lang_adaptor.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

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
    B, C, H, W = rgb_image.shape
    H_, W_ = H // self.model.patch_size, W // self.model.patch_size
    out = self.model(rgb_image).features

    if not self.return_radio_features:
      if self.use_summ_proj_for_spatial:
        mlp = self.lang_adaptor.head_mlp
      else:
        mlp = self.lang_adaptor.feat_mlp
      out = mlp(out)

    return out.permute(0, 2, 1).reshape(B, -1, H_, W_)

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

    return feat_map, global_vector

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

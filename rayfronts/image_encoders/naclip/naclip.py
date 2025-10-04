"""Includes the NACLIP encoder https://github.com/sinahmr/NACLIP

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0),size=(512, 512))

  labels = ["car", "person"]

  enc = NACLIPEncoder(model_version="ViT-B/16")
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

from typing_extensions import override, Tuple, List

import torch
from torchvision.transforms import v2

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder
from rayfronts.image_encoders.naclip.clip_utils import clip


class NACLIPEncoder(LangSpatialGlobalImageEncoder):
  """Defines the NACLIP encoder."""

  def __init__(self, device: str = None,
               model_version: str = "ViT-B/16",
               gauss_std: float = 5.0):
    """

    Args:
      device: "cpu" or "cuda", set to None to use CUDA if available.
      model_version: Choose from ["ViT-B/32", "ViT-B/16", "ViT-L/14",
        "ViT-L/14@336px"]
      gauss_std: Standard deviation of the gaussian kernel.
    """
    super().__init__(device)
    self.model, _ = clip.load(model_version, device=device, jit=False)
    self.model.eval()
    self.model = self.model.to(self.device)
    self.model.visual.set_params(arch="reduced", attn_strategy="naclip",
                                 gaussian_std=gauss_std)

    self.norm = v2.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))

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
    query = clip.tokenize(prompts).to(self.device)
    text_features = self.model.encode_text(query)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.float()

  @override
  def encode_image_to_vector(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    return self.encode_image_to_feat_map_and_vector(rgb_image)[1]

  @override
  def encode_image_to_feat_map(
    self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
    return self.encode_image_to_feat_map_and_vector(rgb_image)[0]

  @override
  def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
      -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    B, C, H, W = rgb_image.shape
    rgb_image = self.norm(rgb_image)
    patch_size = self.model.visual.patch_size
    H_, W_ = H // patch_size, W // patch_size
    out = self.model.encode_image(rgb_image, return_all = True)
    feat_map = out[:, 1:].permute(0, 2, 1).reshape(B, -1, H_, W_).float()
    feat_global = out[:, 0].float()

    return feat_map, feat_global

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor):
    return features

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor):
    return features

  @override
  def is_compatible_size(self, h: int, w: int):
    return h == 224 and w == 224

  @override
  def get_nearest_size(self, h, w):
    return (224, 224)

"""Includes the Talk2DINO encoder.

Typical Usage (with automatic resizing):

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255

  labels = ["car", "person"]

  # Option 1: Auto-resize to 518x518 or some other dimension which is divisible by 14
  enc = Talk2DinoEncoder(device="cuda", model_name="dinov2_vitb14", 
                         clip_model_name="ViT-B/16", resize_image=True, resize_dim=518)
  
  # Option 2: Use original image size (must be divisible by 14 / can use the get_nearest_size method to get the nearest size)
  enc = Talk2DinoEncoder(device="cuda", model_name="dinov2_vitb14", 
                         clip_model_name="ViT-B/16")
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

import os
import sys
from math import sqrt
from typing_extensions import override, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder
from rayfronts.image_encoders import clip

talk2dino_dir = os.path.dirname(os.path.abspath(__file__))
if talk2dino_dir not in sys.path:
    sys.path.insert(0, talk2dino_dir)
from src.model import ProjectionLayer, VisualProjectionLayer, CLIPLastLayer, DoubleMLP  


def normalize_feature(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + 1e-6)


class Talk2DINOEncoder(LangSpatialGlobalImageEncoder):
    """Talk2DINO encoder.
    
    This encoder uses DINOv2 as the vision backbone and CLIP for text encoding,
    with a projection layer to align the features.
    """

    def __init__(self, device=None,
                 model_name: str = "dinov2_vitb14",
                 resize_image: bool = False,
                 resize_dim: int = 518,
                 clip_model_name: str = "ViT-B/16",
                 proj_config: str = None,
                 proj_model_name: str = "ProjectionLayer",
                 proj_weights: str = None):
        """
        Args:
            device: "cpu" or "cuda", set to None to use CUDA if available.
            model_name: DINOv2 model name (e.g., "dinov2_vitb14")
            resize_image: Whether to resize images to resize_dim. If False, accepts arbitrary sizes.
            resize_dim: Target image size if resize_image=True. Only used for compatibility checks.
            clip_model_name: CLIP model name for text encoding
            proj_config: Path to projection config file
            proj_model_name: Name of projection model class
            proj_weights: Path to projection weights
        """
        super().__init__(device)
        
        self.model_name = model_name
        self.resize_image = resize_image
        self.resize_dim = resize_dim
        
        # Set default paths if not provided
        if proj_config is None:
            proj_config = os.path.join(os.path.dirname(__file__), "src", "configs", "vitb_mlp_infonce.yaml")
        if proj_weights is None:
            proj_weights = os.path.join(os.path.dirname(__file__), "src", "weights", "vitb_mlp_infonce.pth")

        if 'dinov2' in model_name:
            self.vision = torch.hub.load('facebookresearch/dinov2', model_name)
        elif any(k in model_name for k in ['mae', 'sam', 'clip', 'dino']):
            self.vision = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=resize_dim)
        else:
            raise ValueError(f"Unknown vision model_name: {model_name}")
        self.vision.eval().to(self.device)
        for p in self.vision.parameters():
            p.requires_grad = False

        self.clip_model, _ = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Projection head
        import yaml
        import importlib
        with open(proj_config, 'r') as f:
            cfg = yaml.safe_load(f)['model']
        ProjClass = getattr(importlib.import_module('src.model'), proj_model_name)
        self.proj = ProjClass.from_config(cfg)
        if os.path.isfile(proj_weights):
            self.proj.load_state_dict(torch.load(proj_weights, map_location='cpu'))
        self.proj.eval().to(self.device)

        mean = (0.485, 0.456, 0.406) if 'clip' not in model_name else (0.4815, 0.4578, 0.4082)
        std = (0.229, 0.224, 0.225) if 'clip' not in model_name else (0.2686, 0.2613, 0.2758)
        
        transforms_list = []
        if self.resize_image:
            transforms_list.append(T.Resize((resize_dim, resize_dim)))
        transforms_list.extend([
            T.Normalize(mean, std),
        ])
        self.image_transforms = T.Compose(transforms_list)


    @torch.no_grad()
    def encode_image_tokens(self, image_bchw: torch.Tensor) -> torch.Tensor:
        x = self.image_transforms(image_bchw).to(self.device)
        if 'dinov2' in self.model_name:
            tokens = self.vision.forward_features(x)['x_norm_patchtokens']
        else:
            tokens = self.vision.forward_features(x)[:, 1:, :]
        return tokens  # [B, N, C]

    @torch.no_grad()
    def project_and_reshape(self, tokens: torch.Tensor) -> torch.Tensor:
        # Project visual tokens if projection provides a visual projector
        if hasattr(self.proj, 'project_dino'):
            tokens = self.proj.project_dino(tokens.float())
        if hasattr(self.proj, 'project_visual'):
            tokens = self.proj.project_visual(tokens.float())

        b, npatch, c = tokens.shape
        side = int(sqrt(npatch))
        feat = tokens.view(b, side, side, c).permute(0, 3, 1, 2).contiguous()
        return feat  # [B, C, H', W']

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
        text_features = self.clip_model.encode_text(query)

        # Project text if required
        if hasattr(self.proj, 'project_clip_txt'):
            text_features = self.proj.project_clip_txt(text_features)
        
        text_features = normalize_feature(text_features, dim=-1)
        return text_features.float()

    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        return self.encode_image_to_feat_map_and_vector(rgb_image)[1]

    @override
    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        return self.encode_image_to_feat_map_and_vector(rgb_image)[0]

    @override
    def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        tokens = self.encode_image_tokens(rgb_image)   
        feat_map = self.project_and_reshape(tokens)
        
        # Get global vector (CLS token or mean pooled)
        if 'dinov2' in self.model_name:
            x = self.image_transforms(rgb_image).to(self.device)
            global_vector = self.vision.forward_features(x)['x_norm_clstoken']
        else:
            global_vector = tokens.mean(dim=1)
        
        if hasattr(self.proj, 'project_dino'):
            global_vector = self.proj.project_dino(global_vector)
        if hasattr(self.proj, 'project_visual'):
            global_vector = self.proj.project_visual(global_vector)
        
        global_vector = normalize_feature(global_vector, dim=-1)
        
        return feat_map.float(), global_vector.float()

    @override
    def align_global_features_with_language(self, features: torch.FloatTensor):
        return features

    @override
    def align_spatial_features_with_language(self, features: torch.FloatTensor):
        return features

    @override
    def is_compatible_size(self, h: int, w: int):
        if self.resize_image:
            return True
        patch_size = 14
        return (h % patch_size == 0) and (w % patch_size == 0)

    @override
    def get_nearest_size(self, h, w):
        if self.resize_image:
            return self.resize_dim, self.resize_dim
        patch_size = 14
        nearest_h = int(round(h / patch_size) * patch_size)
        nearest_w = int(round(w / patch_size) * patch_size)
        return nearest_h, nearest_w

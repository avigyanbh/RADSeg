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

from einops import rearrange

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
        self.clip_model_name = clip_model_name
        
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

        self.h_bar = 0
        self.w_bar = 0
        self.use_avg_text_token = False


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
        # Project visual tokens based on projection type
        # Check type explicitly to match original implementation
        proj_type = type(self.proj).__name__
        
        if proj_type == 'VisualProjectionLayer':
            tokens = self.proj.project_dino(tokens.float())
        elif proj_type == 'DoubleMLP':
            tokens = self.proj.project_visual(tokens.float())
        # For ProjectionLayer, no visual projection is applied

        b, npatch, c = tokens.shape
        #side = int(sqrt(npatch))
        feat = tokens.view(b, int(self.h_bar), int(self.w_bar), c).permute(0, 3, 1, 2).contiguous()
        return feat  # [B, C, H', W']

    def average_text_tokens(text_embeddings, mask, keep_cls=False, keep_end_seq=False):
        if not keep_end_seq:
            mask[torch.arange(mask.shape[0]), mask.sum(dim=1) - 1] = False # excluding end of sequence
        if not keep_cls:
            mask[:, 0] = False # excluding CLS token
        
        
        masked_embeddings = text_embeddings * mask.unsqueeze(-1)  # shape: [BS, SEQ_LEN, 512]

        sum_embeddings = masked_embeddings.sum(dim=1)  # shape: [BS, 512]

        valid_elements = mask.sum(dim=1, keepdim=True)  # shape: [BS, 1]

        mean_embeddings = sum_embeddings / valid_elements  # shape: [BS, 512]
        
        return mean_embeddings
    
    @torch.no_grad()
    def build_dataset_class_tokens(self, classnames):
        tokens = []
        templates = ["a photo of a {}.",]
        for classname in classnames:
            if 'bert' not in self.clip_model_name:
                tokens.append(
                    clip.tokenize([template.format(classname) for template in templates]).to(self.device)
                )
            else:
                tokens.append(self.tokenizer([template.format(classname) for template in templates], return_tensors='pt', padding='max_length')['input_ids'])
        # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
        tokens = torch.stack(tokens)

        return tokens
        
    @torch.no_grad()
    def build_text_embedding(self, text):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        #import pdb; pdb.set_trace()
        text = text.to(self.device)
        num_classes, num_templates = text.shape[:2]
        text_argmax = text.argmax(dim=-1)
        text_argmax = rearrange(text_argmax, 'n t -> (n t)', n=num_classes, t=num_templates)
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        # chunked inference for memory limitation
        chunk_size = 32
        N = text.size(0)
        if type(self.proj) == CLIPLastLayer:
            text_embs = torch.cat([
            self.proj.project_clip_txt(self.encode_text(text[i:i + chunk_size]).permute(1, 0, 2), text_argmax=text_argmax[i:i + chunk_size])
            for i in range(0, N, chunk_size)
        ])
        else:
            if not self.use_avg_text_token:
                # performing classification using CLS textual token
                if 'bert' not in self.clip_model_name:
                    text_embs = torch.cat([
                        self.clip_model.encode_text(text[i:i + chunk_size])
                        for i in range(0, N, chunk_size)
                    ])
                else:
                    # encoding with BERT
                    text_embs = []
                    for i in range(0, N, chunk_size):
                        outputs = self.clip_model(text[i:i + chunk_size])
                        text_embs.append(outputs['pooler_output'])
                    text_embs = torch.cat(text_embs)
            else:
                # using text token average
                text_embs = []
                for i in range(0, N, chunk_size):
                    self.clip_model.encode_text(text[i:i + chunk_size])
                    text_embs.append(self.average_text_tokens(self.feats['clip_txt_out_tokens'] @ self.clip_model.text_projection, text[i:i + chunk_size] > 0))
                text_embs = torch.cat(text_embs)
        # [N, T, C]
        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_embs = text_embs.mean(dim=1).float()
        if type(self.proj) == ProjectionLayer or type(self.proj) == DoubleMLP:
            text_embs = self.proj.project_clip_txt(text_embs)
        #text_embs = F.normalize(text_embs, dim=-1, eps=1e-6)

        return text_embs
    
    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        tokens = self.build_dataset_class_tokens(labels)
        # prompts_per_label = self.insert_labels_into_templates(labels)
        # all_text_features = list()
        # for i in range(len(labels)):
        #     text_features = self.encode_prompts(prompts_per_label[i])
        #     text_features = text_features.mean(dim=0, keepdim=True)
        #     all_text_features.append(text_features)

        # all_text_features = torch.cat(all_text_features, dim=0)
        # return all_text_features
        return self.build_text_embedding(tokens)
    @override
    def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
        query = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(query)

        # Project text based on projection type
        proj_type = type(self.proj).__name__
        
        if proj_type in ['ProjectionLayer', 'DoubleMLP']:
            text_features = self.proj.project_clip_txt(text_features)
        elif proj_type == 'CLIPLastLayer':
            # CLIPLastLayer requires special handling - not supported in simplified API
            # Use the base CLIP features without additional projection
            pass
        
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
        #import pdb; pdb.set_trace()
        
        # Convert BGR to RGB if needed (reverse channel order)
        #rgb_image = rgb_image.flip(dims=[1])  # Uncomment this line if input is BGR
        #rgb_image = rgb_image[:, [2, 1, 0], :, :]
        
        nearest_h, nearest_w = self.get_nearest_size(rgb_image.shape[2], rgb_image.shape[3])
        rgb_image = torch.nn.functional.interpolate(rgb_image,
          size=(nearest_h, nearest_w), mode="bilinear",
          antialias=True)
        self.h_bar = rgb_image.shape[2] / 14
        self.w_bar = rgb_image.shape[3] / 14
        #import pdb; pdb.set_trace()
        tokens = self.encode_image_tokens(rgb_image)   
        feat_map = self.project_and_reshape(tokens)
        
        # Get global vector (CLS token or mean pooled)
        if 'dinov2' in self.model_name:
            x = self.image_transforms(rgb_image).to(self.device)
            global_vector = self.vision.forward_features(x)['x_norm_clstoken']
        else:
            global_vector = tokens.mean(dim=1)
        
        # Project global vector based on projection type
        proj_type = type(self.proj).__name__
        
        if proj_type == 'VisualProjectionLayer':
            global_vector = self.proj.project_dino(global_vector)
        elif proj_type == 'DoubleMLP':
            global_vector = self.proj.project_visual(global_vector)
        # For ProjectionLayer, no visual projection is applied
        
        global_vector = normalize_feature(global_vector, dim=-1)
        feat_map = normalize_feature(feat_map, dim=-1)
        
        return feat_map.float(), global_vector.float()

    @override
    def align_global_features_with_language(self, features: torch.FloatTensor):
        """Normalize global features to be in the same space as text features."""
        return normalize_feature(features, dim=-1)

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

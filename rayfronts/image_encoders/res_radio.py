"""ResRADIO encoder implementation.

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0), size=(512, 512))

  labels = ["car", "person"]

  enc = ResRadioEncoder(model_version="radio_v2.5-b", lang_model="siglip",
                       input_resolution=[512,512])
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

from typing_extensions import override, List, Tuple, Optional
from typing import Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder


class ResCLIPAttention(nn.Module):
    
    def __init__(
        self,
        orig_attn,
        input_resolution: Tuple[int, int],
        gaussian_std: float,
        device,
        dim: int,
        num_prefix_tokens: int = 8,
        use_rcs: bool = True,
        use_sfr: bool = True,
        rcs_layers: List[int] = None,  # Which intermediate layers to use for RCS
        attn_rcs_weights: Tuple[float, float] = (2.0, 0.6),
        attn_sfr_weights: Tuple[float, float] = (2.1, 0.6),
        temp_thd: float = 0.20,
        delete_same_entity: bool = True,
    ) -> None:
        super().__init__()
        
        num_heads = orig_attn.num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.input_resolution = input_resolution
        
        # ResCLIP parameters
        self.use_rcs = use_rcs
        self.use_sfr = use_sfr
        self.rcs_layers = rcs_layers if rcs_layers is not None else [5, 6, 7, 8]
        self.attn_rcs_weights = attn_rcs_weights
        self.attn_sfr_weights = attn_sfr_weights
        self.temp_thd = temp_thd
        self.delete_same_entity = delete_same_entity
        self.gaussian_std = gaussian_std
        
        # Initialize gaussian attention addition (similar to NACLIP)
        h, w = input_resolution
        patch_size = 16
        n_patches = (w // patch_size, h // patch_size)
        window_size = [side * 2 - 1 for side in n_patches]
        window = self.gaussian_window(*window_size, std=gaussian_std, device=device)
        self.attn_addition = self.get_attention_addition(
            *n_patches, window, num_prefix_tokens
        ).unsqueeze(0)
        
        # original attention components
        self.qkv = orig_attn.qkv
        self.q_norm = getattr(orig_attn, 'q_norm', nn.Identity())
        self.k_norm = getattr(orig_attn, 'k_norm', nn.Identity())
        self.attn_drop = orig_attn.attn_drop
        self.proj = orig_attn.proj
        self.proj_drop = orig_attn.proj_drop
        self.device = device
        self.num_prefix_tokens = num_prefix_tokens
        
        # intermediate attention maps and features
        self.intermediate_attentions = []
        self.current_features = None
        self.query_features = None
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        self.current_features = x  # Storing current features for potential SFR use
        x_out = self.custom_attn(x.permute(1, 0, 2))
        x_out = x_out.permute(1, 0, 2)
        return x_out
    
    def store_intermediate_attention(self, attention_weights: torch.Tensor):
        self.intermediate_attentions.append(attention_weights.detach())
    
    def clear_intermediate_attentions(self):
        self.intermediate_attentions = []
    
    def get_rcs_attention(self) -> Optional[torch.Tensor]:
        if not self.use_rcs or len(self.intermediate_attentions) == 0:
            return None
            
        selected_attns = []
        for layer_idx in self.rcs_layers:
            if layer_idx < len(self.intermediate_attentions):
                selected_attns.append(self.intermediate_attentions[layer_idx])
        
        if len(selected_attns) == 0:
            return None
            
        selected_attns = torch.stack(selected_attns)
        attention_rcs = selected_attns.mean(dim=0)
        return attention_rcs
    
    def get_sfr_attention(self) -> Optional[torch.Tensor]:
        if not self.use_sfr or self.current_features is None or self.query_features is None:
            return None
        return None # TODO: Implement SFR attention, returning None for now
    
    def custom_attn(self, x: torch.Tensor) -> torch.Tensor:
        num_heads = self.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5
        
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
        #import pdb; pdb.set_trace()
        
        attention_rcs = self.get_rcs_attention()
        attention_sfr = self.get_sfr_attention()
        
        # Apply SFR if available
        if attention_sfr is not None:
            attention_sfr = attention_sfr / torch.max(attention_sfr)
            attention_sfr = attention_sfr.to(attn_weights.dtype)
            tau_sfr, lambda_sfr = self.attn_sfr_weights
            attn_weights = tau_sfr * ((1 - lambda_sfr) * attn_weights + lambda_sfr * attention_sfr)
        
        # Add gaussian neighbor prior
        attn_weights += self.attn_addition
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply RCS if available
        if attention_rcs is not None:
            tau_rcs, lambda_rcs = self.attn_rcs_weights
            attn_weights = tau_rcs * ((1 - lambda_rcs) * attn_weights + lambda_rcs * attention_rcs)
        
        attn_weights = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        
        return attn_output
    
    @staticmethod
    def gaussian_window(dim1: int, dim2: int, std: float = 5.0, device: str = "cuda") -> torch.Tensor:
        constant = 1 / (std * math.sqrt(2))
        start = -(dim1 - 1) / 2.0
        k1 = torch.linspace(
            start=start * constant,
            end=(start + (dim1 - 1)) * constant,
            steps=dim1,
            dtype=torch.float, 
            device=device
        )
        start = -(dim2 - 1) / 2.0
        k2 = torch.linspace(
            start=start * constant,
            end=(start + (dim2 - 1)) * constant,
            steps=dim2,
            dtype=torch.float, 
            device=device
        )
        dist_square_to_mu = (torch.stack(torch.meshgrid(k1, k2, indexing="ij")) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)
    
    @staticmethod
    def get_attention_addition(dim1: int, dim2: int, window: torch.Tensor, num_prefix_tokens: int = 8) -> torch.Tensor:
        d = window.device
        m = torch.einsum("ij,kl->ijkl", torch.eye(dim1, device=d), torch.eye(dim2, device=d))
        m = m.permute((0, 3, 1, 2)).contiguous()
        out = F.conv2d(
            m.view(-1, dim1, dim2).unsqueeze(1),
            window.unsqueeze(0).unsqueeze(1),
            padding='same'
        ).squeeze(1)
        
        out = out.view(dim1 * dim2, dim1 * dim2)
        if num_prefix_tokens > 0:
            v_adjusted = torch.vstack([
                torch.zeros((num_prefix_tokens, dim1 * dim2), device=d), 
                out
            ])
            out = torch.hstack([
                torch.zeros((dim1 * dim2 + num_prefix_tokens, num_prefix_tokens), device=d),
                v_adjusted
            ])
        
        return out
    
    def update_input_resolution(self, input_resolution: Tuple[int, int]):
        h, w = input_resolution
        patch_size = 16
        n_patches = (w // patch_size, h // patch_size)
        window_size = [side * 2 - 1 for side in n_patches]
        window = self.gaussian_window(*window_size, std=self.gaussian_std, device=self.device)
        self.attn_addition = self.get_attention_addition(
            *n_patches, window, self.num_prefix_tokens
        ).unsqueeze(0)
        self.input_resolution = input_resolution


class IntermediateAttentionCollector(nn.Module):
    """Wrapper for intermediate ViT blocks to collect attention maps for RCS."""
    
    def __init__(self, original_block, block_idx: int, final_attention: ResCLIPAttention, dim: int):
        super().__init__()
        self.original_block = original_block
        self.block_idx = block_idx
        self.final_attention = final_attention
        
        # original attention forward to collect weights
        self.original_attn = original_block.attn
        self.original_attn_forward = self.original_attn.forward
        
        # Replace with collector
        self.original_attn.forward = self._collect_attention_forward
        self.original_block.attn.forward = self._collect_attention_forward

        num_heads = self.original_attn.num_heads
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        qk_norm = False
        self.qkv = self.original_attn.qkv
        self.q_norm = self.original_attn.q_norm if qk_norm else nn.Identity()
        self.k_norm = self.original_attn.k_norm if qk_norm else nn.Identity()
        self.attn_drop = self.original_attn.attn_drop
        self.proj = self.original_attn.proj
        self.proj_drop = self.original_attn.proj_drop
        self.device = 'cuda'
        self.num_prefix_tokens = 8
    
    def _collect_attention_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        x = x.permute(1, 0, 2)


        num_heads = self.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        attn = torch.bmm(q * scale, k.transpose(1, 2))
        attn = attn.softmax(dim=-1)
        
        self.final_attention.store_intermediate_attention(attn)
        
        attn = self.original_attn.attn_drop(attn)
        x = torch.bmm(attn, v).transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        x = self.original_attn.proj(x)
        x = self.original_attn.proj_drop(x)

        x = x.permute(1, 0, 2)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_block(x)


class ResRadioEncoder(LangSpatialGlobalImageEncoder):
    """ResCLIP-enhanced RADIO encoder.
    
    This encoder applies ResCLIP modifications RCS to AM-RADIO models.
    """
    
    def __init__(
        self,
        device: str = None,
        model_version: str = "radio_v2.5-b",
        lang_model: str = "siglip",
        input_resolution: Tuple[int, int] = [512, 512],
        gaussian_std: float = 5.0,
        use_rcs: bool = True,
        use_sfr: bool = True,
        rcs_layers: List[int] = None,
        attn_rcs_weights: Tuple[float, float] = (2.0, 0.6),
        attn_sfr_weights: Tuple[float, float] = (2.1, 0.6),
        temp_thd: float = 0.20,
        delete_same_entity: bool = True,
        return_radio_features: bool = True,
        compile: bool = False,
        amp: bool = False,
        use_sliding_window: bool = False,
        slide_crop_size: int = 336,
        slide_stride: int = 112
    ):
        """
        Args:
            device: "cpu" or "cuda", set to None to use CUDA if available.
            model_version: Choose from "radio_v2.5-x" where x can be b, l, or g.
            lang_model: choose from ["siglip", "clip"]
            input_resolution: Tuple of ints (height, width) of the input images.
            gaussian_std: Standard deviation of the gaussian kernel.
            use_rcs: Whether to use Residual Cross-correlation Self-attention.
            use_sfr: Whether to use Semantic Feedback Refinement.
            rcs_layers: Which intermediate layers to use for RCS (default: [5,6,7,8]).
            attn_rcs_weights: Weights for RCS attention blending (tau, lambda).
            attn_sfr_weights: Weights for SFR attention blending (tau, lambda).
            temp_thd: Temperature threshold for SFR.
            delete_same_entity: Whether to apply distance decay in SFR.
            return_radio_features: Whether to return raw RADIO features.
            compile: Whether to compile the model.
            amp: Whether to use automatic mixed precision.
            use_sliding_window: Whether to use sliding window inference for large images.
            slide_crop_size: Size of the sliding window crop (default 336).
            slide_stride: Stride of the sliding window (default 112).
        """
        
        super().__init__(device)
        
        self.compile = compile
        self.amp = amp
        self.model_version = model_version
        self.return_radio_features = return_radio_features
        self.use_rcs = use_rcs
        self.use_sfr = use_sfr
        self.use_sliding_window = use_sliding_window
        self.slide_crop_size = slide_crop_size
        self.slide_stride = slide_stride
        
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model",
            version=model_version, 
            progress=True,
            skip_validation=True,
            adaptor_names=[lang_model]
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.make_preprocessor_external()
        
        self.lang_adaptor = self.model.adaptors[lang_model]
        self.model.adaptors = None

        if self.use_sliding_window:
            init_resolution = [int(self.slide_crop_size), int(self.slide_crop_size)]
        else:
            init_resolution = input_resolution

        self._apply_resclip_modifications(
            input_resolution=init_resolution,
            gaussian_std=gaussian_std,
            rcs_layers=rcs_layers,
            attn_rcs_weights=attn_rcs_weights,
            attn_sfr_weights=attn_sfr_weights,
            temp_thd=temp_thd,
            delete_same_entity=delete_same_entity
        )
        
        if self.compile:
            self.model.compile(fullgraph=True, options={"triton.cudagraphs": True})
            self.lang_adaptor.compile(fullgraph=True, options={"triton.cudagraphs": True})
    
    def _apply_resclip_modifications(
        self,
        input_resolution: Tuple[int, int],
        gaussian_std: float,
        rcs_layers: List[int],
        attn_rcs_weights: Tuple[float, float],
        attn_sfr_weights: Tuple[float, float],
        temp_thd: float,
        delete_same_entity: bool
    ):
        
        # Replace the last attention layer with ResCLIP attention
        last_block = self.model.model.blocks[-1]
        resclip_attention = ResCLIPAttention(
            orig_attn=last_block.attn,
            input_resolution=input_resolution,
            gaussian_std=gaussian_std,
            device=self.device,
            dim=self.model.model.embed_dim,
            num_prefix_tokens=self.model.num_summary_tokens,
            use_rcs=self.use_rcs,
            use_sfr=self.use_sfr,
            rcs_layers=rcs_layers,
            attn_rcs_weights=attn_rcs_weights,
            attn_sfr_weights=attn_sfr_weights,
            temp_thd=temp_thd,
            delete_same_entity=delete_same_entity
        )
        last_block.attn = resclip_attention
        
        # Wrapping intermediate blocks to collect attention maps for RCS
        if self.use_rcs:
            for i, block in enumerate(self.model.model.blocks[:-1]):  # All except last
                wrapped_block = IntermediateAttentionCollector(
                    original_block=block,
                    block_idx=i,
                    final_attention=resclip_attention,
                    dim=self.model.model.embed_dim
                )
                self.model.model.blocks[i] = wrapped_block
    
    @property
    def input_resolution(self):
        return self.model.model.blocks[-1].attn.input_resolution
    
    @input_resolution.setter
    def input_resolution(self, value: Tuple[int, int]):
        if hasattr(value, "__len__") and len(value) == 2:
            if self.is_compatible_size(*value):
                self.model.model.blocks[-1].attn.update_input_resolution(value)
                self.model.model.blocks[-1].attn.clear_intermediate_attentions()
                
                if self.compile:
                    self.model.compile(fullgraph=True, options={"triton.cudagraphs": True})
            else:
                raise ValueError(f"Incompatible input resolution {value}")
        else:
            raise ValueError("Input resolution must be a tuple of two ints")
    
    def _clear_attention_cache(self):
        if hasattr(self.model.model.blocks[-1].attn, 'clear_intermediate_attentions'):
            self.model.model.blocks[-1].attn.clear_intermediate_attentions()
    
    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        prompts_per_label = self.insert_labels_into_templates(labels)
        all_text_features = []
        
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
        return text_features
    

    
    @override
    def encode_image_to_vector(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        self._clear_attention_cache()
        
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(rgb_image)
            C = out.summary.shape[-1] // 3
            i = self.lang_adaptor.head_idx
            out = out.summary[:, C*i: C*(i+1)]
            
            if not self.return_radio_features:
                out = self.lang_adaptor.head_mlp(out)
        
        return out
    
    @override
    def encode_image_to_feat_map(self, rgb_image: torch.FloatTensor) -> torch.FloatTensor:
        self._clear_attention_cache()
        # Use sliding window inference if enabled and image is large
        if self.use_sliding_window:
            return self._encode_image_to_feat_map_sliding(rgb_image)
        
        B, C, H, W = rgb_image.shape
        H_, W_ = H // self.model.patch_size, W // self.model.patch_size
        
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(rgb_image).features
            if not self.return_radio_features:
                out = self.lang_adaptor.head_mlp(out)
        
        return out.permute(0, 2, 1).reshape(B, -1, H_, W_)
        
    
    @override
    def encode_image_to_feat_map_and_vector(self, rgb_image: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        self._clear_attention_cache()
        
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
        
        feat_map = feat_map.permute(0, 2, 1).reshape(B, -1, H_, W_)
        
        # If sliding window is enabled, re-encode feat_map with sliding window
        if self.use_sliding_window:
            feat_map = self._encode_image_to_feat_map_sliding(rgb_image)
        
        return feat_map, global_vector
    
    @override
    def align_global_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
        if self.lang_adaptor is None:
            raise ValueError("Cannot align to language without a lang model")
        if not self.return_radio_features:
            return features
        
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self.lang_adaptor.head_mlp(features)
    
    @override
    def align_spatial_features_with_language(self, features: torch.FloatTensor) -> torch.FloatTensor:
        if self.lang_adaptor is None:
            raise ValueError("Cannot align to language without a lang model")
        if not self.return_radio_features:
            return features
        
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.lang_adaptor.head_mlp(features)
        
        return out.permute(0, 2, 1).reshape(B, -1, H, W)
    
    @override
    def is_compatible_size(self, h: int, w: int) -> bool:
        hh, ww = self.get_nearest_size(h, w)
        return hh == h and ww == w
    
    @override
    def get_nearest_size(self, h: int, w: int) -> Tuple[int, int]:
        return self.model.get_nearest_supported_resolution(h, w)
    
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
        self._clear_attention_cache()
        return feat_map

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
    
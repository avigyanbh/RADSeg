from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing_extensions import override
import sys
import os

textregion_dir = os.path.dirname(os.path.abspath(__file__))
if textregion_dir not in sys.path:
    sys.path.insert(0, textregion_dir)

from sam2.build_sam import build_sam2
from sam2.custom_automatic_mask_generator import CustomAutomaticMaskGenerator
import custom_clip
from custom_open_clip import create_model, tokenizer
from custom_clip.clip import tokenize
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from rayfronts.image_encoders.prompt_templates import openai_imagenet_template
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from rayfronts.image_encoders.base import LangSpatialImageEncoder


class TextRegionFeatsEncoder(LangSpatialImageEncoder):
    """TextRegion encoder
    """

    def __init__(
        self,
        device: str = None,
        classes: List[str] = None,
        clip_pretrained: str = 'openai',
        clip_architecture: str = 'ViT-B/16',
        sam2_checkpoint: str = None,
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_bplus.yaml",
        clip_download_root: str = None,
        points_per_side: int = 16,
        dtype: str = 'fp32',
        resize_method: str = 'multi_resolution',
        crop_size: int = 336,
        remove_global_patch: bool = True,
        global_patch_threshold: float = 0.07,
        region_logit_scale: float = 50.0,
    ):
        """Initialize the TextRegion encoder.

        Args:
            device: Device to run the model on ("cpu" or "cuda"). If None, uses CUDA if available.
            classes: List of class names for segmentation. If None, must be set later.
            clip_pretrained: CLIP model variant. Choose from 'openai', 'meta', 'siglip2'.
            clip_architecture: CLIP architecture. E.g., 'ViT-B/16', 'ViT-L/14@336px', 'PE-Core-L14-336', 
                               'ViT-L-16-SigLIP2-256'.
            sam2_checkpoint: Path to SAM2 checkpoint file.
            sam2_model_cfg: Path to SAM2 model configuration file.
            clip_download_root: Root directory for downloading CLIP models.
            points_per_side: Number of points per side for SAM2 mask generation.
            dtype: Data type for computation ('fp32' or 'bf16').
            resize_method: Image resizing method ('multi_resolution' or 'resize').
            crop_size: Size for cropping images.
            remove_global_patch: Whether to remove global patches based on threshold.
            global_patch_threshold: Threshold for removing global patches.
            region_logit_scale: Scale factor for region logits.
        """
        super().__init__(device)
        
        if dtype == "fp32":
            self.dtype = torch.float32
        elif dtype == "bf16":
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        print(f"Using device: {self.device}, dtype: {self.dtype}")
        
        self.clip_pretrained = clip_pretrained
        self.resize_method = resize_method
        self.crop_size = crop_size
        self.remove_global_patch = remove_global_patch
        self.global_patch_threshold = global_patch_threshold
        self.region_logit_scale = region_logit_scale
        self.points_per_side = points_per_side
        self.encode_global_image = True
        
        self.classes = classes
        if self.classes is not None:
            self.class_mappings()
        else:
            self._cat_index_to_name = {0: ''}  # Ignore class
            self._cat_name_to_index = {'': 0}
        
        self.init_clip_model(clip_architecture, clip_download_root)
        
        if sam2_checkpoint is not None:
            self.init_sam2_model(sam2_model_cfg, sam2_checkpoint)
        else:
            print("Warning: SAM2 checkpoint not provided. SAM2 will not be initialized.")
            self.sam2_generator = None
        
        # if self.classes is not None:
        #     self._encode_labels()
    
    def class_mappings(self):
        # Index 0 is reserved for ignore/background class
        self._cat_index_to_name = {0: ''}
        self._cat_index_to_name.update({i+1: name for i, name in enumerate(self.classes)})
        self._cat_name_to_index = {name: idx for idx, name in self._cat_index_to_name.items()}
    
    def _encode_labels(self):
        with torch.no_grad():
            all_text_features = []
            for label in self.classes:
                if self.clip_pretrained in ['meta', 'siglip2']:
                    label_prompts = self.tokenizer(
                        [temp(label) for temp in openai_imagenet_template]
                    ).to(self.device)
                else:
                    label_prompts = tokenize(
                        [temp(label) for temp in openai_imagenet_template]
                    ).to(self.device)
                
                feature = self.clip.encode_text(label_prompts)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                all_text_features.append(feature.unsqueeze(0))
            
            self.text_features = torch.cat(all_text_features, dim=0)

    @override
    def encode_labels(self, labels: List[str]) -> torch.FloatTensor:
        self.classes = labels
        self._encode_labels()
        return self.text_features.float()
    
    @override
    def encode_prompts(self, prompts: List[str]) -> torch.FloatTensor:
        pass
    
    @property
    @override
    def num_classes(self) -> int:
        if self.classes is None:
            return 1  # Just ignore class
        return len(self.classes) + 1  # +1 for ignore class
    
    @property
    def cat_index_to_name(self):
        return self._cat_index_to_name
    
    @property
    def cat_name_to_index(self):
        return self._cat_name_to_index
    
    def init_clip_model(self, clip_architecture: str, clip_download_root: str):
        clip_preprocess = v2.Compose([
            v2.Resize((self.crop_size, self.crop_size), interpolation=Image.BICUBIC),
            v2.ToDtype(self.dtype, scale=True),
            v2.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        
        if self.clip_pretrained == 'openai':
            self.clip, _ = custom_clip.load(
                clip_architecture, 
                device=self.device, 
                jit=False, 
                download_root=clip_download_root
            )
            self.patch_size = self.clip.visual.patch_size
            
        elif self.clip_pretrained == 'meta':
            self.clip = pe.CLIP.from_config(clip_architecture, pretrained=True)
            self.clip.eval().to(self.device)
            self.patch_size = self.clip.visual.patch_size
            self.tokenizer = transforms.get_text_tokenizer(self.clip.context_length)
            self.crop_size = self.clip.visual.image_size
            
            clip_preprocess = v2.Compose([
                v2.Resize((self.crop_size, self.crop_size), interpolation=InterpolationMode.BILINEAR),
                v2.ToDtype(self.dtype, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
        elif self.clip_pretrained == 'siglip2':
            import open_clip
            model, _, clip_preprocess = open_clip.create_model_and_transforms(
                clip_architecture, pretrained='webli'
            )
            model = model.to(device=self.device, dtype=self.dtype)
            self.tokenizer = open_clip.get_tokenizer(clip_architecture)
            self.text_model = model.text
            self.patch_size = model.visual.trunk.patch_embed.patch_size[0]
            self.clip = model.eval()
            self.crop_size = model.visual.image_size[0]
            
            clip_preprocess = v2.Compose([
                v2.Resize((self.crop_size, self.crop_size), interpolation=InterpolationMode.BICUBIC),
                v2.ToDtype(self.dtype, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.clip = create_model(
                clip_architecture, 
                pretrained=self.clip_pretrained, 
                precision=self.dtype, 
                cache_dir=clip_download_root
            )
            self.clip.eval().to(self.device)
            self.patch_size = self.clip.visual.patch_size[0]
        
        self.clip_preprocess = clip_preprocess
    
    def init_sam2_model(self, model_cfg: str, sam2_checkpoint: str):
        device_obj = torch.device(self.device) if isinstance(self.device, str) else self.device
        
        if device_obj.type == "cuda":
            if torch.cuda.get_device_properties(device_obj).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        sam2_model = build_sam2(
            model_cfg, 
            sam2_checkpoint, 
            device=self.device, 
            apply_postprocessing=False
        )
        self.sam2_generator = CustomAutomaticMaskGenerator(
            prompt_method="grid",
            model=sam2_model,
            point_grids=None,
            min_mask_region_area=0,
            points_per_side=self.points_per_side,
            points_per_batch=2048,
            pred_iou_thresh=0.6,
            stability_score_thresh=0.6,
            box_nms_thresh=0.9,
            multimask_output=True,
            fuse_mask=True,
            fuse_mask_threshold=0.8,
        )
        self.sam_transform = self.sam2_generator.predictor._transforms
        print("SAM2 model loaded successfully")
    
    def prepare_clip_inputs(
        self, 
        rgb_image: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, int, int, int, int]:
        """
        Prepare CLIP inputs with multi-resolution support.
        """
        B, C, H, W = rgb_image.shape
        
        if self.resize_method == 'multi_resolution':
            clip_inputs = []
            if self.encode_global_image:
                for b in range(B):
                    clip_inputs.append(self.clip_preprocess(rgb_image[b]))
            
            crop_num_h = max(H // self.crop_size, 1)
            crop_num_w = max(W // self.crop_size, 1)
            points_per_w = (self.crop_size // self.patch_size) * crop_num_w
            points_per_h = (self.crop_size // self.patch_size) * crop_num_h
            crop_size_h = int(np.ceil(H / crop_num_h))
            crop_size_w = int(np.ceil(W / crop_num_w))
            
            for b in range(B):
                for h_idx in range(crop_num_h):
                    for w_idx in range(crop_num_w):
                        y1 = h_idx * crop_size_h
                        x1 = w_idx * crop_size_w
                        y2 = min(y1 + crop_size_h, H)
                        x2 = min(x1 + crop_size_w, W)
                        y1 = max(y2 - crop_size_h, 0)
                        x1 = max(x2 - crop_size_w, 0)
                        crop_img = rgb_image[b, :, y1:y2, x1:x2]
                        clip_inputs.append(self.clip_preprocess(crop_img))
            
            clip_inputs = torch.stack(clip_inputs).to(self.device)
        else:
            points_per_w = W // self.patch_size
            points_per_h = H // self.patch_size
            crop_num_h = 1
            crop_num_w = 1
            clip_inputs = torch.stack([self.clip_preprocess(rgb_image[b]) for b in range(B)])
        
        return clip_inputs, points_per_h, points_per_w, crop_num_h, crop_num_w
    
    def generate_sam2_masks(
        self, 
        rgb_image: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, Tuple[int, int]]:
        if self.sam2_generator is None:
            raise ValueError("SAM2 generator not initialized. Provide sam2_checkpoint.")
        
        B, C, H, W = rgb_image.shape
        ori_shape = (H, W)
        
        img_tensor = (rgb_image * 255.0).to(dtype=torch.float32)
        image_tensor_for_sam2 = img_tensor.permute(0, 2, 3, 1)
        image_tensor_for_sam2 = self.sam_transform(image_tensor_for_sam2)
        
        with torch.inference_mode(), torch.autocast(
            "cuda", 
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        ):
            sam2_masks_list = self.sam2_generator.generate_for_batch(
                image_tensor_for_sam2, 
                [ori_shape] * B, 
                None
            )
        
        if B == 1:
            unique_masks = torch.stack([mask['segmentations'] for mask in sam2_masks_list[0]])
            return unique_masks, ori_shape
        else:
            raise NotImplementedError("Batch processing not yet supported for SAM2")
    
    def compute_region_features_clip(
        self,
        clip_value: torch.FloatTensor,
        low_res_masks: torch.FloatTensor,
        blk,
        points_per_h: int,
        points_per_w: int,
        crop_num_h: int,
        crop_num_w: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        attn_layer = blk.attn
        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = clip_value.size()
        head_dim = embed_dim // num_heads
        
        x = blk.ln_1(clip_value)
        q, k, v_ori = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            v = v_ori[1:, :, :].permute(1, 2, 0).contiguous().view(bsz, embed_dim, patch_num, patch_num)
            
            if self.encode_global_image:
                crop_id = 1
                v_multi_reso = F.interpolate(v[:1], [points_per_h, points_per_w], mode="bilinear")
            else:
                crop_id = 0
                v_multi_reso = torch.zeros(
                    1, embed_dim, points_per_h, points_per_w, 
                    device=self.device, dtype=self.dtype
                )
            
            for h_idx in range(crop_num_h):
                for w_idx in range(crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num
                    
                    if self.encode_global_image:
                        v_multi_reso[:, :, y1:y2, x1:x2] = v_multi_reso[:, :, y1:y2, x1:x2] + v[crop_id]
                    else:
                        v_multi_reso[:, :, y1:y2, x1:x2] = v[crop_id]
                    crop_id += 1
            
            v_single_head = v_multi_reso.contiguous().view(1, embed_dim, points_per_h * points_per_w)
            v_multi_head = v_single_head.contiguous().view(
                1 * num_heads, head_dim, points_per_h * points_per_w
            ).permute(0, 2, 1)
        else:
            v_single_head = v_ori[1:].permute(1, 2, 0)
            v_multi_head = v_ori[1:].contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        if self.remove_global_patch:
            patch_features = v_single_head.permute(0, 2, 1)[0].to(low_res_masks.dtype)
            patch_features /= patch_features.norm(dim=-1, keepdim=True)
            patch_similarity = patch_features @ patch_features.T
            
            patch_2_region = patch_similarity @ (low_res_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (low_res_masks > 0).sum(dim=-1)
            
            belong_score = patch_2_region_avg * (low_res_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((low_res_masks > 0).sum(dim=0) + 1e-9)
            
            outside_score = patch_2_region_avg * (low_res_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((low_res_masks == 0).sum(dim=0) + 1e-9)
            
            difference_score = (belong_score_avg - outside_score_avg).cpu().detach().float().numpy()
            low_res_masks[:, difference_score < self.global_patch_threshold] = 0
        
        keep_masks = torch.sum(low_res_masks, dim=1) > 0
        low_res_masks = low_res_masks[keep_masks]
        
        attn_weights = low_res_masks.unsqueeze(0).repeat(num_heads, 1, 1)
        attn_weights = attn_weights.to(dtype=v_multi_head.dtype)
        
        attn_output = torch.bmm(attn_weights, v_multi_head)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, 1, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        attn_output += blk.mlp(blk.ln_2(attn_output))
        region_features = attn_output.permute(1, 0, 2)
        
        region_features = self.clip.visual.ln_post(region_features) @ self.clip.visual.proj
        region_features /= region_features.norm(dim=-1, keepdim=True)
        
        return region_features, keep_masks
    
    def compute_region_features_pe(
        self,
        pe_value: torch.FloatTensor,
        low_res_masks: torch.FloatTensor,
        blk,
        points_per_h: int,
        points_per_w: int,
        crop_num_h: int,
        crop_num_w: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        if self.clip.visual.use_cls_token:
            pe_value = pe_value[:, 1:]
        
        bsz, _, embed_dim = pe_value.shape
        
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            x_ori = pe_value.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)
            
            crop_id = 1
            x_multi_reso = F.interpolate(x_ori[:1], [points_per_h, points_per_w], mode="bilinear")
            for h_idx in range(crop_num_h):
                for w_idx in range(crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num
                    
                    x_multi_reso[:, :, y1:y2, x1:x2] = 0.5 * x_multi_reso[:, :, y1:y2, x1:x2] + x_ori[crop_id]
                    crop_id += 1
            
            x_input = x_multi_reso.contiguous().view(1, embed_dim, points_per_h * points_per_w).permute(0, 2, 1)
        else:
            x_input = pe_value
        
        if self.remove_global_patch:
            patch_norm = x_input.norm(dim=-1, keepdim=True)
            patch_features = (x_input / patch_norm)[0]
            patch_similarity = patch_features @ patch_features.T
            
            patch_2_region = patch_similarity @ (low_res_masks > 0).float().T
            patch_2_region_avg = patch_2_region / (low_res_masks > 0).sum(dim=-1)
            
            belong_score = patch_2_region_avg * (low_res_masks > 0).float().T
            belong_score_avg = belong_score.sum(dim=-1) / ((low_res_masks > 0).sum(dim=0) + 1e-9)
            
            outside_score = patch_2_region_avg * (low_res_masks == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((low_res_masks == 0).sum(dim=0) + 1e-9)
            
            difference_score = (belong_score_avg - outside_score_avg).cpu().detach().float().numpy()
            low_res_masks[:, difference_score < self.global_patch_threshold] = 0
        
        keep_masks = torch.sum(low_res_masks, dim=1) > 0
        low_res_masks = low_res_masks[keep_masks]
        batch = low_res_masks.shape[0]
        
        assert x_input.shape[0] == 1 or x_input.shape[0] == low_res_masks.shape[0]
        if x_input.shape[0] == 1:
            x = x_input.repeat(batch, 1, 1)
        else:
            x = x_input
        
        q = blk.probe.repeat((batch, 1, 1)).to(x.dtype)
        k = blk.layernorm(x.mean(dim=-2, keepdim=True))
        k = k.repeat(1, x.shape[-2], 1).to(x.dtype)
        x = blk.attn(q, k, x, need_weights=False, key_padding_mask=low_res_masks<=0)[0]
        
        with torch.no_grad():
            region_features = x @ self.clip.visual.proj
        region_features = F.normalize(region_features, dim=-1)
        
        return region_features, keep_masks
    
    def compute_region_features_siglip(
        self,
        siglip_value: torch.FloatTensor,
        low_res_masks: torch.FloatTensor,
        attn_blk,
        points_per_h: int,
        points_per_w: int,
        crop_num_h: int,
        crop_num_w: int,
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        bsz, _, embed_dim = siglip_value.shape
        
        if self.resize_method == 'multi_resolution':
            patch_num = self.crop_size // self.patch_size
            x_ori = siglip_value.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)
            
            crop_id = 1
            x_multi_reso = F.interpolate(x_ori[:1], [points_per_h, points_per_w], mode="bilinear")
            for h_idx in range(crop_num_h):
                for w_idx in range(crop_num_w):
                    y1 = h_idx * patch_num
                    x1 = w_idx * patch_num
                    y2 = y1 + patch_num
                    x2 = x1 + patch_num
                    
                    x_multi_reso[:, :, y1:y2, x1:x2] = 0.5 * x_multi_reso[:, :, y1:y2, x1:x2] + x_ori[crop_id]
                    crop_id += 1
            
            x_input = x_multi_reso.contiguous().view(1, embed_dim, crop_num_h * crop_num_w * patch_num ** 2).permute(0, 2, 1)
        else:
            x_input = siglip_value
        
        if self.remove_global_patch:
            keep_masks = torch.sum(low_res_masks, dim=1) > 0
            low_res_mask = low_res_masks[keep_masks]
            
            patch_norm = x_input.norm(dim=-1, keepdim=True)
            patch_features = (x_input / patch_norm)[0]
            patch_similarity = (patch_features @ patch_features.T).float()
            
            patch_2_region = patch_similarity @ (low_res_mask > 0).float().T
            patch_2_region_avg = patch_2_region / (low_res_mask > 0).sum(dim=-1)
            
            blong_score = patch_2_region_avg * (low_res_mask > 0).float().T
            blong_score_avg = blong_score.sum(dim=-1) / ((low_res_mask > 0).sum(dim=0) + 1e-9)
            
            outside_score = patch_2_region_avg * (low_res_mask == 0).float().T
            outside_score_avg = outside_score.sum(dim=-1) / ((low_res_mask == 0).sum(dim=0) + 1e-9)
            
            difference_score = (blong_score_avg - outside_score_avg).cpu().detach().float().numpy()
            low_res_masks[:, difference_score < self.global_patch_threshold] = 0
        
        keep_masks = torch.sum(low_res_masks, dim=1) > 0
        low_res_masks = low_res_masks[keep_masks]
        low_res_masks = torch.clamp(low_res_masks, min=0, max=1)
        
        assert x_input.shape[0] == 1
        region_num = low_res_masks.shape[0]
        
        _, N, C = x_input.shape
        q_latent = attn_blk.latent.expand(region_num, -1, -1)
        q = attn_blk.q(q_latent).reshape(
            region_num, attn_blk.latent_len, attn_blk.num_heads, attn_blk.head_dim
        ).transpose(1, 2)
        
        x = x_input.expand(region_num, -1, -1)
        kv = attn_blk.kv(x).reshape(
            region_num, N, 2, attn_blk.num_heads, attn_blk.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = attn_blk.q_norm(q), attn_blk.k_norm(k)
        
        attn_mask = low_res_masks.unsqueeze(1).unsqueeze(1).repeat(1, attn_blk.num_heads, 1, 1)
        
        k = attn_blk.k_norm(k.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True))
        k = k.repeat(1, 1, v.shape[-2], v.shape[-1])
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask > 0)
        
        x = x.transpose(1, 2).reshape(region_num, attn_blk.latent_len, C)
        x = attn_blk.proj(x)
        x = attn_blk.proj_drop(x)
        
        x = self.clip.visual.trunk.fc_norm(x)
        x = self.clip.visual.trunk.head_drop(x)
        
        region_features = x.permute(1, 0, 2)
        region_features /= region_features.norm(dim=-1, keepdim=True)
        
        return region_features, keep_masks
    
    @override
    def encode_image_to_feat_map(
        self, 
        rgb_image: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Encodes image to segmentation logits.
        
        It combines SAM2 masks with region-text similarity to output class probabilities at each spatial location.
        
        Note:
            Due to SAM2's region-based processing (which generates a variable number of masks per image),
            this encoder currently only supports batch_size=1. For batch processing, call this function
            in a loop for each image.
        
        Args:
            rgb_image: Input RGB image tensor of shape (B, C, H, W) in [0, 1] range.
                       Currently only B=1 is supported.
            
        Returns:
            Segmentation logits of shape (B, num_classes+1, H, W) where each spatial
            location contains soft class assignments. Class 0 is the ignore/background class.
        """
        if self.classes is None:
            raise ValueError("Classes must be set before encoding. Provide 'classes' during initialization.")
        
        B, C, H, W = rgb_image.shape
        
        if B != 1:
            raise NotImplementedError(
                f"TextRegion encoder only supports batch_size=1 due to variable number of SAM2 regions. "
                f"Got batch_size={B}. Please process images one at a time."
            )
        
        # Prepare CLIP inputs
        clip_inputs, points_per_h, points_per_w, crop_num_h, crop_num_w = \
            self.prepare_clip_inputs(rgb_image)
        
        # Generate SAM2 masks
        unique_masks, ori_shape = self.generate_sam2_masks(rgb_image)
        
        # Encode image with CLIP
        clip_inputs = clip_inputs.to(self.device, dtype=self.clip.visual.proj.dtype)
        
        if self.clip_pretrained == 'meta':
            pe_last_blk_value, pe_last_blk = self.clip.encode_image(
                clip_inputs, return_value=True, region_attn_mask=None
            )
        elif self.clip_pretrained == 'siglip2':
            siglip_last_blk_value, intermediates = \
                self.clip.visual.trunk.forward_intermediates(clip_inputs)
            siglip_last_blk = self.clip.visual.trunk.attn_pool
        else:
            clip_last_blk_value, clip_last_blk = self.clip.encode_image(
                clip_inputs, return_value=True
            )
        
        # Downsample masks to feature map resolution
        unique_masks = unique_masks.to(self.device, dtype=self.dtype)
        unique_low_res_masks = F.interpolate(
            unique_masks.unsqueeze(0), 
            [points_per_h, points_per_w], 
            mode="bilinear"
        )
        unique_low_res_masks = unique_low_res_masks.reshape(-1, points_per_h * points_per_w)
        unique_low_res_masks = torch.clamp(unique_low_res_masks, min=0, max=1)
        
        keep_masks = torch.sum(unique_low_res_masks, dim=1) > 0
        unique_low_res_masks = unique_low_res_masks[keep_masks]
        unique_masks = unique_masks[keep_masks]
        
        # Compute region features
        if self.clip_pretrained == 'openai':
            region_features, keep_masks = self.compute_region_features_clip(
                clip_last_blk_value, unique_low_res_masks, clip_last_blk,
                points_per_h, points_per_w, crop_num_h, crop_num_w
            )
        elif self.clip_pretrained == 'meta':
            region_features, keep_masks = self.compute_region_features_pe(
                pe_last_blk_value, unique_low_res_masks, pe_last_blk,
                points_per_h, points_per_w, crop_num_h, crop_num_w
            )
            region_features = region_features.permute(1, 0, 2)
        elif self.clip_pretrained == 'siglip2':
            region_features, keep_masks = self.compute_region_features_siglip(
                siglip_last_blk_value, unique_low_res_masks, siglip_last_blk,
                points_per_h, points_per_w, crop_num_h, crop_num_w
            )
        else:
            raise NotImplementedError(f"Feature extraction for {self.clip_pretrained} not yet implemented")
        
        unique_masks = unique_masks[keep_masks]
        
        # Compute region-text similarity logits
        # region_features: (1, R, D), text_features: (num_classes, D)
        # if self.clip_pretrained == 'siglip2':
        #     region_logits = (
        #         torch.matmul(self.text_features, region_features[0].t()) * self.clip.logit_scale.exp()
        #         + self.clip.logit_bias
        #     ).t()  # (R, num_classes)
        # else:
        #     region_logits = region_features[0] @ self.text_features.T  # (R, num_classes)
        
        unique_masks = torch.clamp(unique_masks, min=0, max=1)
        
        # Broadcast region logits to spatial locations: (R, num_classes, H, W)
        # region_feats: (1, R, D) -> (R, D, 1, 1)
        # unique_masks: (R, H, W) -> (R, 1, H, W)
        # seg_feats: (R, D, H, W)
        
        #seg_feats = region_features.squeeze(0).unsqueeze(-1).unsqueeze(-1) * unique_masks.unsqueeze(1)
        #import pdb; pdb.set_trace()
        region_features = region_features.squeeze(0)
        seg_feats = torch.zeros(region_features.shape[1], unique_masks.shape[1], unique_masks.shape[2], device=self.device, dtype=region_features.dtype)
        for r in range(region_features.shape[0]):
            mask = unique_masks[r]
            feat = region_features[r]
            seg_feats[:, mask.bool()] = feat.view(region_features.shape[1], 1)
        
        # Sum across regions: (D, H, W)
        #seg_feats = seg_feats.sum(0)
        
        # # Add background/ignore class at index 0
        # ignore_class_logits = torch.zeros(1, H, W, device=self.device, dtype=seg_logits.dtype)
        # seg_logits = torch.cat([ignore_class_logits, seg_logits], dim=0)  # (num_classes+1, H, W)
        
        # # Apply softmax with scaling
        # seg_logits = torch.softmax(seg_logits * self.region_logit_scale, dim=0)
        
        # Add batch dimension
        seg_feats = seg_feats.unsqueeze(0)  # (1, D, H, W)
        
        return seg_feats.float()
    
    @override
    def align_spatial_features_with_language(self, features: torch.FloatTensor):
        return features.float()
    
    @override
    def get_nearest_size(self, h: int, w: int) -> Tuple[int, int]:
        nearest_h = int(np.round(h / self.patch_size) * self.patch_size)
        nearest_w = int(np.round(w / self.patch_size) * self.patch_size)
        return nearest_h, nearest_w
    
    @override
    def is_compatible_size(self, h: int, w: int) -> bool:
        hh, ww = self.get_nearest_size(h, w)
        return hh == h and ww == w


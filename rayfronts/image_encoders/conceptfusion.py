"""Includes the ConceptFusion encoder https://github.com/concept-fusion/concept-fusion

Typical Usage:

  rgb_img = torchvision.io.read_image(rgb_path)
  rgb_img = rgb_img.float() / 255
  rgb_img = torch.nn.functional.interpolate(
    rgb_img.unsqueeze(0), size=(512, 512))

  labels = ["car", "person"]

  enc = ConceptFusionEncoder(device="cuda", sam_model="vit_h", sam_model_path="sam_vit_h_4b8939.pth")
  
  feat_map = enc.encode_image_to_feat_map(rgb_img)
  lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

  text_features = enc.encode_labels(labels)

  from rayfronts.utils import compute_cos_sim
  r = compute_cos_sim(text_features, lang_aligned_feat_map, softmax=True)
"""

from typing_extensions import override, Tuple, List
from PIL import Image

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import open_clip

from rayfronts.image_encoders.base import LangSpatialGlobalImageEncoder

class ConceptFusionEncoder(LangSpatialGlobalImageEncoder):


  def __init__(self, device = None,
               sam_model = "vit_h",
               sam_model_path = "sam_vit_h_4b8939.pth",
               open_clip_model = "ViT-H-14",
               open_clip_pretrained_dataset = "laion2b_s32b_b79k"):
    super().__init__(device)
    self.sam = sam_model_registry[sam_model](checkpoint=sam_model_path)
    self.sam.to(device=device)
    self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=6,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=25,
    )

    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        open_clip_model, open_clip_pretrained_dataset, device=self.device
    )
    self.model.eval()

    self.tokenizer = open_clip.get_tokenizer(open_clip_model)

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
    text = self.tokenizer(prompts).to(self.device)
    text_features = self.model.encode_text(text)
    text_features = torch.nn.functional.normalize(text_features, dim=-1).float()
    return text_features

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

    feat_map = []
    feat_vector = []

    for idx in range(B):
      img = rgb_image[idx]
      img = img.permute(1, 2, 0)
      img = (img * 255).clamp(0, 255).byte()
      img = img.cpu().numpy()
      masks = self.mask_generator.generate(img)
      img_pil = Image.fromarray(img)
      
      global_feat = None
      with torch.cuda.amp.autocast():
          _img = self.preprocess(img_pil).unsqueeze(0).to(self.device)
          global_feat = self.model.encode_image(_img)
          global_feat /= global_feat.norm(dim=-1, keepdim=True)
      global_feat = global_feat.half().to(self.device)
      global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
      feat_dim = global_feat.shape[-1]
      cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

      feat_per_roi = []
      roi_nonzero_inds = []
      similarity_scores = []
      for maskidx in range(len(masks)):
          _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
          seg = masks[maskidx]["segmentation"]
          nonzero_inds = torch.argwhere(torch.from_numpy(masks[maskidx]["segmentation"]))
          # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
          img_roi = img[_y : _y + _h, _x : _x + _w, :]
          img_roi = self.preprocess(Image.fromarray(img_roi)).unsqueeze(0).to(self.device)
          roifeat = self.model.encode_image(img_roi)
          roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
          feat_per_roi.append(roifeat)
          roi_nonzero_inds.append(nonzero_inds)
          _sim = cosine_similarity(global_feat, roifeat)
          similarity_scores.append(_sim)

      similarity_scores = torch.cat(similarity_scores)
      softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
      outfeat = torch.zeros(H, W, feat_dim, dtype=torch.half).to(self.device)
      for maskidx in range(len(masks)):
          _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
          _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
          outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().half()
          outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
              outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
          ).half()

      outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
      outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
      outfeat = torch.nn.functional.interpolate(outfeat, [self.out_height, self.out_width], mode="nearest")
      outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
      outfeat = torch.nn.functional.normalize(outfeat, dim=-1)

      outfeat = outfeat.permute(0, 3, 1, 2).half()

      feat_map.append(outfeat)
      feat_vector.append(global_feat)

    feat_map = torch.cat(feat_map).float()
    feat_vector = torch.cat(feat_vector).float()

    return feat_map, feat_vector

  @override
  def align_global_features_with_language(self, features: torch.FloatTensor):
    return features

  @override
  def align_spatial_features_with_language(self, features: torch.FloatTensor):
    return features

  @override
  def is_compatible_size(self, h: int, w: int):
    return True

  @override
  def get_nearest_size(self, h, w):
    return h,w


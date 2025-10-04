"""A script to fit a feature compressor to a data distribution.

The script uses the default rayfronts configs. Particularly ones related to:
1. Feature Compressors
2. Encoder
3. Dataset
It also allows the user to choose an imagefolder instead of a dataset by setting
`dataset=imagefolder`. See the two dataclasses below for all additional options.

Typical Usage:
  python scripts/fit_feat_compressor.py \
    dataset=imagefolder dataset.path="..." rgb_resolution=[224,224] \
    encoder=naradio encoder.model="radio_v2.5-l" \
    feat_compressor=pca feat_compressor.out_dim=128
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import os
import sys
import tqdm

import torch
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

logger = logging.getLogger(__name__)

@dataclass
class ImageFolderDatasetConfig:
    _target_: str = "__main__.ImageFolderDataset"
    path: Optional[str] = None
    rgb_resolution: Tuple = (224, 224)

@dataclass
class FittingConfig:
  """Configuration for fitting feature compressors."""

  # Where to save the fitted feature compressor state. Set to None to save as
  # ./<encoder_name>_<dataset_name>_<H>x<W>_<feat_compressor_name>.pt
  out: Optional[str] = None

  # Rough percentage of the batches to use for fitting.
  batch_sampling: float = 0.5

  # Rough percentage of the features within a batch to use for fitting.
  feature_sampling: float = 0.1

cs = ConfigStore.instance()
cs.store(name="extras", node=FittingConfig)
cs.store(name="imagefolder", node=ImageFolderDatasetConfig, group="dataset")

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageFolderDataset(Dataset):
  def __init__(self, path, rgb_resolution, 
               extensions={'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}):
    self.path = path
    self.transform = transforms.Compose([
      transforms.Resize(rgb_resolution),
      transforms.ToTensor(),
    ])
    self.extensions = extensions
    self.rgb_h = rgb_resolution[0]
    self.rgb_w = rgb_resolution[1]

    self.image_paths = [
      os.path.join(path, fname)
      for fname in os.listdir(path)
      if os.path.splitext(fname)[1].lower() in self.extensions
    ]

  def __len__(self):
      return len(self.image_paths)

  def __getitem__(self, idx):
      img_path = self.image_paths[idx]
      image = Image.open(img_path).convert("RGB")
      if self.transform:
          image = self.transform(image)
      return {"rgb_img": image}


@hydra.main(version_base="1.2",
            config_path="../rayfronts/configs",
            config_name="default")
@torch.inference_mode()
def main(cfg=None):
  logger.info("Initializing..")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = hydra.utils.instantiate(cfg.dataset)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
  feat_compressor = hydra.utils.instantiate(cfg.feat_compressor)
  encoder_kwargs = dict()
  if "NARadioEncoder" in cfg.encoder._target_:
    encoder_kwargs["input_resolution"] = (dataset.rgb_h, dataset.rgb_w)
  encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
  
  features = list()
  logger.info("Computing features from data..")
  for batch in tqdm.tqdm(dataloader):
    if np.random.rand() > cfg.batch_sampling:
      continue

    rgb_img = batch["rgb_img"].to(device)
    feat_img = encoder.encode_image_to_feat_map(rgb_img)
    B, FC, FH, FW = feat_img.shape
    feat_img_flat = feat_img.permute(0, 2, 3, 1).reshape(-1, FC)
    k = int(cfg.feature_sampling * feat_img_flat.size(0))
    perm = torch.randperm(feat_img_flat.size(0), device=device)
    idx = perm[:k] 
    sampled_features = feat_img_flat[idx, :]
    features.append(sampled_features)

  features = torch.cat(features, dim=0)
  logger.info("Fitting feature compressor..")
  feat_compressor.fit(features)
  fn = cfg.out
  if fn is None:
     fn = f"{encoder.__class__.__name__}_{dataset.__class__.__name__}_{dataset.rgb_h}x{dataset.rgb_w}_{feat_compressor.__class__.__name__}.pt"
  feat_compressor.save(fn)
  logger.info("Saved fitted feature compressor to %s.", fn)

if __name__ == "__main__":
  main()

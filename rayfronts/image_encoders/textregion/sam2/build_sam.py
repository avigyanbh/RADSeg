# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2
import sys
import importlib

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
# Modified to only warn since we're using this as a local package within textregion
try:
    if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
        # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
        # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
        # This typically happens because the user is running Python from the parent directory
        # that contains the sam2 repo they cloned.
        import warnings
        warnings.warn(
            "You're likely running Python from the parent directory of the sam2 repository. "
            "This might cause issues if you have both the repo and package named 'sam2'."
        )
except (AttributeError, IndexError):
    # sam2 might not have __path__ if it's being used as a local module
    pass


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def _try_load_cfg_from_filesystem(config_file):
    """
    Try to load a config YAML directly from the filesystem, avoiding reliance on
    Hydra's search path. Returns an OmegaConf object or None if not found.
    """
    # 1) Absolute path or relative to CWD
    if os.path.isfile(config_file):
        return OmegaConf.load(config_file)

    # 2) Relative to this module's directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(this_dir, config_file)
    if os.path.isfile(local_path):
        return OmegaConf.load(local_path)

    # 3) Common pattern: "sam2.1/sam2.1_hiera_bplus" or
    #    "configs/sam2.1/sam2.1_hiera_bplus.yaml"
    #    Normalize to module-local configs directory
    cfg_rel = config_file
    if not cfg_rel.endswith(".yaml"):
        cfg_rel_yaml = cfg_rel + ".yaml"
    else:
        cfg_rel_yaml = cfg_rel

    # If path already contains "configs/", try directly; otherwise, prefix it
    candidates = []
    if cfg_rel_yaml.startswith("configs/"):
        candidates.append(os.path.join(this_dir, cfg_rel_yaml))
    else:
        candidates.append(os.path.join(this_dir, "configs", cfg_rel_yaml))

    for cand in candidates:
        if os.path.isfile(cand):
            return OmegaConf.load(cand)

    return None


def _rewrite_targets_to_local_sam2(cfg_node):
    """Recursively rewrite Hydra _target_ paths from 'sam2.' to the local package
    path 'rayfronts.image_encoders.textregion.sam2.' so importlib resolves to the
    bundled SAM2, not any external package named 'sam2'."""
    local_prefix = "rayfronts.image_encoders.textregion.sam2."

    def transform(node):
        if isinstance(node, dict):
            # rewrite _target_ if needed
            tgt = node.get("_target_")
            if isinstance(tgt, str) and tgt.startswith("sam2."):
                node["_target_"] = local_prefix + tgt[len("sam2."):]
            # recurse
            for k, v in list(node.items()):
                node[k] = transform(v)
            return node
        elif isinstance(node, list):
            return [transform(x) for x in node]
        else:
            return node

    # Convert to plain container, transform, then back to OmegaConf
    plain = OmegaConf.to_container(cfg_node, resolve=False)
    plain = transform(plain)
    return OmegaConf.create(plain)


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Prefer loading config directly from filesystem to avoid Hydra search-path issues
    cfg = _try_load_cfg_from_filesystem(config_file)
    if cfg is None:
        # Fallback to Hydra compose if not found on disk
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    # Force targets to local bundled SAM2
    cfg = _rewrite_targets_to_local_sam2(cfg)
    OmegaConf.resolve(cfg)
    # Ensure we import the local 'sam2' package in this directory (avoid name clashes)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    project_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", ".."))
    # Ensure local package paths are importable
    for p in (project_root, parent_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        mod = sys.modules.get("sam2")
        mod_file = getattr(mod, "__file__", None)
        if (mod_file is None) or (not os.path.abspath(mod_file).startswith(this_dir)):
            if "sam2" in sys.modules:
                del sys.modules["sam2"]
            importlib.invalidate_caches()
            importlib.import_module("sam2")
    except Exception:
        # Best-effort; if this fails we still try instantiate
        pass

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    if device is not None:
        model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

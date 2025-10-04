"""Module defining a minimal occupancy map class using pure pytorch ops.

Typical usage example:

  map = OccupancyVoxelMap(intrinsics_3x3, None, visualizer)
  for batch in dataloader:
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
    map.process_posed_rgbd(rgb_img, depth_img, pose_4x4)
  map.vis_map()
  map.save("test.pt")
"""

from typing_extensions import override, List, Tuple

import torch

from rayfronts.mapping.base import RGBDMapping
from rayfronts import geometry3d as g3d, visualizers

class OccupancyVoxelMap(RGBDMapping):
  """A minimalist occupancy map using pytorch
  
  Attributes:
    intrinsics_3x3: See base.
    device: See base.
    visualizer: See base.
    clip_bbox: See base.

    max_pts_per_frame: See __init__.
    vox_size: See __init__.
    max_empty_pts_per_frame: See __init__.
    max_depth_sensing: See __init__.
    max_empty_cnt: See __init__.
    max_occ_cnt: See __init__.
    occ_observ_weight: See __init__.
    occ_thickness: See __init__.
    vox_accum_period: See __init__.

    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_occ: Nx1 Float tensor describing log odds occupancy of voxels.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               max_pts_per_frame: int = 1000,
               vox_size: int = 1,
               vox_accum_period: int = 1,
               max_empty_pts_per_frame: int = 1000,
               max_depth_sensing: float = -1,
               max_empty_cnt: int = 3,
               max_occ_cnt: int = 5,
               occ_observ_weight: int = 5,
               occ_thickness: int = 2):
    """
    
    Args:
      intrinsics_3x3: See base.
      device: See base.
      visualizer: See base.
      clip_bbox: See base.
      max_pts_per_frame: How many points to project per frame. Set to -1 to 
        project all valid depth points.
      vox_size: Length of a side of a voxel in meters.
      vox_accum_period: How often do we aggregate voxels into the global 
        representation. Setting to 10, will accumulate point clouds from 10
        frames before voxelization. Should be tuned to balance memory,
        throughput, and min latency.
      max_empty_pts_per_frame: How many empty points to project per frame.
        Set to -1 to project all valid depth points.
      max_depth_sensing: Depending on the max sensing range, we project empty
        voxels up to that range if that pixel had +inf depth or out of range.
        Set to -1 to use the max depth in that frame as the max sensor range.
      max_empty_cnt: The maximum log odds value for empty voxels. 3 means the
        cell will be capped at -3 which corresponds to a
        probability of e^-3 / ( e^-3 + 1 ) ~= 0.05 Lower values help compression
        and responsivness to dynamic objects whereas higher values help
        stability and retention of more evidence.
      max_occ_cnt: The maximum log odds value for occupied voxels. Same
        discussion of max_empty_cnt applies here.
      occ_observ_weight: How much weight does an occupied observation hold
        over an empty observation.
      occ_thickness: When projecting occupied points, how many points do we
        project as occupied? e.g. Set to 3 to project 3 points centered around
        the original depth value with vox_size/2 spacing between them. This
        helps reduce holes in surfaces.
    """

    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox)

    self.max_pts_per_frame = max_pts_per_frame

    self.vox_size = vox_size

    self.max_empty_pts_per_frame = max_empty_pts_per_frame
    self.max_depth_sensing = max_depth_sensing
    self.max_empty_cnt = max_empty_cnt
    self.max_occ_cnt = max_occ_cnt
    self.occ_observ_weight = occ_observ_weight
    self.occ_thickness = occ_thickness

    # (Nx3)
    self.global_vox_xyz = None
    self.global_vox_occ = None

    # Temporary separate voxel grid accumulation before aggregation
    self.vox_accum_period = vox_accum_period
    self._vox_accum_cnt = 0
    self._tmp_vox_xyz = list()
    self._tmp_vox_occ = list()

  @override
  def save(self, file_path) -> None:
    torch.save(dict(global_vox_xyz=self.global_vox_xyz,
                    global_vox_occ=self.global_vox_occ),
                    file_path)

  @override
  def load(self, file_path) -> None:
    d = torch.load(file_path, weights_only=True)
    self.global_vox_xyz = d["global_vox_xyz"].to(self.device)
    self.global_vox_occ = d["global_vox_occ"].to(self.device)

  @override
  def is_empty(self) -> bool:
    return self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None) -> dict:
    update_info = dict()

    vox_xyz, vox_occ = g3d.depth_to_sparse_occupancy_voxels(
      depth_img, pose_4x4, self.intrinsics_3x3, self.vox_size, conf_map,
      max_num_pts = self.max_pts_per_frame,
      max_num_empty_pts = self.max_empty_pts_per_frame,
      max_depth_sensing = self.max_depth_sensing,
      occ_thickness=self.occ_thickness
    )
    vox_xyz, vox_occ = self._clip_pc(vox_xyz, vox_occ)

    # [0, 1] to [-1, occ_observ_weight]
    vox_occ = vox_occ*self.occ_observ_weight - 1

    B, C, H, W = depth_img.shape
    self._vox_accum_cnt += B
    self._tmp_vox_occ.append(vox_occ)
    self._tmp_vox_xyz.append(vox_xyz)

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0
      self.accum_occ_voxels()

    return update_info

  def accum_occ_voxels(self) -> None:
    """Accumulate the temporarilly gathered occupancy voxels."""
    if len(self._tmp_vox_xyz) == 0:
      return
    if self.global_vox_xyz is not None:
      self._tmp_vox_xyz.append(self.global_vox_xyz)
      self._tmp_vox_occ.append(self.global_vox_occ)
    pts_xyz = torch.cat(self._tmp_vox_xyz, dim = 0)
    pts_occ = torch.cat(self._tmp_vox_occ, dim = 0)
    self._tmp_vox_occ.clear()
    self._tmp_vox_xyz.clear()
    self.global_vox_xyz, self.global_vox_occ = \
      g3d.pointcloud_to_sparse_voxels(
        pts_xyz, feat_pc=pts_occ, vox_size=self.vox_size, aggregation="sum")
    self.global_vox_occ = torch.clamp(self.global_vox_occ,
                                      min = self.max_empty_cnt,
                                      max = self.max_occ_cnt)

  @override
  def vis_map(self) -> None:
    if self.visualizer is None or self.is_empty():
      return
    self.visualizer.log_occ_pc(self.global_vox_xyz, self.global_vox_occ,
                               layer="voxel_occ")

  @override
  def vis_update(self, **kwargs) -> None:
    pass

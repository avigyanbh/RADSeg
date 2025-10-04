from typing_extensions import override

import torch

from rayfronts.feat_compressors import FeatCompressor

class PcaCompressor(FeatCompressor):
  """Compress features using Principal Component Analysis"""

  def __init__(self, out_dim: int, in_dim: int = None, path: str = None):
    """
    Args:
      out_dim: Output dimension
      in_dim: Input dimension
    """
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.mean = None
    self.basis = None
    if path is not None:
      self.load(path)

  @override
  def fit(self, X: torch.FloatTensor) -> None:
    D = X.shape[-1]
    if self.in_dim is not None and D == self.in_dim:
      raise ValueError("Data feature dimension does not match stored input dim")
    self.in_dim = D

    X_flatten = X.flatten(0, -2)
    self.mean = torch.mean(X_flatten, dim=0)
    X = X-self.mean
    _,_,V = torch.pca_lowrank(X_flatten, q = self.out_dim)
    self.basis = V

  @override
  def save(self, fp: str) -> None:
    torch.save(dict(
      metadata=dict(in_dim=self.in_dim, out_dim=self.out_dim),
      mean=self.mean, basis=self.basis), fp)

  @override
  def load(self, fp: str) -> None:
    d = torch.load(fp)
    self.in_dim = d["metadata"]["in_dim"]
    self.out_dim = d["metadata"]["out_dim"]
    self.mean = d["mean"]
    self.basis = d["basis"]

  @override
  def compress(self, X):
    s = list(X.shape)
    s[-1] = self.out_dim
    return (X.flatten(0, -2) @ self.basis).reshape(*s)

  @override
  def decompress(self, Y):
    s = list(Y.shape)
    s[-1] = self.in_dim
    return (Y.flatten(0, -2) @ self.basis.T).reshape(*s)

  @override
  def is_fitted(self):
    return self.mean is not None and self.basis is not None
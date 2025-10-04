"""Defines the feature compressor abstraction."""

import abc

import torch

class FeatCompressor(abc.ABC):
  """Feature Compressor class.
  
  A feature compressor takes a (*, D) tensor where D corresponds to the
  feature dimension and produces a (*, C) tensor where C < D.

  The compressor may need to be fitted first to some data distribution.
  """

  def __init__(self):
    pass
  
  @abc.abstractmethod
  def fit(self, X: torch.FloatTensor) -> None:
    """Fit the compressor to a particular data distribution. May be passthrough.

    Args:
      X: (*, D) float tensor
    """
    pass
  
  @abc.abstractmethod
  def save(self, fp: str) -> None:
    """Save the internal state, if it exists, of the compressor to a file.
    
    Args:
      fp: File path.
    """
    pass

  @abc.abstractmethod
  def load(self, fp: str) -> None:
    """Load the internal state, if it exists, of the compressor from a file
    
    Args:
      fp: File path.
    """
    pass

  @abc.abstractmethod
  def compress(self, X: torch.FloatTensor) -> torch.FloatTensor:
    """Compress a feature tensor.
    
    Args:
      X: (*, D) float tensor to be compressed
    Returns:
      (*, C) float tensor compressed. 
    """
    pass

  @abc.abstractmethod
  def decompress(self, Y: torch.FloatTensor) -> torch.FloatTensor:
    """Decompress a feature tensor.
    
    Args:
      X: (*, C) float tensor to be decompressed
    Returns:
      (*, D) float tensor decompressed. 
    """
    pass

  @abc.abstractmethod
  def is_fitted(self) -> bool:
    pass
"""
Loss functions
"""

from my_torch.losses.cross_entropy import CrossEntropyLoss
from my_torch.losses.loss import Loss
from my_torch.losses.mse import MSELoss

__all__ = ["CrossEntropyLoss", "Loss", "MSELoss"]

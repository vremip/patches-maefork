
from typing import Callable, Optional

import torch


def init_weights(
  module: torch.nn.Module,
  init_method: Callable,
  bias_init_method: Optional[Callable] = None
):
  def _init_weights(sub_module: torch.nn.Module):
    if hasattr(sub_module, "weight"):
      init_method(sub_module.weight)
    if hasattr(sub_module, "bias"):
      if bias_init_method is not None:
        bias_init_method(sub_module.bias)
      else:
        torch.nn.init.zeros_(sub_module.bias)

  module.apply(_init_weights)


from typing import Callable, Optional

import torch

from .models_mae import mae_vit_base_dec512d8b, mae_vit_huge_patch14_dec512d8b, mae_vit_large_patch16_dec512d8b
from .mvit_policy import MViTPolicy

MODELS = dict(
  mae_vit_base = mae_vit_base_dec512d8b,  # decoder: 512 dim, 8 blocks
  mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b,  # decoder: 512 dim, 8 blocks
  mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b,  # decoder: 512 dim, 8 blocks
  mvit_policy = MViTPolicy,
)

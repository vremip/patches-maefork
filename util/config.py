from argparse import Namespace
from dataclasses import dataclass

import torch

from helpers.enums import SimpleEnum


class Enums:
  class InitPatches(SimpleEnum):
    exhaustive: str
    random: str  # patches placed at random locations
    random_cover: str
    grid_conv: str  # exhaustive grid, a la ViT
    grid_mod: str


@dataclass
class Datasetconfig:
  name: str = "imagenet"
  input_size: int = 224  # number of pixels for width and height
  input_channels: int = 3


@dataclass
class PatchesConfig:
  size: int = 16
  init_type: str = Enums.InitPatches.random


@dataclass
class Config:
  dataset = Datasetconfig()
  patches = PatchesConfig()
  device: torch.device = "cpu"


def build_config(args: Namespace):
  config = Config()

  if getattr(args, "device"):
    config.device = torch.device(args.device)

  if getattr(args, 'dataset'):
    config.dataset.name = args.dataset
    
    if args.dataset == "mnist":
      config.dataset.input_size = 28
      config.dataset.input_channels = 1
  
  if getattr(args, "patch_size"):
    config.patches.size = args.patch_size

  return config

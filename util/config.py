from argparse import Namespace
from dataclasses import dataclass

import torch


@dataclass
class Datasetconfig:
  name: str = "imagenet"
  input_size: int = 224
  input_channels: int = 3


@dataclass
class Config:
  dataset = Datasetconfig()
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
  
  return config
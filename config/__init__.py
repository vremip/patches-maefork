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
  
  class ModelTypes(SimpleEnum):
    mvit_policy: str

  class Losses(SimpleEnum):
    margin: str

  class TrainerNames(SimpleEnum):
    policy_trainer: str


@dataclass
class Datasetconfig:
  name: str = "imagenet"
  input_size: int = 224  # number of pixels for width and height
  input_channels: int = 3


@dataclass
class ModelConfig:
  name: str = None
  type: str = None
  batch_size: int = 128
  normalizer: str = "softmax"

  hidden_size: int = None
  num_attn_heads: int = None
  mlp_dim: int = None
  num_layers: int = None
  
  # Size of the MLP hidden layer used for classication
  representation_size: int = None
  classifier: str = "token"
  attention_dropout_rate: float = 0.0
  dropout_rate: float = 0.1
  dtype = torch.float32


@dataclass
class PatchesConfig:
  size: int = 16
  init_type: str = Enums.InitPatches.random
  num_patches_init: int = 0
  num_patches: int = 1  # TODO ?


@dataclass
class OptimizerConfig:
  name: str = "adam"
  beta1: float = 0.9
  beta2: float = 0.999
  weight_decay: float = 0.1  # 0.3
  explicit_weight_decay = None


@dataclass
class TrainingConfig:
  trainer_name: str = None
  loss: str = None

  l2_decay_factor = None
  max_grad_norm = 1.0
  label_smoothing = None
  init_head_bias = -10.0


@dataclass
class LearningRateConfig:
  num_training_epochs: int = 100

  base_lr: float = 1e-3  # 3e-3
  end_lr: float = 1e-5
  learning_rate_schedule: str = "compound"
  factors: str = "constant*linear_warmup*linear_decay"
  warmup_steps: int = 1000

  steps_per_epoch: int = None

  @property
  def total_steps(self):
    return self.num_training_epochs * self.steps_per_epoch


@dataclass
class LoggingConfig:
  log_eval_steps = 10000
  write_summary = True  # write TB and/or XM summary
  write_xm_measurements = False  # write XM measurements
  xprof = False  # Profile using xprof
  checkpoint = False  # do checkpointing
  checkpoint_steps = 5000
  debug_train = False  # debug mode during training
  debug_eval = False  # debug mode during eval


@dataclass
class Config:
  dataset = Datasetconfig()
  patches = PatchesConfig()
  model = ModelConfig()
  optimizer = OptimizerConfig()
  learning_rate = LearningRateConfig()
  logging = LoggingConfig()
  training = TrainingConfig()

  exp: str = None
  device: torch.device = "cpu"
  rng_seed: int = 42


def build_config(args: Namespace):
  config_file = getattr(args, "config")
  
  if config_file == "default":
    config = Config()
  else:
    if config_file == "mnist_policy":
      from mnist_policy_config import get_config

    else:
      raise NotImplementedError(config_file)

    config = get_config(getattr(args, "device", "cuda") == "cpu")


  if getattr(args, "device"):
    config.device = torch.device(args.device)

  if getattr(args, "dataset"):
    config.dataset.name = args.dataset
    
    if args.dataset == "mnist":
      config.dataset.input_size = 28
      config.dataset.input_channels = 1
  
  if getattr(args, "patch_size"):
    config.patches.size = args.patch_size

  return config

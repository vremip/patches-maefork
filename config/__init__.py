from argparse import Namespace
from dataclasses import dataclass
from enum import Enum

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
    softplus: str
    l2: str

  class TrainerNames(SimpleEnum):
    policy_trainer: str

  class ClassifierTypes(SimpleEnum):
    token: str
    gap: str
    gmp: str
    gsp: str

  class NoiseTypes(Enum):
    gaussian = 0
    uniform = 1
    none = 2
  
  class PatchTransforms(Enum):
    kld = 0
    ent = 1


@dataclass
class Datasetconfig:
  name: str = "imagenet"
  input_size: int = 224  # number of pixels for width and height
  input_channels: int = 3
  num_classes: int = 100  # number of output classes.
  num_labels: int = 1


@dataclass
class ModelMaevitConfig:
  norm_pix_loss: bool = False


@dataclass
class ModelConfig:
  name: str = None
  type: str = None
  batch_size: int = 128
  normalizer: str = "softmax"

  hidden_size: int = None  # Size of the hidden state of the output of model's stem
  num_attn_heads: int = None
  mlp_dim: int = None  # Dimension of the mlp on top of attention block
  num_layers: int = None
  
  representation_size: int = None  # Size of the representation layer in the model's head. if None, we skip the extra projection + tanh activation at the end
  classifier: str = Enums.ClassifierTypes.token
  attention_dropout_rate: float = 0.0
  dropout_rate: float = 0.1
  dtype = torch.float32

  single_pass: bool = False  # Do a single pass. Corresponds to basic ViT with fancy position embeddings
  learn_scale: bool = False  # Whether to learn both patch locations and scales

  stochastic = None  # Whether to learn stochastic patch placement. If yes, each patch parameter is a distrib, whose type is defined by noise_type below
  init_var = 0.01  # For stochastic patch placement, regularize the variance of the gaussian so the patches do not collapse to deterministic ones
  noise_type = Enums.NoiseTypes.gaussian
  noise_level: float = 0.0

  pen_type = 0  # Variance regularization, 0 for kld, 1 for ent, -1 for no penalty
  patch_transform = Enums.PatchTransforms.kld

  min_num_patches = 0  # Whether to consider only the first random(low=min_num_patches, high=num_patches) patches in the second pass. Consider all patches when 0
  use_first_pass = False  # Whether to reuse the "columns" from the first pass in the second one.

  autoreg_passes: int = None

  maevit = ModelMaevitConfig()


@dataclass
class PatchesConfig:
  size: int = 16
  init_type: str = Enums.InitPatches.random
  num_patches_init: int = 0
  num_patches: int = 1  # TODO ?
  max_scale_mult: float = 1.0  # Float defining the patch scale range: [1, 1 + max_scale_mult]


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
class MiscConfig:
  """ Laziness, will sort fields in their respective configs."""
  per_patch_token = False
  stochastic_depth: float = 0.0


@dataclass
class Config:
  dataset = Datasetconfig()
  patches = PatchesConfig()
  model = ModelConfig()
  optimizer = OptimizerConfig()
  learning_rate = LearningRateConfig()
  logging = LoggingConfig()
  training = TrainingConfig()
  misc = MiscConfig()

  exp: str = None
  device: torch.device = "cpu"
  rng_seed: int = 42


def build_config(args: Namespace):
  config_file = getattr(args, "config")
  
  if config_file == "default":
    config = Config()
  else:
    if config_file == "mnist_policy":
      from .mnist_policy_config import get_config

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
  if getattr(args, "model"):
    config.model.type = args.model
  if getattr(args, "norm_pix_loss"):
    config.model.maevit.norm_pix_loss = args.norm_pix_loss

  return config

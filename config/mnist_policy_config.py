
from . import Enums
from mnist_base_config import get_base_config


VARIANT = ["mnist_large", "mnist2", "mnist3", "mnist4"][1]
IMG_DIMS = dict(mnist_large=48, mnist2=28, mnist3=28, mnist4=28)[VARIANT]
PATCH_SIZE = dict(mnist_large=3, mnist2=2, mnist3=3, mnist4=4)[VARIANT]
NUM_PATCHES_INIT = dict(mnist_large=220, mnist2=196, mnist3=81, mnist4=49)[VARIANT]


def get_config(runlocal: bool):

  version = "Mi"

  config = get_base_config(runlocal, version)
  config.exp = "mnist_pol"
  config.dataset.input_size = IMG_DIMS

  config.learning_rate.num_training_epochs = 200

  # config overrides
  # config.model.num_heads = max(config.model.num_heads, 3)
  # config.model.num_layers = max(config.model.num_layers, 8)
  config.model.normalizer = "softmax"

  config.model.name = "mvit_classification"
  config.model.type = Enums.ModelTypes.mvit_policy
  config.training.loss = Enums.Losses.margin
  config.training.trainer_name = Enums.TrainerNames.policy_trainer

  # patch configurations
  config.patches.num_patches = 1
  config.patches.num_patches_init = NUM_PATCHES_INIT
  config.patches.size = PATCH_SIZE

  config.model.autoreg_passes = 8
  config.repulsion_weight = 0.0
  config.model.stochastic = None
  config.model.init_var = 0.01
  config.model.noise_type = 0  # 0 for gaussian, 1 for uniform
  config.model.pen_type = 0  # 0 for kld, 1 for ent
  config.model.patch_transform = True  # 0 for kld, 1 for ent
  config.model.epsilon = 0.5  # How many times are we looking at the best patch vs a random patch

  return config


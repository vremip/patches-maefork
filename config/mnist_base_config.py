
from . import Config, Enums

_MNIST_TRAIN_SIZE = 60000

# Dimension of the patch embedding (via a conv applied to the patch)
HIDDEN_SIZE = {
  "Mi": 96,
  "Ti": 192,
  "S": 384,
  "B": 768,
  "L": 1024,
  "H": 1280,
}

ATTN_MLP_DIM = {
  "Mi": 384,
  "Ti": 768,
  "S": 1536,
  "B": 3072,
  "L": 4096,
  "H": 5120,
}

ATTN_NUM_HEADS = {"Mi": 2, "Ti": 3, "S": 6, "B": 12, "L": 16, "H": 16}

ATTN_NUM_BLOCKS = {"Mi": 3, "Ti": 12, "S": 12, "B": 12, "L": 24, "H": 32}


def get_base_config(runlocal: bool, version: str):
  """Returns the base experiment configuration for MNIST."""

  config = Config()
  config.rng_seed = 42

  config.dataset.name = "mnist"
  config.dataset.num_classes = 10

  config.model.batch_size = 8 if runlocal else 128
  config.model.hidden_size = HIDDEN_SIZE[version]

  config.model.num_attn_heads = ATTN_NUM_HEADS[version]
  config.model.mlp_dim = ATTN_MLP_DIM[version]
  config.model.num_layers = ATTN_NUM_BLOCKS[version]

  # Size of the MLP hidden layer used for classication
  config.model.representation_size = None
  config.model.classifier = Enums.ClassifierTypes.token
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.1

  # Training.
  config.optimizer.name = "adam"
  config.optimizer.beta1 = 0.9
  config.optimizer.beta2 = 0.999
  # config.optimizer_configs.weight_decay = 0.3
  config.optimizer.weight_decay = 0.1
  config.optimizer.explicit_weight_decay = None  # No explicit weight decay
  
  config.training.l2_decay_factor = None
  config.training.max_grad_norm = 1.0
  config.training.label_smoothing = None
  config.training.init_head_bias = -10.0

  base_lr = 1e-3
  config.learning_rate.num_training_epochs = 100
  config.learning_rate.steps_per_epoch = _MNIST_TRAIN_SIZE // config.model.batch_size
  config.learning_rate.learning_rate_schedule = "compound"
  config.learning_rate.factors = "constant*linear_warmup*linear_decay"
  config.learning_rate.warmup_steps = 1_000
  config.learning_rate.end_lr = 1e-5
  config.learning_rate.base_lr = base_lr

  # Logging.
  config.logging.log_eval_steps = 10000
  config.logging.write_summary = True  # write TB and/or XM summary
  config.logging.write_xm_measurements = False  # write XM measurements
  config.logging.xprof = False  # Profile using xprof
  config.logging.checkpoint = False  # do checkpointing
  config.logging.checkpoint_steps = 5000
  config.logging.debug_train = False  # debug mode during training
  config.logging.debug_eval = False  # debug mode during eval

  return config

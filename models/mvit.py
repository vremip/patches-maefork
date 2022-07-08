
from typing import Any, Optional
import torch
import torch.nn as nn

from .mvit_encoder import Encoder
from utils.config import PatchesConfig
from utils.patches import PatchExtractor, select_patches


class MViT(nn.Module):
  """Main model.
  Does two passes, extracts patches between the first and the second.

  Attributes:
  num_classes: Number of output classes.
  mlp_dim: Dimension of the mlp on top of attention block.
  num_layers: Number of layers.
  num_heads: Number of self-attention heads.
  patches: Configuration of the patches extracted in the stem of the model.
  hidden_size: Size of the hidden state of the output of model's stem.
  representation_size: Size of the representation layer in the model's head.
    if None, we skip the extra projection + tanh activation at the end.
  dropout_rate: Dropout rate.
  attention_dropout_rate: Dropout for attention heads.
  classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
    'token'.
  dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: PatchesConfig
  hidden_size: int
  img_dims: list
  num_patches_init: int
  num_patches: int
  min_num_patches: int = None
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier_type: str = "gap"
  dtype: Any = torch.float32
  run_time: float = 0.0
  init_patches: str = "random"
  single_pass: bool = False
  max_scale_mult: float = 1.0
  num_labels: int = 1
  per_patch_token: bool = False
  stochastic: float = None
  normalizer: str = "softmax"
  noise_type: int = 0
  patch_transform: bool = True
  autoreg_passes: int = 0
  noise_level: float = 0
  epsilon: float = 0.0
  learn_scale: bool = False
  use_first_pass: bool = False

  def setup(self):
    fh, fw = self.patches.size
    if self.init_patches == "random":
      h = w = 1
    else:
      h = self.img_dims[0] // fh
      w = self.img_dims[1] // fw
    self.encoder = Encoder(
      mlp_dim=self.mlp_dim,
      num_layers=self.num_layers,
      num_heads=self.num_heads,
      num_patches=self.num_patches if self.per_patch_token else 1,
      img_dims=[h, w],
      patches_size=[fh, fw],
      dropout_rate=self.dropout_rate,
      attention_dropout_rate=self.attention_dropout_rate,
      stochastic_depth=self.stochastic_depth,
      classifier=self.classifier_type,
      dtype=self.dtype,
      hidden_size=self.hidden_size,
      name="Transformer",
      locscale_token=not self.single_pass,
      num_labels=self.num_labels,
      normalizer=self.normalizer,
    )

    self.locscale_dim = 3 if self.learn_scale else 2

    self.locscale_extractor = LocScaleExtractor(
      self.hidden_size,
      self.num_patches,
      stochastic=self.stochastic,
      locscale_dim=self.locscale_dim,
    )

    output_projection = torch.nn.Linear(1, self.num_classes)
    torch.nn.init.zeros_(output_projection)

    self.classifier = Classifier(
      representation_size=self.representation_size,
      num_classes=self.num_classes,
      output_projection=output_projection,
    )

    self.transform = None
    # self.transform = Transform(
    #     name='Patch_Transform'
    # )

  def _get_init_patches(self, inputs: torch.Tensor, num_patches: int = None):

    # If random: Extract self.num_patches_init random patches from the image
    # If grid_mod: Extract an exhaustive grid of patches from the image with the patch extractor. Does go through the smoothing with a small variance gaussian.
    # If grid_conv: split the image into an exhaustive grid.
    if num_patches is None:
      num_patches = self.num_patches_init
    if self.init_patches in ["random", "grid_mod", "random_cover"]:
      return PatchExtractor(
        rng=self.make_rng("patch_extractor"),
        num_patches=num_patches,
        img_dims=self.img_dims,
        patches=self.patches,
        type_init_patches=self.init_patches,
        max_scale_mult=self.max_scale_mult,
      )(inputs)
    # Standard exhaustive grid with strided convs.
    # The strided conv will act as a patch extractor + "embedder"
    elif self.init_patches == "grid_conv":
      return dict(patches=torch.unsqueeze(inputs, 1))
    else:
      raise NotImplementedError(
        f'Did not recognize init_patches value. Got "{self.init_patches}", expected one of "random", "grid_conv" or "grid_mod"'
      )

  def _get_locscale_embeds(self, embeds: torch.Tensor, extract: bool = True):
    if extract:
      if self.classifier_type == "token":
        begin = self.num_labels
      else:
        begin = 0

      if self.per_patch_token:
        embeds = embeds[:, begin : self.num_labels + self.num_patches]
      else:
        embeds = embeds[:, begin]

    return embeds

  def _get_locscale(self, embeds: torch.Tensor, extract: bool = True):
    embeds = self._get_locscale_embeds(embeds, extract)
    # Use locscale_token to determine which patches to extract from initial image
    # Returns either [batch_size, 3 * num_patches] or [batch_size, num_patches, 3]
    # 3 if self.learn_scale, 2 otherwise
    return self.locscale_extractor(embeds)

  def _get_patch_embeds(self, embeds: torch.Tensor):
    if self.classifier_type == "token":
      begin = self.num_labels
    else:
      begin = 0
    if self.per_patch_token:
      return embeds[:, :, begin + self.num_patches :]
    else:
      return embeds[:, :, begin + 1 :]

  def _reparametrize(
    self, mean: torch.Tensor, var: torch.Tensor, rng, train: bool = True
  ):
    # Reparametrize assuming Gaussian distribution

    def _no_noise(mean, var):
      return mean

    def _gaussian(mean, var):
      eps = torch.randn_like(var)
      return mean + eps * torch.exp(0.5 * var)

    def _uniform(mean, var):
      eps = torch.rand_like(var)
      return mean + 2 * (eps - 0.5) * torch.exp(var)

    if self.noise_type == 0:  # 'gaussian'
      _noise = _gaussian
    elif self.noise_type == 1:  # 'uniform'
      _noise = _uniform

    return _noise(mean, var)

    # if train:
    #   return _noise(mean, var)
    # else:
    #   return _no_noise(mean, var)

  def _extract_patches(
    self, inputs: torch.Tensor, locscale_params: torch.Tensor, train: bool = True
  ):
    batch_size = locscale_params.shape[0]

    if self.stochastic is not None:
      size = locscale_params.shape[-1] // 2
      locscale = self._reparametrize(
        locscale_params[..., :size],
        locscale_params[..., size:],
        self.make_rng("patch_extractor"),
        train=train,
      )
      if self.transform:  # Apply transformation to patch parameters
        locscale = self.transform(locscale)
    else:  # Deterministic patch extraction
      locscale = locscale_params

    if self.per_patch_token:
      assert locscale.shape == (
        batch_size,
        self.num_patches,
        self.locscale_dim,
      ), f"Wrong shape, expected {(batch_size, self.num_patches, self.locscale_dim)}, got {locscale.shape}"
      # Shape [batch_size, num_patches, 1]
      # x-coordinates of the center of the patch / mask
      x_pos = locscale[:, :, :1]
      # Shape [batch_size, num_patches, 1]
      # y-coordinates of the center of the patch / mask
      y_pos = locscale[:, :, 1:2]
      # Shape [batch_size, num_patches, 1]
      # Scale of the patch
      if self.learn_scale:
        scale = locscale[:, :, 2:] / self.max_scale_mult
      else:
        scale = torch.zeros_like(x_pos)
    else:
      # Expected shape of locscale = [batch_size, 3 * num_patches]
      assert locscale.shape == (
        batch_size,
        self.locscale_dim * self.num_patches,
      ), f"Wrong shape, expected {(batch_size, self.locscale_dim * self.num_patches)}, got {locscale.shape}"
      # Need to resize the token to the proper dimension.
      # For each patch, three values are expected, x_pos, y_pos, and scale
      # Patch extraction inspired from DRAW https://arxiv.org/pdf/1502.04623.pdf
      locscale = locscale.reshape((batch_size, self.locscale_dim * self.num_patches, 1))
      # Shape [batch_size, num_patches, 1]
      # x-coordinates of the center of the patch / mask
      x_pos = locscale[:, : self.num_patches]
      # Shape [batch_size, num_patches, 1]
      # y-coordinates of the center of the patch / mask
      y_pos = locscale[:, self.num_patches : 2 * self.num_patches]
      # Shape [batch_size, num_patches, 1]
      # Scale of the patch
      if self.learn_scale:
        scale = locscale[:, 2 * self.num_patches :] / self.max_scale_mult
      else:
        scale = torch.zeros_like(x_pos)

    if self.patch_transform:
      x_pos = torch.tanh(x_pos)  # [-1, 1]
      y_pos = torch.tanh(y_pos)  # [-1, 1]
      if self.learn_scale:
        scale = torch.sigmoid(scale)  # [0, 1]
    else:
      if self.learn_scale:
        scale = torch.exp(scale)  # [0, inf]

    # To see whether it helps against steganography
    if self.noise_level:
      x_pos += self.noise_level * torch.randn_like(x_pos)
      y_pos += self.noise_level * torch.randn_like(y_pos)
      # scale += self.noise_level * torch.randn_like(scale)

    # Extract patches from image using the locscale info
    return (
      PatchExtractor(
        rng=self.make_rng("patch_extractor"),
        num_patches=self.num_patches,
        img_dims=self.img_dims,
        patches=self.patches,
        max_scale_mult=self.max_scale_mult,
      )(inputs, x_pos=x_pos, y_pos=y_pos, scale=scale),
      locscale,
    )

  def _single_pass(
    self,
    train: bool,
    patches_info=None,
    batch_size=None,
    prev_columns: torch.Tensor = None,
  ):
    """
    patches_info: dictionary containing patches and their positions / scales
    prev_columns: [num_layers, batch_size, length, hidden_size]
      Prior computations from the processing of length tokens.
      Reused as keys in the transformer (in addition to the patches and locscale + class token).
      Used for the autoregressive algorithm.
    """
    # Pass on the image to extract the relevant patches
    emb_patches, embeddings = self.encoder(
      patches_info=patches_info,
      train=train,
      batch_size=batch_size,
      prev_columns=prev_columns,
    )

    # Classification token, trained via CE
    if self.classifier_type == "token":
      logits = self.classifier(emb_patches[:, : self.num_labels])
      if self.num_labels == 1:
        logits = torch.squeeze(logits, dim=1)
    else:
      # If no token, use output of last column
      logits = self.classifier(emb_patches[:, -1])
    return logits, emb_patches, embeddings

  def __call__(self, inputs: torch.Tensor, *, train: bool, debug: bool = False):

    init_patches_info = self._get_init_patches(inputs)
    first_logits, locscale_embeds, embeddings = self._single_pass(
      train, init_patches_info
    )

    if self.single_pass:
      return (
        (first_logits, torch.zeros_like(first_logits), torch.zeros_like(first_logits)),
        (init_patches_info,),
        (),
      )

    # Get "output tokens" pertaining to locscale
    locscale_params = self._get_locscale(locscale_embeds)

    # Extract patches from image using the locscale info
    sec_patches_info, locscale = self._extract_patches(
      inputs, locscale_params, train=train
    )

    # Try to predict class from loc scale information if single label prediction
    if self.num_labels == 1:
      locscale = torch.reshape(locscale, (locscale.shape[0], -1))
      locscale_logits = PredictorFromLocScale(
        num_classes=self.num_classes, name="Locscale_Classifier"
      )(locscale)
    else:
      # Could be adapted to multi label
      locscale_logits = torch.zeros_like(first_logits)

    # Select top-K patches if specified
    if train and self.min_num_patches:
      num_patches = torch.randint(
        low=min(self.num_patches, self.min_num_patches), high=self.num_patches + 1
      )
      sec_patches_info = select_patches(sec_patches_info, num_patches)

    if self.use_first_pass:
      prev_columns = self._get_patch_embeds(embeddings)
      final_logits, _, _ = self._single_pass(train, sec_patches_info, prev_columns)
    else:
      final_logits, _, _ = self._single_pass(train, sec_patches_info)

    return (
      (first_logits, final_logits, locscale_logits),
      (init_patches_info, sec_patches_info),
      (locscale_params,),
    )


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x


class Classifier(nn.Module):
  """
  Takes the class token as input and outputs logits
  representation_size gives the dim of the hidden layer of the mlp is passed, otherwise linear transform
  """
  def __init__(self, representation_size: int, num_classes: int, output_projection: nn.Module):
    if representation_size is not None:
      self.layers = nn.Sequential(
        nn.Linear(1, representation_size),
        nn.Tanh(),
      )
    else:
      self.layers = IdentityLayer()
    self.output_projection = output_projection

  def forward(self, inputs: torch.Tensor):
    logits = self.layers(inputs)
    return self.output_projection(logits)


class LocScaleExtractor(nn.Module):
  def __init__(self, hidden_size: int, num_patches: int, locscale_dim: int, stochastic: bool = False):
    # Mean and variance of patch parameters when stochastic
    output_mult = 2 if stochastic is not None else 1

    if True:  # TODO if learn_scale?
      self.layers = torch.nn.Sequential(
        torch.nn.Linear(1, hidden_size),  # TODO
        torch.nn.ReLU(),
        # torch.nn.Linear(self.hidden_size, self.hidden_size),
        # torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, locscale_dim * num_patches * output_mult),
      )
    else:
      self.layers = torch.nn.Linear(hidden_size, locscale_dim * output_mult)

    torch.nn.init.normal_(self.layers, std=0.08)

  def forward(self, x: torch.Tensor):
    """
    Use locscale_token to determine which patches to extract from initial image
    """
    return self.layers(x)


class PredictorFromLocScale(nn.Module):
  """
  A small MLP predicting the class from the loc+scale information.
  Evaluates steganography.
  """

  def __init__(self, num_classes: int, hidden_size: int = 96):
    self.layers = nn.Sequential(
      nn.Linear(1, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, num_classes),
    )
    torch.nn.init.xavier_normal_(self.layers)

  def forward(self, loc_scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
      loc_scale: [batch_size, 3 * num_patches]
    """
    return self.layers(loc_scale.detach())


from typing import Any
import torch
import torch.nn as nn


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value. Our implementation of stochastic depth
      follows timm library, which does per-example layer dropping and uses
      independent dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  hidden_size: int
  num_patches: int
  img_dims: list[int]
  patches_size: list[int]
  classifier: str = "token"
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = torch.float32
  locscale_token: bool = True
  num_labels: int = 1
  normalizer: str = "softmax"

  def setup(self):
    fh, fw = self.patches_size
    # Embedding patches is in fact a single convolution.
    self.conv_embedding = nn.Conv(
      self.hidden_size,
      (1, fh, fw),
      strides=(1, fh, fw),
      padding="VALID",
      name="Conv_Embedding",
    )

  @nn.compact
  def __call__(
    self,
    patches_info: dict,
    train: bool = False,
    batch_size: int = None,
    prev_columns: jnp.ndarray = None,
    extra_keys: jnp.ndarray = None,
  ):
    """Applies Transformer model on the inputs.
    patches_info contains the loc/scale of the patches, if not passed, assumes a grid.
    Coordinates belong to [-1, 1]
    """
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    if patches_info:
      # Embed the patches using the conv
      x = self.conv_embedding(patches_info["patches"])

      # Shape is either `[batch size, num masks, 1, 1, emb]` or `[batch size, 1, h, w, emb]`
      assert x.ndim == 5
      n = x.shape[0]
      x = jnp.reshape(x, [n, -1, self.hidden_size])

      # Shape stays the same
      x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        hidden_size=self.hidden_size,
        img_dims=self.img_dims,
        name="posembed_input",
      )(x, patches_info=patches_info)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      
    else:
      # For autoreg. No patches yet. Just do an "empty" pass to get the locscale of the first patches to extract
      assert batch_size
      n = batch_size
      x = jnp.zeros((n, 0, self.hidden_size))

    # Adding the token containing patches information.
    if self.locscale_token:
      cls = self.param(
        "patch",
        nn.initializers.normal(stddev=0.08),
        (1, self.num_patches, self.hidden_size),
        x.dtype,
      )
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # If we want to add class token, add them here.
    if self.classifier == "token":
      cls = self.param(
        "cls", nn.initializers.zeros, (1, self.num_labels, self.hidden_size), x.dtype
      )
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # Input Encoder.
    embeds = jnp.zeros((0,) + x.shape)
    for lyr in range(self.num_layers):
      embeds = jnp.concatenate((jnp.expand_dims(x, axis=0), embeds), axis=0)
      if prev_columns is not None:
        _extra_keys = prev_columns[lyr]
        if extra_keys is not None:
          _extra_keys = jnp.concatenate((_extra_keys, extra_keys), axis=1)
      elif extra_keys is not None:
        _extra_keys = extra_keys
      else:
        _extra_keys = None
      x = Encoder1DBlockWithFixedTokens(
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=(lyr / max(self.num_layers - 1, 1)) * self.stochastic_depth,
        name=f"encoderblock_{lyr}",
        dtype=dtype,
        normalizer=self.normalizer,
      )(x, deterministic=not train, extra_keys=_extra_keys)
    encoded = nn.LayerNorm(name="encoder_norm")(x)
    return encoded, embeds


class Encoder1DBlockWithFixedTokens(Encoder1DBlock):  # rendu là
  """Transformer encoder layer.
  If prev_columns is passed, consider them as extra keys to attend to as well.
  """

  @nn.compact
  def __call__(
    self, inputs: jnp.ndarray, deterministic: bool, extra_keys: jnp.ndarray = None
  ) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    if extra_keys is not None:
      kv = nn.LayerNorm(dtype=self.dtype)(extra_keys)
      kv = jnp.concatenate((x, kv), axis=1)
    else:
      kv = x
    x = MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      dtype=self.dtype,
      kernel_init=nn.initializers.xavier_uniform(),
      broadcast_dropout=False,
      deterministic=deterministic,
      dropout_rate=self.attention_dropout_rate,
      normalizer=self.normalizer,
    )(x, kv)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = x * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
      mlp_dim=self.mlp_dim,
      dtype=self.dtype,
      dropout_rate=self.dropout_rate,
      activation_fn=nn.gelu,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)

    return y * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + x


from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn as nn

from utils.patches import PatchExtractor


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

  def forward(
    self,
    patches_info: dict,
    train: bool = False,
    batch_size: int = None,
    prev_columns: torch.Tensor = None,
    extra_keys: torch.Tensor = None,
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


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value.

  Returns:
    output after transformer encoder block.
  """

  mlp_dim: int
  num_heads: int
  dtype: Any = torch.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  normalizer: str = "softmax"

  def get_stochastic_depth_mask(
    self, x: torch.Tensor, deterministic: bool
  ) -> torch.Tensor:
    """Generate the stochastic depth mask in order to apply layer-drop.

    Args:
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Stochastic depth mask.
    """
    if not deterministic and self.stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
        self.make_rng("dropout"), self.stochastic_depth, shape
      )
    else:
      return 0.0

  def forward(self, inputs: torch.Tensor, deterministic: bool) -> torch.Tensor:
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
    x = MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      dtype=self.dtype,
      kernel_init=nn.initializers.xavier_uniform(),
      broadcast_dropout=False,
      deterministic=deterministic,
      dropout_rate=self.attention_dropout_rate,
      normalizer=self.normalizer,
    )(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = x * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = AttentionLayers.MlpBlock(
      mlp_dim=self.mlp_dim,
      dtype=self.dtype,
      dropout_rate=self.dropout_rate,
      activation_fn=nn.gelu,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)

    return y * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + x


class Encoder1DBlockWithFixedTokens(Encoder1DBlock):
  """Transformer encoder layer.
  If prev_columns is passed, consider them as extra keys to attend to as well.
  """

  @nn.compact
  def __call__(
    self, inputs: torch.Tensor, deterministic: bool, extra_keys: torch.Tensor = None
  ) -> torch.Tensor:
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
    y = AttentionLayers.MlpBlock(
      mlp_dim=self.mlp_dim,
      dtype=self.dtype,
      dropout_rate=self.dropout_rate,
      activation_fn=nn.gelu,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)

    return y * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + x


class AttentionLayers:
  class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    use_bias: bool = True
    kernel_init: Initializer = nn.initializers.xavier_uniform()
    bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
    activation_fn: Callable[[torch.Tensor], torch.Tensor] = nn.gelu
    precision: Optional[jax.lax.Precision] = None
    dtype: torch.Tensor = torch.float32

    def forward(self, inputs: torch.Tensor, *, deterministic: bool):
      """Applies Transformer MlpBlock module."""
      actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
      x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
      )(inputs)
      x = self.activation_fn(x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
      output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision,
      )(x)
      output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
      return output


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """

  hidden_size: int
  img_dims: List[int]
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  def setup(self, min_rescale=0.1, max_rescale=10.0):
    """
    Compute a set of rescaling factors for sin+cos features.

    Output:
      rescales: (1, n_scales)
    """
    assert self.hidden_size % 4 == 0
    n_scales = self.hidden_size // 4
    self.rescales = torch.reshape(
      torch.logspace(torch.log10(min_rescale), torch.log10(max_rescale), n_scales),
      (1, 1, n_scales),
    )

    # Use the basic mesh on the image and transform it appropriately.
    # Compute this once during setup for reuse
    x_grid_features = self._sincos_1d(
      torch.reshape(PatchExtractor.create_grid(self.img_dims[1]), (1, -1, 1))
    )
    x_grid_features = torch.tile(x_grid_features, (self.img_dims[0], 1, 1))
    y_grid_features = torch.transpose(
      self._sincos_1d(torch.reshape(PatchExtractor.create_grid(self.img_dims[0]), (1, -1, 1))), (1, 0, 2)
    )
    y_grid_features = torch.tile(y_grid_features, (1, self.img_dims[1], 1))
    self.pos_embs = torch.cat((y_grid_features, x_grid_features), dim=-1)
    self.pos_embs = torch.reshape(
      self.pos_embs, (1, self.img_dims[0] * self.img_dims[1], self.hidden_size)
    )

  def _sincos_1d(self, x, scale=1.0):
    """
    Helper func for _sincos_features...

    Input:
      x: (n_batch, num_patches, 1)
    Output:
      x: (n_batch, num_patches, self.hidden_size / 2)  # sin+cos features
    """
    # x = (x * (1 + scale)) * self.rescales
    x = x * self.rescales
    x = torch.cat([torch.sin(x), torch.cos(x)], axis=2)
    return x

  def _sincos_features(self, patches_info):
    """
    Input:
      patches_info containing x_pos, y_pos and scale of shape: (n_batch, num_patches, 1)
    Output:
      x: (n_batch, num_patches, self.hidden_size)  # sin+cos features
    """
    x_pos, y_pos, scale = (
      patches_info["x_pos"],
      patches_info["y_pos"],
      patches_info["scale"],
    )
    y_pos = (
      self._sincos_1d(y_pos, scale=scale) * scale
    )  # (n_batch, n_locs, self.hidden_size/2)
    x_pos = (
      self._sincos_1d(x_pos, scale=scale) * scale
    )  # (n_batch, n_locs, self.hidden_size/2)
    x = jnp.concatenate([y_pos, x_pos], axis=2)
    return x

  def forward(
    self, inputs: torch.Tensor, patches_info: list[torch.Tensor] = None
  ) -> torch.Tensor:
    # Inputs.shape is (batch_size, num patches, emb_dim).
    assert inputs.ndim == 3, (
      "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
    )

    # NERF
    if "x_pos" in patches_info:
      pos_embs = self._sincos_features(patches_info)
    else:
      # Use a uniform grid between -1 and 1 describing the center of the image pixels
      # Used for grid + strided conv
      pos_embs = self.pos_embs
    # Stopping gradient here to reduce steganography (ie using locscale info to predict classes)
    pos_embs = nn.relu(
      nn.Dense(self.hidden_size, kernel_init=self.posemb_init, name="pos_embedding_1")(
        pos_embs.detach()
      )
    )
    pos_embs = nn.relu(
      nn.Dense(self.hidden_size, kernel_init=self.posemb_init, name="pos_embedding_2")(
        pos_embs
      )
    )
    pos_embs = nn.Dense(
      self.hidden_size, kernel_init=self.posemb_init, name="pos_embedding_last"
    )(pos_embs)

    return inputs + pos_embs


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly
      using dropout, whereas if true, the attention weights
      are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape
      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
  """

  num_heads: int
  dtype = torch.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  use_bias: bool = True
  attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
  decode: bool = False
  normalizer: str = "softmax"

  def forward(
    self,
    inputs_q: Array,
    inputs_kv: Array,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
      deterministic = deterministic or self.deterministic

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert (
      qkv_features % self.num_heads == 0
    ), "Memory dimension must be divisible by number of heads."
    head_dim = qkv_features // self.num_heads

    dense = partial(
      DenseGeneral,
      axis=-1,
      features=(self.num_heads, head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
      dense(dtype=self.dtype, name="query")(inputs_q),
      dense(dtype=self.dtype, name="key")(inputs_kv),
      dense(dtype=self.dtype, name="value")(inputs_kv),
    )

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable("cache", "cached_key")
      cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable(
        "cache", "cached_value", jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            "Autoregressive cache shape error, "
            "expected query shape %s instead got %s." % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(max_length) <= cur_index, tuple(batch_dims) + (1, 1, max_length)
          ),
        )

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng("dropout")

    # apply attention
    x = self.attention_fn(
      query,
      key,
      value,
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=deterministic,
      dtype=self.dtype,
      precision=self.precision,  # pytype: disable=wrong-keyword-args
      normalizer=self.normalizer,
    )  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    out = DenseGeneral(
      features=features,
      axis=(-2, -1),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      precision=self.precision,
      name="out",
    )(x)
    return out

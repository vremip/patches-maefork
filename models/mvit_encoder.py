

from typing import Any, Callable, Dict, List, Optional, Tuple
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

  def __init__(
    self,
    num_layers: int,
    mlp_dim: int,
    num_heads: int,
    hidden_size: int,
    num_patches: int,
    img_dims: Tuple[int, int],
    patches_size: Tuple[int, int],
    classifier: str = "token",
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    stochastic_depth: float = 0.0,
    dtype: Any = torch.float32,
    locscale_token: bool = True,
    num_labels: int = 1,
    normalizer: str = "softmax",
  ):
    # Embedding patches is in fact a single convolution.
    fh, fw = patches_size
    assert fh == fw

    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.conv_embedding = nn.Conv2d(
      99,  # in_channels, TODO
      hidden_size,
      fh,
      stride=fh,
      padding=0,
    )
    self.positional_embedding = AddPositionEmbs(
      hidden_size=hidden_size,
      img_dims=img_dims,
    )
    self.positional_embedding_dropout = nn.Dropout(dropout_rate)

    # Contains patches information
    if locscale_token:
      self.locscale_token = nn.parameter.Parameter(torch.empty((1, num_patches, hidden_size)))  # TODO instantiate every time we call forward? or not?
      torch.nn.init.normal_(self.locscale_token, std=0.08)
    else:
      self.locscale_token = None
    
    if classifier == "token":
      self.class_token = nn.parameter.Parameter(torch.zeros((1, num_labels, hidden_size), dtype=dtype))  # TODO instantiate every time we call forward? or not?
    else:
      self.class_token = None
    
    self.encoder_1ds = nn.ModuleList((
      Encoder1DBlockWithFixedTokens(
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        stochastic_depth=(lyr / max(num_layers - 1, 1)) * stochastic_depth,
        dtype=dtype,
        normalizer=normalizer,
        hidden_size=hidden_size,
      ) for lyr in range(num_layers)
    ))
    self.layer_norm = nn.LayerNorm()

  def forward(
    self,
    patches_info: dict,
    batch_size: Optional[int] = None,
    prev_columns: Optional[torch.Tensor] = None,
    extra_keys: Optional[torch.Tensor] = None,
  ):
    """Applies Transformer model on the inputs.
    patches_info contains the loc/scale of the patches, if not passed, assumes a grid.
    Coordinates belong to [-1, 1]
    """
    if patches_info:
      # Embed the patches using the conv
      x = self.conv_embedding(patches_info["patches"])

      # Shape is either `[batch size, num masks, 1, 1, emb]` or `[batch size, 1, h, w, emb]`
      assert x.ndim == 5
      n = x.shape[0]
      x = torch.reshape(x, [n, -1, self.hidden_size])

      # Shape stays the same
      x = self.positional_embedding(x, patches_info=patches_info)
      x = self.positional_embedding_dropout(x)
      
    else:
      # For autoreg. No patches yet. Just do an "empty" pass to get the locscale of the first patches to extract
      assert batch_size
      n = batch_size
      x = torch.zeros((n, 0, self.hidden_size))

    if self.locscale_token:
      locscale_token = torch.tile(self.locscale_token, [n, 1, 1])
      x = torch.cat([locscale_token, x], dim=1)

    if self.class_token:
      class_token = torch.tile(self.class_token, [n, 1, 1])
      x = torch.cat([class_token, x], dim=1)

    # Input Encoder.
    embeds = torch.zeros((0,) + x.shape)
    for lyr, encoder in enumerate(self.encoder_1ds):
      embeds = torch.cat((torch.unsqueeze(x, dim=0), embeds), dim=0)
      if prev_columns is not None:
        _extra_keys = prev_columns[lyr]
        if extra_keys is not None:
          _extra_keys = torch.cat((_extra_keys, extra_keys), dim=1)
      elif extra_keys is not None:
        _extra_keys = extra_keys
      else:
        _extra_keys = None
      x = encoder(x, extra_keys=_extra_keys)
    encoded = self.layer_norm(x)
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

  def __init__(
    self,
    mlp_dim: int,
    num_heads: int,
    hidden_size: int,
    dtype: Any = torch.float32,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    stochastic_depth: float = 0.0,
    normalizer: str = "softmax",
  ):
    self.stochastic_depth = stochastic_depth

    # Attention block
    self.layer_norm1 = nn.LayerNorm(dtype=dtype)
    self.multi_head_dot_prod_attn = MultiHeadDotProductAttention(
        num_heads=num_heads,
        dtype=dtype,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        normalizer=normalizer,
        out_features=hidden_size,  # TODO i think?
        qkv_features=hidden_size,  # TODO i think?
      )
    self.dropout = nn.Dropout(dropout_rate)

    # MLP block
    self.layer_norm2 = nn.LayerNorm(dtype=dtype)
    self.mlp_block = AttentionLayers.MlpBlock(
      mlp_dim=mlp_dim,
      dtype=dtype,
      dropout_rate=dropout_rate,
    )

  def get_stochastic_depth_mask(self, x: torch.Tensor) -> torch.Tensor:
    """Generate the stochastic depth mask in order to apply layer-drop.
    """
    if self.training and self.stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return torch.rand(shape) < self.stochastic_depth
    else:
      return 0.0

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.

    Returns:
      Output after transformer encoder block.
    """
    assert inputs.ndim == 3

    # Attention block.    
    x = self.layer_norm1(inputs)
    x = self.multi_head_dot_prod_attn(x, x)
    x = self.dropout(x)
    x = x * (1.0 - self.get_stochastic_depth_mask(x)) + inputs

    # MLP block.
    y = self.layer_norm2(x)
    y = self.mlp_block(y)
    return y * (1.0 - self.get_stochastic_depth_mask(x)) + x


class Encoder1DBlockWithFixedTokens(Encoder1DBlock):
  """Transformer encoder layer.
  If prev_columns is passed, consider them as extra keys to attend to as well.
  """

  def __init__(self, *args, **kwargs):
    self.layer_norm_kv = nn.LayerNorm(dtype=kwargs.get('dtype'))
    super().__(*args, **kwargs)

  def forward(
    self, inputs: torch.Tensor, extra_keys: torch.Tensor = None
  ) -> torch.Tensor:
    assert inputs.ndim == 3

    # Attention block.
    x = self.layer_norm1(inputs)

    if extra_keys is not None:
      kv = self.layer_norm_kv(extra_keys)
      kv = torch.cat((x, kv), dim=1)
    else:
      kv = x

    x = self.multi_head_dot_prod_attn(x, kv)
    x = self.dropout(x)
    x = x * (1.0 - self.get_stochastic_depth_mask(x)) + inputs

    # MLP block.
    y = self.layer_norm2(x)
    y = self.mlp_block(y)
    return y * (1.0 - self.get_stochastic_depth_mask(x)) + x


class AttentionLayers:
  class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
      self,
      mlp_dim: int,
      out_dim: Optional[int] = None,
      dropout_rate: float = 0.1,
      activation_fn: Callable[[torch.Tensor], torch.Tensor] = None,
      dtype: torch.Tensor = torch.float32,
    ):
      out_dim = out_dim or 1   # TODO
      
      self.linear1 = nn.Linear(1, mlp_dim, dtype=dtype)
      self.linear2 = nn.Linear(mlp_dim, out_dim, dtype=dtype)
      self.layers = nn.Sequential(
        self.linear1,
        activation_fn or nn.GELU,
        nn.Dropout(dropout_rate),
        self.linear2,
        nn.Dropout(dropout_rate),
      )
      self.initialize()
  
    def initialize(self):
      torch.nn.init.xavier_uniform_(self.linear1.weight)
      torch.nn.init.xavier_uniform_(self.linear2.weight)
      torch.nn.init.normal_(self.linear1.bias, std=1e-6)
      torch.nn.init.normal_(self.linear2.bias, std=1e-6)

    def forward(self, inputs: torch.Tensor):
      """Applies Transformer MlpBlock module."""
      return self.layers(inputs)


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """

  def __init__(self, hidden_size: int, img_dims: Tuple[int, int]):
    min_rescale = 0.1
    max_rescale = 10.0
  
    assert hidden_size % 4 == 0
    n_scales = hidden_size // 4

    self.rescales = torch.reshape(
      torch.logspace(torch.log10(min_rescale), torch.log10(max_rescale), n_scales),
      (1, 1, n_scales),
    )

    # Use the basic mesh on the image and transform it appropriately.
    # Compute this once during setup for reuse
    x_grid_features = self._sincos_1d(
      torch.reshape(PatchExtractor.create_grid(img_dims[1]), (1, -1, 1))
    )
    x_grid_features = torch.tile(x_grid_features, (img_dims[0], 1, 1))
    y_grid_features = torch.transpose(
      self._sincos_1d(torch.reshape(PatchExtractor.create_grid(img_dims[0]), (1, -1, 1))), (1, 0, 2)
    )
    y_grid_features = torch.tile(y_grid_features, (1, img_dims[1], 1))
    self.pos_embs = torch.cat((y_grid_features, x_grid_features), dim=-1)
    self.pos_embs = torch.reshape(
      self.pos_embs, (1, img_dims[0] * img_dims[1], hidden_size)
    )

    self.layers = nn.Sequential(
      nn.Linear(1, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
    )
    torch.nn.init.normal_(self.layers, std=0.02)  # From BERT.

  def _sincos_1d(self, x, scale=1.0):
    """
    Helper func for _sincos_features...

    Input:
      x: (n_batch, num_patches, 1)
    Output:
      x: (n_batch, num_patches, self.hidden_size / 2)  # sin+cos features
    """
    # x *= (1 + scale)
    x *= self.rescales
    x = torch.cat([torch.sin(x), torch.cos(x)], axis=2)
    return x

  def _sincos_features(self, patches_info: Dict):
    """
    Input:
      patches_info containing x_pos, y_pos and scale of shape: (n_batch, num_patches, 1)
    Output:
      x: (n_batch, num_patches, self.hidden_size)  # sin+cos features
    """
    scale = patches_info["scale"]

    # (n_batch, n_locs, self.hidden_size/2)
    y_pos = self._sincos_1d(patches_info["y_pos"], scale=scale) * scale  
    x_pos = self._sincos_1d(patches_info["x_pos"], scale=scale) * scale
    x = torch.cat([y_pos, x_pos], dim=2)
    return x

  def forward(
    self, inputs: torch.Tensor, patches_info: Dict[str, torch.Tensor] = None
  ) -> torch.Tensor:
    # Inputs.shape is (batch_size, num patches, emb_dim).
    assert inputs.ndim == 3, "Number of dimensions should be 3, got: %d" % inputs.ndim

    # NERF
    if "x_pos" in patches_info:
      pos_embs = self._sincos_features(patches_info)
    else:
      # Use a uniform grid between -1 and 1 describing the center of the image pixels
      # Used for grid + strided conv
      pos_embs = self.pos_embs

    # Stopping gradient here to reduce steganography (ie using locscale info to predict classes)
    pos_embs = self.layers(pos_embs.detach())
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
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape
      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
  """

  def __init__(
    self,
    num_heads: int,
    dtype = torch.float32,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    normalizer: str = "softmax",
  ):

    features = out_features  # or inputs_q.shape[-1]
    qkv_features = qkv_features  # or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, "Memory dimension must be divisible by number of heads."
    self.head_dim = qkv_features // num_heads

    self.linear_query = nn.Linear(1, num_heads * self.head_dim, bias=use_bias)
    self.linear_key = nn.Linear(1, num_heads * self.head_dim, bias=use_bias)
    self.linear_value = nn.Linear(1, num_heads * self.head_dim, bias=use_bias)
    nn.init.xavier_uniform_(self.linear_query)
    nn.init.xavier_uniform_(self.linear_key)
    nn.init.xavier_uniform_(self.linear_value)

    self.dot_product_attention = dot_product_attention


  def forward(
    self,
    inputs_q: torch.Tensor,
    inputs_kv: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
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

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """

    # project inputs_q to multi-headed q/k/v
    query: torch.Tensor = self.linear_query(inputs_q)
    key: torch.Tensor = self.linear_key(inputs_kv)
    value: torch.Tensor = self.linear_value(inputs_kv)

    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    shapes = query.shape[:-2]
    query = query.view(*shapes, -1, self.head_dim)
    key = key.view(*shapes, -1, self.head_dim)
    value = value.view(*shapes, -1, self.head_dim)

    # if self.decode:
    #   ...

    # apply attention
    x = self.attention_fn(
      query,
      key,
      value,
      mask=mask,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
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


def dot_product_attention(
  query: torch.Tensor
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[PRNGKey] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Dtype = jnp.float32,
  precision: Optional[lax.Precision] = None,
  normalizer="softmax",
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks, padding masks,
      proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`.
      This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value
      is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), "q, k, v batch dims must match."
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), "q, k, v num_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

  # compute attention weights
  attn_weights = dot_product_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    normalizer=normalizer,
  )

  # return weighted sum over values for each query position
  return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)


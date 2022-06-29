from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import ml_collections
from jax import jit


class PatchExtractor:
  """
  Takes the mask token and extracts the patches from the images in inputs
  The mask token will hopefully learn positions and resolutions of the patches.
  """

  @staticmethod
  def create_grid(num_points):
    """
    Creates a uniform grid between -1 and 1 containing num_points.
    Corresponds to the coordinate of the center of the num_points bins.
    """
    return (0.5 + jnp.arange(0, num_points)) / num_points * 2 - 1


  def __init__(
    self,
    rng,
    patches: ml_collections.ConfigDict,
    num_patches: int,
    img_dims: List[int],
    type_init_patches: str = "random",
    max_scale_mult: float = 1,
    stochastic: bool = False,
  ):
    self.rng = rng
    # Dimensions of the extracted patches
    self.patches = patches
    # Number of masks / patches extracted
    self.num_patches = num_patches
    # Dimensions of the input images, MNIST : [28, 28]
    self.img_dims = img_dims
    # Bool whether to use exhaustive grid or random patches
    self.type_init_patches = type_init_patches
    # Float defining the scale range: [1, 1 + max_scale_mult]
    self.max_scale_mult = max_scale_mult
    self.stochastic = stochastic
    self.setup()

  def setup(self):
    """Running this once to avoid creating those grids over and over"""
    # Shape [image_height]
    # Pixel coordinates in [-1, 1]
    self.h_grid = self.create_grid(self.img_dims[0])
    # Shape [image_width]
    # Pixel coordinates in [-1, 1]
    self.w_grid = self.create_grid(self.img_dims[1])

    self.fh, self.fw = self.patches.size
    # template of coordinates of the image pixels forming the center of the patch pixels
    # Will be rescaled by delta
    self.fh_grid = self.create_grid(self.fh)
    self.fw_grid = self.create_grid(self.fw)
    # Normalize patch pixel coordinates so that with delta = 0, the patch is just a zoom of the image
    # self.fw_grid should correspond exactly to the center coordinates of self.w_grid
    # (when self.img_dims[1] and self.fw have the same parity)
    self.fh_grid /= (
      self.fh_grid[-1] / self.h_grid[(self.img_dims[0] + self.fh) // 2 - 1]
    )
    self.fw_grid /= (
      self.fw_grid[-1] / self.w_grid[(self.img_dims[1] + self.fw) // 2 - 1]
    )

    self.x_pos, self.y_pos = None, None
    if self.type_init_patches in ["grid_conv", "grid_mod"]:
      h, w = (
        self.img_dims[0] // self.patches.size[0],
        self.img_dims[1] // self.patches.size[1],
      )
      # Use the basic mesh on the image and transform it appropriately.
      self.x_pos = jnp.reshape(self.create_grid(w), (1, 1, -1, 1))
      self.x_pos = jnp.tile(self.x_pos, (1, h, 1, 1))
      self.x_pos = jnp.reshape(self.x_pos, (1, -1, 1))
      self.y_pos = jnp.transpose(
        jnp.reshape(self.create_grid(h), (1, 1, -1, 1)), (0, 2, 1, 3)
      )
      self.y_pos = jnp.tile(self.y_pos, (1, 1, w, 1))
      self.y_pos = jnp.reshape(self.y_pos, (1, -1, 1))
    elif self.type_init_patches == "exhaustive":
      # Use the full mesh on the image and transform it appropriately.
      self.x_pos = jnp.reshape(self.create_grid(self.img_dims[1]), (1, 1, -1, 1))
      self.x_pos = jnp.tile(self.x_pos, (1, self.img_dims[0], 1, 1))
      self.x_pos = jnp.reshape(self.x_pos, (1, -1, 1))
      self.y_pos = jnp.transpose(
        jnp.reshape(self.create_grid(self.img_dims[0]), (1, 1, -1, 1)), (0, 2, 1, 3)
      )
      self.y_pos = jnp.tile(self.y_pos, (1, 1, self.img_dims[1], 1))
      self.y_pos = jnp.reshape(self.y_pos, (1, -1, 1))

  @partial(jit, static_argnums=0)
  def _sample_random_patches_info(self, template: jnp.ndarray):
    """Create a random loc_scales array. Used in the first pass on the inputs.
    Replaces the standard grid.
    """
    shape = template.shape

    if self.type_init_patches == "random":
      # x-coordinates of the center of the patch / mask
      # Shape [batch_size, num_patches, 1]
      _, rng = jax.random.split(self.rng)
      x_pos = jax.random.uniform(rng, shape=shape, minval=-1)
      # y-coordinates of the center of the patch / mask
      # Shape [batch_size, num_patches, 1]
      _, rng = jax.random.split(rng)
      y_pos = jax.random.uniform(rng, shape=shape, minval=-1)
      # Scale of the patch
      # Shape [batch_size, num_patches, 1]
      self.rng, rng = jax.random.split(rng)
      scale = jax.random.uniform(rng, shape=shape)
      return x_pos, y_pos, scale

    x_pos = jnp.tile(self.x_pos, (shape[0], 1, 1))
    y_pos = jnp.tile(self.y_pos, (shape[0], 1, 1))

    if self.type_init_patches == "exhaustive":
      # Return patches at each pixel location
      scale = jnp.zeros(x_pos.shape) + 0.5
    elif self.type_init_patches == "grid_conv":
      scale = jnp.zeros(x_pos.shape)
    else:
      # x-coordinates of the center of the patch / mask
      # Shape [batch_size, num_patches, 1]
      _, rng = jax.random.split(self.rng)
      x_pos = x_pos + 2 / self.img_dims[1] * jax.random.uniform(
        rng, shape=shape, minval=-1
      )
      # y-coordinates of the center of the patch / mask
      # Shape [batch_size, num_patches, 1]
      _, rng = jax.random.split(rng)
      y_pos = y_pos + 2 / self.img_dims[0] * jax.random.uniform(
        rng, shape=shape, minval=-1
      )
      # Scale of the patch
      # Shape [batch_size, num_patches, 1]
      self.rng, rng = jax.random.split(rng)
      scale = jax.random.uniform(rng, shape=shape)

    return x_pos, y_pos, scale

  @partial(jit, static_argnums=0)
  def __call__(
    self,
    inputs: jnp.ndarray,
    x_pos: jnp.ndarray = jnp.zeros(()),
    y_pos: jnp.ndarray = jnp.zeros(()),
    scale: jnp.ndarray = jnp.zeros(()),
  ) -> jnp.ndarray:
    """
    Extracts patches from inputs.
    If x_pos, y_pos and scale are provided, use those values.
    If not, and self.type_init_patches is True, extract patches at random locations and scales.
    Otherwise, extract patches using the exhaustive grid and a scale of 0 (ie no zooming and no blurring).
    """
    # Batch size, height = self.img_dims[0], width = self.img_dims[1], channel s
    n, _, _, _ = inputs.shape

    if not len(x_pos.shape):
      template = jnp.zeros((n, self.num_patches, 1))
      x_pos, y_pos, scale = self._sample_random_patches_info(template)

    # # Manual positions that "make sense"
    # x_pos = jnp.array([[[-0.26], [0.24], [-0.26], [0.24]]])
    # x_pos = jnp.tile(x_pos, (inputs.shape[0], 1, 1))
    # y_pos = jnp.array([[[0.24], [-0.37], [-0.37], [0.24]]])
    # y_pos = jnp.tile(y_pos, (inputs.shape[0], 1, 1))
    # scale = jnp.array([[[1], [1], [1], [1]]])
    # scale = jnp.tile(scale, (inputs.shape[0], 1, 1))

    # Rescale scale
    scale *= self.max_scale_mult

    # Shape [batch_size, num_patches, 1, 1]
    # Fuzziness of the patch
    # The 0.4 factor can be tuned. It's the fraction of the distance between gaussian centers corresponding to one stdev.
    precision = (2 * (0.4 * 2 * (1 + scale) / self.img_dims[0]) ** 2).reshape(
      (n, self.num_patches, 1, 1)
    )
    # Shape [batch_size, num_patches, patch_size, 1]. x-coord of the centers of the extracted patches
    mu_x = (x_pos + self.fw_grid * (1 + scale)).reshape(
      (n, self.num_patches, self.fw, 1)
    )
    # Shape [batch_size, num_patches, patch_size, 1]. y-coord of the centers of the extracted patches
    mu_y = (y_pos + self.fh_grid * (1 + scale)).reshape(
      (n, self.num_patches, self.fh, 1)
    )

    # Shape [batch_size, num_patches, patch_size, image_width].
    # Normalized pixel weights on the x axis for each filters of each patch of each mask of each input image
    F_X = jnp.exp(-((self.w_grid - mu_x) ** 2) / precision)
    F_X = F_X / F_X.sum(axis=-1, keepdims=True, initial=1e-8)
    # Shape [batch_size, num_patches, patch_size, image_height].
    # Normalized peights on the y axis for each filters of each patch of each mask of each input image
    F_Y = jnp.exp(-((self.h_grid - mu_y) ** 2) / precision)
    F_Y = F_Y / F_Y.sum(axis=-1, keepdims=True, initial=1e-8)

    # Summing over x-axis
    extracted_patches = jnp.einsum("nhwc,nmpw->nmphc", inputs, F_X)
    # Summing over y-axis
    extracted_patches = jnp.einsum("nmphc,nmqh->nmqpc", extracted_patches, F_Y)
    return dict(
      patches=extracted_patches,
      x_pos=x_pos,
      y_pos=y_pos,
      scale=scale,
      precision=precision,
    )

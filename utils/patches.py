import torch

from config import Config, Enums


class PatchExtractor:
  """
  Takes the mask token and extracts the patches from the images in inputs
  The mask token will hopefully learn positions and resolutions of the patches.
  """

  @staticmethod
  def create_grid(num_points: int):
    """
    Creates a uniform grid between -1 and 1 containing num_points.
    Corresponds to the coordinate of the center of the num_points bins.
    """
    return (0.5 + torch.arange(0, num_points)) / num_points * 2. - 1.

  def __init__(
    self,
    config: Config,
    num_patches: int,
    stochastic: bool = False,
  ):
    self.config = config

    # Number of masks / patches extracted
    self.num_patches = num_patches
    # Float defining the scale range: [1, 1 + max_scale_mult]
    self.max_scale_mult = config.patches.max_scale_mult
    self.stochastic = stochastic
    self.setup()

  def setup(self):
    """
    Running this once to avoid creating those grids over and over.
    """
    self.h_grid = self.create_grid(self.config.dataset.input_size)
    self.w_grid = self.create_grid(self.config.dataset.input_size)

    # Template of coordinates of the image pixels forming the center of the patch pixels.
    # Will be rescaled by delta
    self.fh_grid = self.create_grid(self.config.patches.size)
    self.fw_grid = self.create_grid(self.config.patches.size)

    # Normalize patch pixel coordinates so that with delta = 0, the patch is just a zoom of the image
    # self.fw_grid should correspond exactly to the center coordinates of self.w_grid
    # (when self.config.dataset.input_size and self.config.patches.size have the same parity)
    self.fh_grid /= (
      self.fh_grid[-1] / self.h_grid[(self.config.dataset.input_size + self.config.patches.size) // 2 - 1]
    )
    self.fw_grid /= (
      self.fw_grid[-1] / self.w_grid[(self.config.dataset.input_size + self.config.patches.size) // 2 - 1]
    )

    self.x_pos, self.y_pos = None, None
    if self.config.patches.init_type in (Enums.InitPatches.grid_conv, Enums.InitPatches.grid_mod):
      h, w = (
        self.config.dataset.input_size // self.config.patches.size,
        self.config.dataset.input_size // self.config.patches.size,
      )
      # Use the basic mesh on the image and transform it appropriately.
      self.x_pos = torch.reshape(self.create_grid(w), (1, 1, -1, 1))
      self.x_pos = torch.tile(self.x_pos, (1, h, 1, 1))
      self.x_pos = torch.reshape(self.x_pos, (1, -1, 1))
      self.y_pos = torch.transpose(
        torch.reshape(self.create_grid(h), (1, 1, -1, 1)), (0, 2, 1, 3)
      )
      self.y_pos = torch.tile(self.y_pos, (1, 1, w, 1))
      self.y_pos = torch.reshape(self.y_pos, (1, -1, 1))

    elif self.config.patches.init_type is Enums.InitPatches.exhaustive:
      # Use the full mesh on the image and transform it appropriately.
      self.x_pos = torch.reshape(self.create_grid(self.config.dataset.input_size), (1, 1, -1, 1))
      self.x_pos = torch.tile(self.x_pos, (1, self.config.dataset.input_size, 1, 1))
      self.x_pos = torch.reshape(self.x_pos, (1, -1, 1))
      self.y_pos = torch.transpose(
        torch.reshape(self.create_grid(self.config.dataset.input_size), (1, 1, -1, 1)), (0, 2, 1, 3)
      )
      self.y_pos = torch.tile(self.y_pos, (1, 1, self.config.dataset.input_size, 1))
      self.y_pos = torch.reshape(self.y_pos, (1, -1, 1))

  def _sample_random_patches_info(self, template):
    """Create a random loc_scales array. Used in the first pass on the inputs.
    Replaces the standard grid.

    Returns:
      x_pos: x-coordinates of the center of the patch / mask, [batch_size, num_patches, 1]  (-1, 1)
      y_pos: y-coordinates of the center of the patch / mask, [batch_size, num_patches, 1]  (-1, 1)
      scale: Scale of the patch in (0, 1), [batch_size, num_patches, 1]  (0, 1)
    """
    bs = template.shape[0]

    if self.config.patches.init_type is Enums.InitPatches.random:
      x_pos = torch.rand_like(template) * 2. - 1.
      y_pos = torch.rand_like(template) * 2. - 1.
      scale = torch.rand_like(template)
      return x_pos, y_pos, scale

    x_pos = torch.tile(self.x_pos, (bs, 1, 1))
    y_pos = torch.tile(self.y_pos, (bs, 1, 1))

    if self.config.patches.init_type is Enums.InitPatches.exhaustive:
      # Return patches at each pixel location
      scale = torch.zeros_like(x_pos) + 0.5
    elif self.config.patches.init_type is Enums.InitPatches.grid_conv:
      scale = torch.zeros_like(x_pos)
    else:
      x_pos += 2 / self.config.dataset.input_size * torch.rand_like(template) * 2. - 1.
      y_pos += 2 / self.config.dataset.input_size * torch.rand_like(template) * 2. - 1.
      scale = torch.rand_like(template)

    return x_pos, y_pos, scale

  def forward(
    self,
    inputs: torch.Tensor,
    x_pos: torch.Tensor = None,
    y_pos: torch.Tensor = None,
    scale: torch.Tensor = None,
  ) -> torch.Tensor:
    """
    Extracts patches from inputs.
    If x_pos, y_pos and scale are provided, use those values.
    If not, and self.type_init_patches is True, extract patches at random locations and scales.
    Otherwise, extract patches using the exhaustive grid and a scale of 0 (ie no zooming and no blurring).
    """
    # Batch size, height = self.img_dims[0], width = self.img_dims[1], channel s
    n, _, _, _ = inputs.shape

    x_pos = x_pos or torch.zeros()
    y_pos = y_pos or torch.zeros()
    scale = scale or torch.zeros()

    if not len(x_pos.shape):
      template = torch.zeros((n, self.num_patches, 1))
      x_pos, y_pos, scale = self._sample_random_patches_info(template)

    # # Manual positions that "make sense"
    # x_pos = torch.Tensor([[[-0.26], [0.24], [-0.26], [0.24]]])
    # x_pos = torch.tile(x_pos, (inputs.shape[0], 1, 1))
    # y_pos = torch.Tensor([[[0.24], [-0.37], [-0.37], [0.24]]])
    # y_pos = torch.tile(y_pos, (inputs.shape[0], 1, 1))
    # scale = torch.Tensor([[[1], [1], [1], [1]]])
    # scale = torch.tile(scale, (inputs.shape[0], 1, 1))

    # Rescale scale
    scale *= self.max_scale_mult

    # Shape [batch_size, num_patches, 1, 1]
    # Fuzziness of the patch
    # The 0.4 factor can be tuned. It's the fraction of the distance between gaussian centers corresponding to one stdev.
    precision = (2 * (0.4 * 2 * (1 + scale) / self.config.dataset.input_size) ** 2).reshape(
      (n, self.num_patches, 1, 1)
    )
    # Shape [batch_size, num_patches, patch_size, 1]. x-coord of the centers of the extracted patches
    mu_x = (x_pos + self.fw_grid * (1 + scale)).reshape(
      (n, self.num_patches, self.config.patches.size, 1)
    )
    # Shape [batch_size, num_patches, patch_size, 1]. y-coord of the centers of the extracted patches
    mu_y = (y_pos + self.fh_grid * (1 + scale)).reshape(
      (n, self.num_patches, self.config.patches.size, 1)
    )

    # Shape [batch_size, num_patches, patch_size, image_width].
    # Normalized pixel weights on the x axis for each filters of each patch of each mask of each input image
    F_X = torch.exp(-((self.w_grid - mu_x) ** 2) / precision)
    F_X /= F_X.sum(dim=-1, keepdims=True) + 1e-8
    # Shape [batch_size, num_patches, patch_size, image_height].
    # Normalized peights on the y axis for each filters of each patch of each mask of each input image
    F_Y = torch.exp(-((self.h_grid - mu_y) ** 2) / precision)
    F_Y /= F_Y.sum(dim=-1, keepdims=True) + 1e-8

    # Summing over x-axis
    extracted_patches = torch.einsum("nhwc,nmpw->nmphc", inputs, F_X)
    # Summing over y-axis
    extracted_patches = torch.einsum("nmphc,nmqh->nmqpc", extracted_patches, F_Y)
    return dict(
      patches=extracted_patches,
      x_pos=x_pos,
      y_pos=y_pos,
      scale=scale,
      precision=precision,
    )


def select_patches(patches_info, num_patches):
  """Select num_patches patches from dict containing all extracted patches"""
  return dict(
    patches=patches_info["patches"][:, :num_patches],
    x_pos=patches_info["x_pos"][:, :num_patches],
    y_pos=patches_info["y_pos"][:, :num_patches],
    scale=patches_info["scale"][:, :num_patches],
    precision=patches_info["precision"][:, :num_patches],
  )

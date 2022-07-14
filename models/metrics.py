
from typing import Optional, Union

import torch


class Metrics:
  @staticmethod
  def num_examples(
    logits: torch.Tensor,
    one_hot_targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
  ) -> Union[torch.Tensor, int]:
    del logits
    if weights is None:
      return one_hot_targets.shape[0]
    return weights.sum()

  @staticmethod
  def weighted_correctly_classified(
    logits: torch.Tensor,
    one_hot_targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Computes weighted number of correctly classified over the given batch.

    This computes the weighted number of correctly classified examples/pixels in a
    single, potentially padded minibatch. If the minibatch/inputs is padded (i.e.,
    it contains null examples/pad pixels) it is assumed that weights is a binary
    mask where 0 indicates that the example/pixel is null/padded. We assume the
    trainer will aggregate and divide by number of samples.

    Args:
    logits: Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

    Returns:
      The number of correctly classified examples in the given batch.
    """
    if logits.ndim != one_hot_targets.ndim:
      raise ValueError(
        "Incorrect shapes. Got shape %s logits and %s one_hot_targets"
        % (str(logits.shape), str(one_hot_targets.shape))
      )
    preds = torch.argmax(logits, dim=-1)
    targets = torch.argmax(one_hot_targets, dim=-1)
    correct = torch.equal(preds, targets)

    if weights is not None:
      correct = apply_weights(correct, weights)

    return correct.astype(torch.int32)

  @staticmethod
  def weighted_unnormalized_softmax_cross_entropy(
    logits: torch.Tensor,
    one_hot_targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[torch.Tensor] = None,
    logits_normalized: bool = False,
  ) -> torch.Tensor:
    """Computes weighted softmax cross entropy give logits and targets.

    This computes sum_(x,y) softmax-ce(x, y) for a single, potentially padded
    minibatch. If the minibatch is padded (that is it contains null examples)
    it is assumed that weights is a binary mask where 0 indicates that the
    example is null.

    Args:
      logits: Output of model in shape [batch, ..., num_classes].
      one_hot_targets: One hot vector of shape [batch, ..., num_classes].
      weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
      label_smoothing: Scalar to use to smooth the one-hot labels.
      label_weights: Weight per label of shape [num_classes].
      logits_normalized: If True, the logits are assumed to already be normalized.

    Returns:
      The softmax cross entropy of the examples in the given batch.
    """
    if logits.ndim != one_hot_targets.ndim:
      raise ValueError(
        "Incorrect shapes. Got shape %s logits and %s one_hot_targets"
        % (str(logits.shape), str(one_hot_targets.shape))
      )

    # Optionally apply label smoothing.
    if label_smoothing is not None:
      one_hot_targets = apply_label_smoothing(one_hot_targets, label_smoothing)

    # Optionally apply label weights.
    if label_weights is not None:
      one_hot_targets *= label_weights

    if not logits_normalized:
      logits = _logprob(logits)
    loss = -torch.einsum("...k,...k->...", one_hot_targets, logits)
    if weights is not None:
      loss = apply_weights(loss, weights)

    return loss


def apply_weights(output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  """Applies given weights of the inputs in the minibatch to outputs.

  Note that weights can be per example (i.e. of shape `[batch,]`) or per
  pixel/token (i.e. of shape `[batch, height, width]` or
  `[batch, len]`) so we need to broadcast it to the output shape.

  Args:
    output: Computed output, which can be loss or the correctly
      classified examples, etc.
    weights: Weights of inputs in the batch, which can be None or
      array of shape [batch, ...].

  Returns:
    Weighted output.
  """
  desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
  weights = torch.broadcast_to(weights, desired_weights_shape)  # broadcast_dimensions=tuple(range(weights.ndim))

  # Scale the outputs with weights.
  return output * weights


def apply_label_smoothing(
  one_hot_targets: torch.Tensor, label_smoothing: Optional[float]
) -> torch.Tensor:
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html

  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def _logprob(x):
  """Log of the Softmax."""
  return x - torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))


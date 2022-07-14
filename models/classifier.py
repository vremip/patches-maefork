

import functools
from typing import Dict, Optional, Tuple
from immutabledict import immutabledict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import Metrics


_CLASSIFICATION_METRICS = immutabledict(
  {
    "accuracy": (Metrics.weighted_correctly_classified, Metrics.num_examples),
    "loss": (
      Metrics.weighted_unnormalized_softmax_cross_entropy,
      Metrics.num_examples,
    ),
  }
)


class ClassifierModel(nn.Module):
  def get_metrics_fn(self, split: Optional[str] = None):
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
      classification_metrics_function,
      metrics=_CLASSIFICATION_METRICS,
    )


def classification_metrics_function(
  *,
  logits: torch.Tensor,
  inputs: torch.Tensor,
  labels: torch.Tensor,
  masks: torch.Tensor,
  target_is_onehot: bool = False,
  metrics = _CLASSIFICATION_METRICS,
) -> Dict[str, Tuple[float, int]]:
  """Calculates metrics for the classification task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_onehot: If the target is a one-hot vector.
   metrics: The classification metrics to evaluate. The key is the name of the
     metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_onehot:
    one_hot_targets = labels
  else:
    one_hot_targets = F.one_hot(labels, logits.shape[-1])
  weights = masks  # batch_mask might not be defined

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = psum_metric_normalizer(
      (
        val[0](logits, one_hot_targets, weights),
        val[1](logits, one_hot_targets, weights),
      )
    )
  return evaluated_metrics


def psum_metric_normalizer(
  metrics: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies psum over the given tuple of (metric, normalizer)."""
  
  # TODO distributed?
  psumed_metric = torch.sum(metrics[0])
  psumed_normalizer = torch.sum(metrics[1])
  return (psumed_metric, psumed_normalizer)

import random
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from config import Config, Enums
from models.metrics import _logprob
from models.mvit_encoder import Encoder
from models.weights import init_weights

from .mvit import Classifier, MViT


class MViTPolicy(MViT):
  """Learn a policy extracting patches based on their scores."""

  def __init__(self, config: Config, args: Namespace):
    super().__init__(config, args)

    # Trainable vector from which the first loc and scale are extracted
    self.shape = (1, 1) if config.misc.per_patch_token else (1,)
    self.emb_patches = Parameter(torch.randn(self.shape + (config.model.hidden_size,), dtype=config.model.dtype))

    self.locscale_embeds = nn.Sequential(
      nn.Linear(7, config.model.hidden_size),
      nn.ReLU(),
      nn.Linear(config.model.hidden_size, config.model.hidden_size),
    )
    init_weights(self.locscale_embeds, torch.nn.init.xavier_uniform_)

    self.prefix_embeds_mat = nn.Linear(7, config.model.hidden_size)
    init_weights(self.prefix_embeds_mat, torch.nn.init.xavier_uniform_)

    output_projection = nn.Linear(7, config.dataset.num_classes)
    init_weights(output_projection, torch.nn.init.zeros_)

    self.classifier2 = Classifier(
      representation_size=config.model.representation_size,
      num_classes=config.dataset.num_classes,
      output_projection=output_projection,
    )
    self.encoder2 = Encoder(**self.encoder_kwargs)

  def _get_locscale_embeds_for_score(self, patches: dict):
    """MLP to project location and scale into an RKHS where the score for the patches is computed"""
    locscale = torch.cat(
      (patches["x_pos"], patches["y_pos"], patches["scale"]), dim=2
    )
    return self.locscale_embeds(locscale)

  def _computes_scores(self, patches: dict, prefix_embeds: torch.Tensor):
    locscale_embeds = self._get_locscale_embeds_for_score(patches)
    # n: batch size
    # d: representation dimension (aka. self.hidden_size)
    # p: number of candidates
    return torch.einsum("nd,npd->np", prefix_embeds, locscale_embeds)

  def _get_indices(self, locscale_embeds: torch.Tensor, prefix_embeds: torch.Tensor, epsilon=0.0):
    scores = torch.einsum("ni,npi->np", prefix_embeds, locscale_embeds)
    return scores, torch.argmax(scores, dim=1)
    # Two choices:
    # i. either argmax or random for the whole batch
    # ii. decide per image

    b = jax.random.bernoulli(self.make_rng("patch_extractor"), epsilon)
    indices = jax.lax.cond(
      b,
      lambda: torch.zeros((scores.shape[0]), dtype="int32"),
      lambda: torch.argmax(scores, dim=1),
    )
    return scores, indices

  def _select_patches_and_columns(
    self, indices, candidates, columns, logits, patches_embeds
  ):
    new_patches = dict()
    for k, v in candidates.items():
      new_patches[k] = v[indices]
    new_columns = columns[:, indices]
    new_logits = logits[indices]
    score_embeds = patches_embeds[indices]
    return new_patches, new_columns, new_logits, score_embeds

    # one_hot = jax.nn.one_hot(indices, num_classes=logits.shape[1], dim=1)
    # new_patches = dict()
    # for k, v in candidates.items():
    #   new_patches[k] = torch.einsum("np...,np->n...", v, one_hot)
    # new_columns = torch.einsum("hnpi,np->hni", columns, one_hot)
    # new_logits = torch.einsum("npi,np->ni", logits, one_hot)
    # score_embeds = torch.einsum("npi,np->ni", patches_embeds, one_hot)
    # return new_patches, new_columns, new_logits, score_embeds

  def forward(self, inputs: torch.Tensor, *, loss_eval: bool = False):
    batch_size = inputs.shape[0]

    # Get "output tokens" pertaining to locscale
    emb_patches = torch.tile(self.emb_patches, (batch_size, *self.shape))  # bs x d (self.hidden_size)
    locscale_embeds = self._get_locscale_embeds(emb_patches, extract=False)
    locscale_params = self.locscale_extractor(locscale_embeds)
    patches, _ = self._extract_patches(inputs, locscale_params)
    # Process first patch
    initial_logits, emb_patches, embeddings = self._single_pass(patches_info=patches)
    # Get embeddings for scores:
    prefix_embeds = emb_patches[:, -1]

    # What is faster, dynamic updates or concatenation along an empty axis?
    all_logits = torch.zeros((batch_size, self.config.model.autoreg_passes, self.config.patches.num_patches_init + 1, self.config.dataset.num_classes))
    all_scores = torch.zeros((batch_size, self.config.model.autoreg_passes, self.config.patches.num_patches_init + 1))

    # Used for visualization
    all_candidates = []
    heatmap_scores = []
    all_patches = None

    # All possible patches. Used to create heatmap and select next patch
    all_patches = self._get_all_patches(inputs)
    locscale_embeds = self._get_locscale_embeds_for_score(all_patches)

    if self.training:
      epsilon = random.choice([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:
      epsilon = 0.0

    # selected_logits contains the logits of patches actually selected
    selected_logits = torch.unsqueeze(initial_logits, dim=1)
    # prev_columns contains columns of tokens processed in previous iterations
    prev_columns = self._get_patch_embeds(embeddings)
    for pass_idx in range(self.config.model.autoreg_passes):

      # Embed prefix to predict scores
      prefix_embeds = self.prefix_embeds_mat(prefix_embeds)

      # Sample candidate patches
      candidates = self._get_init_patches(inputs, self.config.patches.num_patches_init - pass_idx - 1)
      candidates = merge_patches_info(candidates, patches)

      # Compute scores of all possible extensions, and select the best one
      heatmap = torch.einsum("ni,npi->np", prefix_embeds, locscale_embeds)
      indices = torch.argmax(heatmap, dim=1)
      if loss_eval:
        heatmap_scores.append(heatmap)
        all_candidates.append(candidates)

      # Add best patch to candidates
      one_hot = F.one_hot(indices, num_classes=locscale_embeds.shape[1])
      new_patches = dict()
      for k, v in candidates.items():
        p = torch.unsqueeze(torch.einsum("np...,np->n...", all_patches[k], one_hot), dim=1)
        new_patches[k] = torch.cat((v, p), dim=1)

      # Compute scores of all candidates and best patch. Used to train the score.
      scores = self._computes_scores(new_patches, prefix_embeds)
      # Put scores aside for optimization
      scores = torch.unsqueeze(scores, dim=1)
      all_scores[:, pass_idx] = scores  ### all_scores = jax.lax.dynamic_update_slice(all_scores, scores, (0, pass_idx, 0))

      # Tile inputs and process all candidates + best patch in single pass
      for k, v in new_patches.items():
        # Use order 'F' to match the tiling
        new_patches[k] = torch.reshape(v, (batch_size * (self.config.patches.num_patches_init + 1), 1) + v.shape[2:], order="F")

      flat_columns = torch.tile(prev_columns, (1, self.config.patches.num_patches_init + 1, 1, 1))
      logits, emb_patches, columns = self._single_pass(
        patches_info=new_patches, prev_columns=flat_columns
      )

      # Pick the proper indices after tiling
      # Either random patches, corresponding to b = 0
      # Either the best patches, corresponding to b = 1
      # The best patches have been appended at the end of "new_patches"
      b = random.uniform() < epsilon
      random_or_best_idx = (1 - b) * (batch_size * self.config.patches.num_patches_init)  # 0 or start of sequence of best patches
      tiled_indices = torch.arange(batch_size) + random_or_best_idx
      (
        new_patches,
        new_columns,
        new_logits,
        prefix_embeds,
      ) = self._select_patches_and_columns(
        tiled_indices,
        new_patches,
        self._get_patch_embeds(columns),
        logits,
        emb_patches[:, -1],
      )

      logits = torch.reshape(logits, (batch_size, self.config.patches.num_patches_init + 1) + logits.shape[1:])  # order="F"

      ### all_logits = jax.lax.dynamic_update_slice(all_logits, torch.unsqueeze(logits, dim=1), (0, pass_idx, 0, 0))
      all_logits[:, pass_idx] = torch.unsqueeze(logits, dim=1)

      # Add computation corresponding to chosen patches
      prev_columns = torch.cat((prev_columns, new_columns), dim=2)
      # Add selected logits for accuracy reporting
      selected_logits = torch.cat(
        (selected_logits, torch.unsqueeze(new_logits, dim=1)), dim=1
      )

      # Add selected patches for visualization purposes
      patches = merge_patches_info(patches, new_patches)

    emb_patches2, _ = self.encoder2(
      patches_info=patches, batch_size=batch_size
    )

    # Classification token, trained via CE
    if self.config.model.classifier == Enums.ClassifierTypes.token:
      logits2 = self.classifier2(emb_patches2[:, : self.config.dataset.num_labels])
      if self.config.dataset.num_labels == 1:
        logits2 = torch.squeeze(logits2, dim=1)
    else:
      # If no token, use output of last column
      logits2 = self.classifier2(emb_patches2[:, -1])

    return (
      all_logits,
      selected_logits,
      all_scores,
      patches,
      (all_candidates, heatmap_scores, all_patches),
      logits2,
    )

  def training_step(self, inputs: torch.Tensor, labels: torch.Tensor, args):
    all_logits, selected_logits, scores, _, _, logits2 = self.forward(
      inputs,
      #mutable=["batch_stats"],
      loss_eval=getattr(args, "loss_eval", None),
    )

    one_hot_targets = F.one_hot(labels, all_logits.shape[-1])
    log_probs = _logprob(all_logits)
    # n: batch size
    # p: num of autoreg passes
    # q: num of candidates
    preds = torch.einsum("npqi,ni->npq", log_probs, one_hot_targets)
    # Cross-entropy loss on all prefixes and possible extensions
    full_loss = torch.mean(-preds) - torch.mean(torch.einsum("ni,ni->n", _logprob(selected_logits[:, 0]), one_hot_targets))
    # Cross-entropy for second transformer
    full_loss -= torch.mean(torch.einsum("ni,ni->n", _logprob(logits2), one_hot_targets))

    metrics = []
    for idx in range(self.config.model.autoreg_passes + 1):
      metric = self.metric_fn(selected_logits[:, idx], inputs, labels)
      metrics.append(metric)
    # Add metric for second transformer
    metric = self.metric_fn(logits2, inputs, labels)
    metrics.append(metric)

    scores_mat = torch.unsqueeze(scores, dim=3) - torch.unsqueeze(scores, dim=2)
    if self.config.training.loss == Enums.Losses.l2:
      pred_mat = (torch.unsqueeze(preds, dim=3) - torch.unsqueeze(preds, dim=2)).detach()
      score_loss = torch.mean((scores_mat - pred_mat) ** 2)
    else:
      pred_mat = (torch.sign(torch.unsqueeze(preds, dim=3) - torch.unsqueeze(preds, dim=2))).detach()

      # Margin loss
      if self.config.training.loss == Enums.Losses.margin:
        score_loss = torch.mean(F.relu(1 - pred_mat * scores_mat))
      # # Softplus loss
      elif self.config.training.loss == Enums.Losses.softplus:
        score_loss = torch.mean(torch.log(1 + torch.exp(-pred_mat * scores_mat)))
      else:
        raise NotImplementedError()
      # correc = nn.ReLU()(torch.sign(scores_mat * pred_mat))
    full_loss += score_loss
    full_loss += 0.01 * torch.mean(scores**2)
    metrics.append(dict(score_loss=(score_loss, 1)))


def merge_patches_info(
  patches_info: Dict[str, torch.Tensor],
  new_patches_info: Dict[str, torch.Tensor],
):
  """Merge patches from new_patches_info into patches_info"""
  if not patches_info:
    return new_patches_info

  if len(patches_info["patches"].shape) == len(new_patches_info["patches"].shape) + 1:
    # Single patch being added
    _process = lambda x: torch.unsqueeze(x, dim=1)
  else:
    _process = lambda x: x
  patches_info["patches"] = torch.cat((patches_info["patches"], _process(new_patches_info["patches"])), dim=1)
  patches_info["x_pos"] = torch.cat((patches_info["x_pos"], _process(new_patches_info["x_pos"])), dim=1)
  patches_info["y_pos"] = torch.cat((patches_info["y_pos"], _process(new_patches_info["y_pos"])), dim=1)
  patches_info["scale"] = torch.cat((patches_info["scale"], _process(new_patches_info["scale"])), dim=1)
  patches_info["precision"] = torch.cat((patches_info["precision"], _process(new_patches_info["precision"])), dim=1)
  return patches_info

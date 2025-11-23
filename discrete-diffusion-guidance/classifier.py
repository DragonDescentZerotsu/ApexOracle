import itertools
import typing

import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import transformers

import dataloader
import models.dit
import noise_schedule


class MicroAveragingMetric(torchmetrics.Metric):
  """Micro-averaging metric.

    Adapted from https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py#L12
  """

  def __init__(self, class_idx: typing.Optional[int] = 1,
               dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.class_idx = torch.tensor(class_idx) \
      if class_idx is not None else None
    self.add_state("numerator", default=torch.tensor(0.0),
                   dist_reduce_fx="sum")
    self.add_state("denominator", default=torch.tensor(0.0),
                   dist_reduce_fx="sum")

  def _update(
      self, numerator, denominator, preds, y) -> tuple:
    raise NotImplementedError

  def update(self, logits: torch.Tensor, y: torch.Tensor):
    # update metric states
    preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    assert preds.shape == y.shape, \
      f"preds shape {preds.shape} != y shape {y.shape}"
    self.numerator, self.denominator = self._update(
      self.numerator, self.denominator, preds, y)

  def compute(self):
    # compute final result
    value = self.numerator.float() / self.denominator \
      if self.denominator.item() > 0. else torch.tensor(0.0)
    return value

  def reset(self):
    self.numerator = torch.tensor(0.0).to(self.device)
    self.denominator = torch.tensor(0.0).to(self.device)


class CrossEntropy(MicroAveragingMetric):
  """Calculates cross-entropy loss."""
  def _update(
      self, numerator, denominator, logits, y) -> tuple:
    with torch.no_grad():
      numerator += F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=-100,
        reduction='sum')
      denominator += y.numel()
    return numerator, denominator

  # Overrides parent class to use logits and not (argmax) preds
  def update(self, logits: torch.Tensor, y: torch.Tensor):
    y = y.view(-1)
    self.numerator, self.denominator = self._update(
      self.numerator, self.denominator, logits, y)


class Accuracy(MicroAveragingMetric):
  """Calculates accuracy.

    Can be used to calculate accuracy per class.
    Copied from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(
      self, numerator, denominator, preds, y) -> tuple:
    if self.class_idx is None:
      numerator += (preds == y).sum()
      denominator += y.numel()
    else:
      class_idx = self.class_idx
      relevant_idxs = (y == class_idx)
      numerator += (preds[relevant_idxs] == class_idx).sum()
      denominator += relevant_idxs.sum()
      relevant_idxs = (y != class_idx)
      numerator += (preds[relevant_idxs] != class_idx).sum()
      denominator += relevant_idxs.sum()
    return numerator, denominator


class Precision(MicroAveragingMetric):
  """Calculates precision.

    Can be used to calculate precision per class.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    relevant_idxs = (preds == class_idx)
    numerator += (y[relevant_idxs] == class_idx).sum()
    denominator += relevant_idxs.sum()
    return numerator, denominator


class Recall(MicroAveragingMetric):
  """Calculate recall.

    Can be used to calculate recall per class.
    Adapted from:
      https://github.com/HazyResearch/hyena-dna/blob/main/src/tasks/metrics.py
  """

  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    relevant_idxs = (y == class_idx)
    numerator += (preds[relevant_idxs] == class_idx).sum()
    denominator += relevant_idxs.sum()
    return numerator, denominator


class Classifier(L.LightningModule):
  def __init__(
      self,
      config,
      tokenizer: transformers.PreTrainedTokenizer,
      pretrained_backbone: typing.Optional[torch.nn.Module] = None):
    super().__init__()
    self.save_hyperparameters(ignore=['pretrained_backbone'])
    self.config = config

    # This param indicates whether this model will be used
    #  for guidance (False) or only evaluation (True).
    self.is_eval_classifier = getattr(
      config, 'is_eval_classifier', False)

    self.tokenizer = tokenizer
    self.vocab_size = tokenizer.vocab_size
    self.antithetic_sampling = config.training.antithetic_sampling
    self.importance_sampling = config.training.importance_sampling
    self.change_of_variables = config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id

    if config.classifier_backbone == 'dit':
      self.classifier_model = models.dit.DITClassifier(self.config, vocab_size=self.vocab_size)
    elif self.config.classifier_backbone == 'dit_AMP':
      self.classifier_model = models.dit.DITClassifier_AMP(self.config, vocab_size=len(self.tokenizer.get_vocab()))
    elif self.config.classifier_backbone == 'dit_reg_cls_AMP':
      self.classifier_model = models.dit.DIT_Reg_Cls_AMP(self.config, vocab_size=len(self.tokenizer.get_vocab()))
    elif self.config.classifier_backbone == 'dit_synergy_cls_AMP':
      self.classifier_model = models.dit.DIT_Syn_Cls_Pep_Cls_AMP(self.config, vocab_size=len(self.tokenizer.get_vocab()))
    elif self.config.classifier_backbone == 'dimamba':
      self.classifier_model = models.dimamba.DiMambaClassifier(
        self.config, vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif config.classifier_backbone == 'hyenadna':
      hyena_config = transformers.AutoConfig.from_pretrained(
        config.classifier_model.hyena_model_name_or_path,
        n_layer=config.classifier_model.n_layer,
        trust_remote_code=True
      )
      self.classifier_model = transformers.AutoModelForSequenceClassification.from_config(
        hyena_config,
        pretrained=False,
        num_labels=config.data.num_classes,
        problem_type='single_label_classification',
        trust_remote_code=True
      )
    else:
      raise NotImplementedError(
        f"Classifier backbone "
        f"{self.config.classifier_backbone} not "
        f"implemented.")
    if pretrained_backbone is not None:  # For PPLM / NOS
      self.classifier_model.load_pretrained_encoder(
        pretrained_backbone)
    # Metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'cross_entropy': CrossEntropy(),
      'accuracy': Accuracy(class_idx=None),
    })
    if config.data.num_classes > 2:
      for c in range(config.data.num_classes):
        metrics.add_metrics(
          {f"accuracy_class{c}": Accuracy(class_idx=c),
           f"precision_class{c}": Precision(class_idx=c),
           f"recall_class{c}": Recall(class_idx=c)})
    else:
      metrics.add_metrics(
        {'precision': Precision(class_idx=1),
         'recall': Recall(class_idx=1)})
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')

    self.T = config.T
    self.noise = noise_schedule.get_noise(config,dtype=self.dtype)
    self.sampling_eps = config.training.sampling_eps
    self.lr = config.optim.lr
    self.time_conditioning = config.time_conditioning
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def on_load_checkpoint(self, checkpoint):
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
      'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is
    #  1 iteration behind, so we're using the optimizer's
    #  progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
      'completed'] = checkpoint['loops']['fit_loop'][
                       'epoch_loop.automatic_optimization.optim_progress'][
                       'optimizer']['step']['total'][
                       'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
      'completed'] = checkpoint['loops']['fit_loop'][
                       'epoch_loop.automatic_optimization.optim_progress'][
                       'optimizer']['step']['current'][
                       'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global
    # steps, not the number of local steps, so we don't
    # multiply with self.trainer.accumulate_grad_batches
    # here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
      '_batches_that_stepped'] = \
    checkpoint['loops']['fit_loop'][
      'epoch_loop.automatic_optimization.optim_progress'][
      'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer. \
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
        'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
        self.trainer._accelerator_connector.use_distributed_sampler
        and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=self.config.loader.persistent_workers
        ))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def forward(self, x, sigma=None, x_emb=None, attention_mask=None, step=None):
    """Returns logits.

      x_emb can be provided during PPLM / NoS-style guidance
      (see: https://arxiv.org/abs/2305.20009).
    """
    if self.is_eval_classifier:
      logits = self.classifier_model(x)
      if hasattr(logits, 'logits'):
        logits = logits.logits
    else:
      sigma = self._process_sigma(sigma) if sigma is not None else sigma
      with torch.cuda.amp.autocast(dtype=torch.float32):
        logits = self.classifier_model(x, sigma, x_emb=x_emb, attention_mask=attention_mask, step=step)
    return logits

  def get_log_probs(self, x, sigma, x_emb=None):
    """Returns log probabilities.
      Use for CBG-style guidance.
    """
    if self.is_eval_classifier:
      raise NotImplementedError(
        '`get_log_prob` not implemented for classifiers '
        'that are meant to be used for evaluation purposes '
        'only.')
    with torch.cuda.amp.autocast(dtype=torch.float32):
      return torch.nn.functional.log_softmax(self.forward(x, sigma, x_emb=x_emb), dim=-1)

  def get_log_probs_antibiotic_guaidance(self, x, sigma, x_emb=None, step=None):
    """Returns log probabilities.
      Use for CBG-style guidance.
    """
    if self.is_eval_classifier:
      raise NotImplementedError(
        '`get_log_prob` not implemented for classifiers '
        'that are meant to be used for evaluation purposes '
        'only.')
    with torch.cuda.amp.autocast(dtype=torch.float32):
      return self.forward(x, sigma, x_emb=x_emb, step=step)

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    self.log(name='lr',
             value=
             self.trainer.optimizers[0].param_groups[0][
               'lr'],
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True, logger=False)
    return loss

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.classifier_model.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  def _q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      move_chance: float torch.Tensor with shape
        (batch_size, 1).
    """
    move_indices = torch.rand(
      *x.shape, device=x.device) < move_chance
    if self.config.diffusion == 'absorbing_state':
      return torch.where(move_indices, self.mask_index, x)
    if self.config.diffusion == 'uniform':
      uniform_tensor = torch.randint(
        0, self.vocab_size, x.shape, device=x.device)
      return torch.where(move_indices, uniform_tensor, x)
    raise NotImplementedError(
        f'Diffusion type {self.config.diffusion} not '
        'implemented.')

  def _compute_loss(self, batch, prefix):
    x0 = batch['input_ids']
    attention_mask = batch['attention_mask']
    t = None
    if self.is_eval_classifier:
      logits = self.forward(x0)
    elif self.config.parameterization == 'ar':
      # do not add noise for AR FUDGE and AR PPLM
      logits = self.forward(x0, attention_mask=attention_mask)
    else:
      t = self._sample_t(x0.shape[0])
      if self.T > 0:
        t = (t * self.T).to(torch.int)
        t = t / self.T
        # t \in {1/T, 2/T, ..., 1}
        t += (1 / self.T)
      if self.change_of_variables:
        time_conditioning = t[:, None]
        f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
        f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
        move_chance = torch.exp(f_0 + t * (f_T - f_0))
        move_chance = move_chance[:, None]
      else:
        sigma, _ = self.noise(t)
        time_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])

      xt = self._q_xt(x0, move_chance)
      logits = self.forward(xt, time_conditioning, attention_mask=attention_mask)
    if hasattr(self.config.data, 'label_col'):
      if f"{self.config.data.label_col}_threshold" in batch:
        y = batch[f"{self.config.data.label_col}_threshold"]
      else:
        y = batch[self.config.data.label_col]
    else:
      y = batch['label']
    if (not self.is_eval_classifier
        and getattr(self.config.training, 'use_label_smoothing', False)):
      # Interpolate between one-hot and uniform distribution
      labels = (torch.nn.functional.one_hot(y, self.config.data.num_classes) * (1 - t)[..., None] +
           (1 / self.config.data.num_classes) * t[..., None])
    else:
      labels = y.view(-1)
    if getattr(self.config, 'is_fudge_classifier', False):
      expanded_y = y.unsqueeze(1).expand(-1, logits.shape[1]) # batch x seq
      logits = logits.view(-1, self.config.data.num_classes)[attention_mask.flatten()==1, ...]
      y = expanded_y.flatten().long()[attention_mask.flatten()==1]
      loss = torch.nn.functional.cross_entropy(
        logits,
        y,
        ignore_index=-100,
        reduction='mean')
    else:
      loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels,
        ignore_index=-100,
        reduction='mean')

    if prefix == 'train':
      self.train_metrics.update(logits, y)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(logits, y)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(logits, y)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def _sample_t(self, n):
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _process_sigma(self, sigma):
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

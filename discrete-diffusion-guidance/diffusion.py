"""Module for modeling discrete diffusion
  (absorbing state or uniform) and AR
  (a special case of absorbing state).
"""
import itertools
import math
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from mamba_ssm.utils.generation import InferenceParams
from torch import Tensor
from tqdm.auto import tqdm

import classifier
import dataloader
import models
import noise_schedule
import selfies as sf

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log()).to(categorical_probs.dtype)
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor
  recon_loss: typing.Optional[torch.FloatTensor] = None
  diffusion_loss: typing.Optional[torch.FloatTensor] = None


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = len(tokenizer.get_vocab())

    self.antithetic_sampling = config.training.antithetic_sampling
    self.importance_sampling = config.training.importance_sampling
    self.change_of_variables = config.training.change_of_variables
    self.noise = noise_schedule.get_noise(config, dtype=self.dtype)

    if self.config.is_vision:
        self.mask_index = getattr(tokenizer, 'mask_token_id', -1)
    else:
      if (not hasattr(self.tokenizer, 'mask_token')
          or tokenizer.mask_token is None):
        self.mask_index = self.vocab_size
        self.vocab_size += 1
      else:
        self.mask_index = tokenizer.mask_token_id

    # Note: creating limiting distribution with
    #  broadcast-able batch and sequence dimensions.
    self.parameterization = config.parameterization
    self.diffusion = config.diffusion
    if config.parameterization == 'ar':
      self.limiting_distribution = None
    else:
      if self.diffusion == 'absorbing_state':
        # Not needed, posterior calculated explicitly.
        limiting_distribution = None
      elif self.diffusion == 'uniform':
        limiting_distribution = torch.ones(
          (1, 1, self.vocab_size), dtype=self.dtype) / self.vocab_size
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")
      self.register_buffer('limiting_distribution',
                           limiting_distribution)

    self.T = config.T
    self.subs_masking = config.subs_masking
    self.time_conditioning = config.time_conditioning

    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config, vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'unet':
      self.backbone = models.unet.UNet(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.model.pretrained_model_name_or_path, trust_remote_code=True)
    else:
      raise NotImplementedError(
        f"Backbone {self.config.backbone} not implemented.")

    self.lr = self.config.optim.lr
    self.sampling_eps = config.training.sampling_eps

    self.softplus = torch.nn.Softplus()
    self.neg_infinity = -1_000_000.0

    if config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=config.training.ema)
    else:
      self.ema = None

    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    self.fast_forward_epochs = None
    self.fast_forward_batches = None

    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.diffusion != 'absorbing_state':
      assert self.parameterization not in {'ar', 'subs'}
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.limiting_distribution is not None:
      checkpoint['state_dict']['limiting_distribution'] = self.limiting_distribution.to(
        list(checkpoint['state_dict'].values())[0].device)
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    # Do not save this buffer
    checkpoint['state_dict'].pop('limiting_distribution',
                                 None)
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
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
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
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

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
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

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # "Zero Masking Prob":
    # log prob at the mask index = - infinity
    logits[..., self.mask_index] += self.neg_infinity

    # "Copy over":
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0

    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    return logits.log_softmax(dim=-1)

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, cond=None, x_emb=None, **kwargs):
    """Returns log_probs / logits."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma, cond, x_emb=x_emb, **kwargs)

    if self.parameterization == 'subs':
      # returns log_probs
      return self._subs_parameterization(
        logits=logits, xt=x)
    if self.parameterization in {'ar', 'd3pm'}:
      # returns log_probs
      if self.subs_masking:  # Can use "zero masking prob"
        logits[:, :, self.mask_index] += self.neg_infinity
      return logits.log_softmax(dim=-1)
    return logits

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size)
    if self.diffusion == 'uniform':
      return (
        (alpha_t * self.vocab_size * x * xt_one_hot +
         (alpha_ts - alpha_t) * xt_one_hot +
         d_alpha * x +
         (1 - alpha_ts) * (1 - alpha_s) * self.limiting_distribution)
        /
        (alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None]) +
         (1 - alpha_t))
      )
    raise NotImplementedError(
      f"Diffusion type {self.diffusion} not implemented.")

  def _d3pm_loss(self, model_output, xt, x0, t):
    assert self.config.noise.type == 'loglinear', (
      'D3PM loss only implemented for log-linear noise.')
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    if self.diffusion == 'absorbing_state':
      log_x_theta_at_x0 = torch.gather(
        model_output, -1, x0[:, :, None]).squeeze(-1)
      log_x_theta_at_m = model_output[:, :, self.mask_index]
      x_theta_at_m = log_x_theta_at_m.exp()

      term_1_coef = dt / t
      term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
      term_1_log_dr = log_x_theta_at_x0

      term_2_coef = 1 - dt / t
      term_2_log_nr = term_1_log_nr
      term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

      L_vb_masked = (
        term_1_coef * (term_1_log_nr - term_1_log_dr)
        + term_2_coef * (term_2_log_nr - term_2_log_dr))

      L_vb = L_vb_masked * (xt == self.mask_index)
    elif self.diffusion == 'uniform':
      posterior = self._compute_posterior(
        x=F.one_hot(x0, num_classes=self.vocab_size).to(self.dtype),
        xt=xt,
        alpha_s=alpha_s[..., None],
        alpha_t=alpha_t[..., None])
      posterior_pred = self._compute_posterior(
        x=model_output.exp(),
        xt=xt,
        alpha_s=alpha_s[..., None],
        alpha_t=alpha_t[..., None])
      L_vb = (
          posterior * (torch.log(posterior + 1e-12) - torch.log(posterior_pred))
      ).sum(dim=-1)
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented for D3PM.")
    return self.T * L_vb

  def _reconstruction_loss(self, x0, cond=None):
    # For D3PM parameterization
    assert self.config.noise.type == 'loglinear', (
      'Reconstruction loss only implemented for log-linear '
      'noise.')
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    time_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, time_conditioning,
                                   cond=cond)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _sample_t(self, n):
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(
        t)
    return t

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
    if self.diffusion == 'absorbing_state':
      return torch.where(move_indices, self.mask_index, x)
    if self.diffusion == 'uniform':
      uniform_tensor = torch.randint(
        0, self.vocab_size, x.shape, device=x.device)
      return torch.where(move_indices, uniform_tensor, x)
    elif self.diffusion == 'uniform_data_marginals':
      return torch.where(
        move_indices,
        self._sample_prior(*x.shape),
        x)
    raise NotImplementedError(
      f"Diffusion type {self.diffusion} not implemented.")

  def _forward_pass_diffusion(self, x0, cond=None):
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
      sigma, dsigma = None, None
    else:
      sigma, dsigma = self.noise(t)
      time_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self._q_xt(x0, move_chance)
    model_output = self.forward(xt, time_conditioning,
                                cond=cond)

    # Discrete (finite T) time
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(
          x0, cond=cond)
        if self.training and self.config.training.use_simple_ce_loss:
          loss = -torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        else:
          loss = reconstruction_loss + diffusion_loss
        return {
          'recon_loss': reconstruction_loss,
          'diffusion_loss': diffusion_loss,
          'loss': loss}
      elif self.parameterization == 'subs':
        if self.training and self.config.training.use_simple_ce_loss:
          loss = -torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        else:
          loss = diffusion_loss
        return {'diffusion_loss': diffusion_loss, 'loss': loss}
      else:
        raise ValueError(
          f"Invalid parameterization: {self.parameterization} for T > 0.")

    # Continuous (T --> infty) time
    if self.diffusion == 'absorbing_state':
      # SUBS parameterization, continuous time.
      log_p_theta = torch.gather(
        input=model_output,
        dim=-1,
        index=x0[:, :, None]).squeeze(-1)

      if self.change_of_variables or self.importance_sampling:
        if self.training and self.config.training.use_simple_ce_loss:
          return {
          'diffusion_loss': log_p_theta * torch.log1p(-torch.exp(- self.noise.sigma_min)),
          'loss': -log_p_theta
          }
        return log_p_theta * torch.log1p(-torch.exp(- self.noise.sigma_min))

      if self.training and self.config.training.use_simple_ce_loss:
        return {
          'diffusion_loss': log_p_theta * (dsigma / torch.expm1(sigma))[:, None],
          'loss': log_p_theta
        }
      return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    elif self.diffusion == 'uniform':
      assert self.config.noise.type == 'loglinear', (
        'Continuous time uniform diffusion only implemented'
        ' for log-linear noise.')
      # TODO: Currently α_t' and α_t are hardcoded to a
      #  log-linear noise.
      #  Make generic (as above, for absorbing state):
      #    alpha_t_prime =  -dsigma * (-sigma).exp()
      #    alpha_t = (-sigma).exp()
      alpha_t_prime = -1.
      alpha_t = 1. - t[..., None, None]  # B, 1, 1

      # x_bar = N * α_t * x + 1 - α_t ; B, L, V
      x_bar = self.vocab_size * alpha_t * F.one_hot(x0, self.vocab_size).float() + 1 - alpha_t
      x_bar_theta = self.vocab_size * alpha_t * model_output.exp() + 1 - alpha_t

      # α_t' / (N*α_t)
      coeff = alpha_t_prime / (self.vocab_size * alpha_t)  # B, 1, 1

      # Term 1: indices where z_t = 1
      x_bar_zt = torch.gather(x_bar, -1, xt[..., None])  # B, L, 1
      x_bar_theta_zt = torch.gather(x_bar_theta, -1, xt[..., None])  # B, L, 1
      term1 = ((self.vocab_size / x_bar_zt) - (self.vocab_size / x_bar_theta_zt))  # B, L, 1

      # Term 2: indices where z_t = 0
      term2 = (  # B, L, V before summing --> B, L, 1 after
          (x_bar / x_bar_zt) *
          (
              x_bar_theta_zt.log() - x_bar_theta.log() +
              x_bar.log() - x_bar_zt.log()
          )
      )
      term2 = term2.sum(dim=-1, keepdim=True)  # B, L, 1

      diffusion_loss = (coeff * (term1 - term2)).squeeze()  # B, L
      reconstruction_loss = self._reconstruction_loss(
        x0, cond=cond)
      if self.training and self.config.training.use_simple_ce_loss:
        return {
          'recon_loss': reconstruction_loss,
          'diffusion_loss': diffusion_loss,
          'loss': -torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        }
      return {
        'recon_loss': reconstruction_loss,
        'diffusion_loss': diffusion_loss,
        'loss': diffusion_loss if getattr(self.config, 'zero_recon_loss', False)
                else diffusion_loss + reconstruction_loss
      }
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not "
        "implemented for continuous time case.")

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is necessary for the text8-crop dataset;
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _loss(self, x0, attention_mask, cond=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
      x0, attention_mask)

    recon_loss, diffusion_loss = None, None

    if (cond is not None and self.training
        and self.config.training.guidance is not None
        and self.config.training.guidance.cond_dropout > 0):
      # Randomly mask out conditioning for classifier-free
      # guidance training.
      p = torch.bernoulli(
        torch.ones_like(cond) *
        self.config.training.guidance.cond_dropout).to(torch.bool)
      # Use num_classes index as conditioning mask_token_id
      cond[p] = self.config.data.num_classes

    if self.parameterization == 'ar':
      logprobs = self.forward(
        input_tokens, sigma=None, cond=cond)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens,
                                          cond=cond)
      if isinstance(loss, dict):
        recon_loss = loss['recon_loss']
        diffusion_loss = loss['diffusion_loss']
        loss = loss['loss']

    nlls = loss * attention_mask
    count = attention_mask.sum()

    if (self.config.training.compute_loss_on_pad_tokens
        and self.training):
      token_nll = loss.mean()
    else:
      batch_nll = nlls.sum()
      token_nll = batch_nll / count

    if recon_loss is not None and diffusion_loss is not None:
      with torch.no_grad():
        recon_loss_batch = (recon_loss * attention_mask).sum() / count
        diffusion_loss_batch = (diffusion_loss * attention_mask).sum() / count
      return Loss(loss=token_nll,
                  nlls=nlls,
                  token_mask=attention_mask,
                  recon_loss=recon_loss_batch,
                  diffusion_loss=diffusion_loss_batch)
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    cond = None
    if (self.config.training.guidance is not None or  # Training for / using CFG
        (hasattr(self.config, 'guidance')
         and self.config.guidance is not None
         and self.config.guidance.method == 'cfg')):
      if self.config.data.label_col in batch:
        cond = batch[self.config.data.label_col]
      elif f"{self.config.data.label_col}_threshold" in batch:
        cond = batch[f"{self.config.data.label_col}_threshold"]
      else:
        raise RuntimeError(
          f"Conditioning {self.config.data.label_col}"
          f" not found in batch.")
    losses = self._loss(batch['input_ids'], attention_mask,
                        cond=cond)

    if prefix == 'train':
      self.train_metrics.update(losses.nlls,
                                losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls,
                                losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls,
                               losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f"Invalid prefix: {prefix}")

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return losses

  def training_step(self, batch, batch_idx):
    losses = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True)
    if losses.recon_loss is not None:
      self.log(name='trainer/recon_loss',
               value=losses.recon_loss.item(),
               on_step=True,
               on_epoch=False,
               sync_dist=True,
               prog_bar=False)
      self.log(name='trainer/diffusion_loss',
               value=losses.diffusion_loss.item(),
               on_step=True,
               on_epoch=False,
               sync_dist=True,
               prog_bar=False)
    self.log(name='lr',
             value=self.trainer.optimizers[0].param_groups[0]['lr'],
             on_step=True,
             on_epoch=False,
             sync_dist=True,
             prog_bar=True, logger=False)
    return losses.loss

  def validation_step(self, batch, batch_idx):
    losses = self._compute_loss(batch, prefix='val')
    return losses.loss

  def load_ema_params(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _restore_non_ema_params(self):
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def on_validation_epoch_start(self):
    self.load_ema_params()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def on_validation_epoch_end(self):
    self._restore_non_ema_params()
    if (not self.trainer.sanity_checking
        and self.config.eval.generate_samples
        and self.trainer.global_rank == 0):
      self.config.sampling.batch_size = 1
      if self.config.is_vision:
        samples = []
        if self.config.training.guidance is not None:
          # Generate one image per class (up to 10 images)

          guidance = {
            'method': 'cfg', 'condition': 0, 'gamma': 1.0}
          omegaconf.OmegaConf.update(
            self.config, key='guidance', value=guidance,
            force_add=True)
          for i in range(max(self.config.data.num_classes, 10)):
            self.config.guidance.condition = i
            samples.append(self.sample())
        else:
          # Generate ten images
          for i in range(10):
            samples.append(self.sample())
        image_samples = self.tokenizer.batch_decode(
          torch.concat(samples, dim=0))
        if hasattr(self.trainer.logger, 'log_image'):
          self.trainer.logger.log_image(
            key=f"samples@global_step{self.global_step}",
            caption=[str(i) for i in range(len(samples))],
            images=[s for s in image_samples.float()])
      else:
        if self.config.training.guidance is not None:
          guidance = {
            'method': 'cfg', 'condition': 0, 'gamma': 1.0}
          omegaconf.OmegaConf.update(
            self.config, key='guidance', value=guidance,
            force_add=True)
          for i in range(self.config.data.num_classes):
            self.config.guidance.condition = i
            samples = self.sample()
            decoded_samples = self.tokenizer.batch_decode(
              samples)
            if hasattr(self.trainer.logger, 'log_table'):
              # Log some generated samples
              self.trainer.logger.log_table(
                key=f"samples@global_step{self.global_step}_class-{i}",
                columns=['Generated Samples'],
                data=[decoded_samples])
        else:
          self.config.sampling.batch_size = 2
          samples = self.sample()
          decoded_samples = self.tokenizer.batch_decode(
            samples)
          if hasattr(self.trainer.logger, 'log_table'):
            # Log some generated samples
            self.trainer.logger.log_table(
              key=f"samples@global_step{self.global_step}",
              columns=['Generated Samples'],
              data=[[s] for s in decoded_samples])

  def _sample_prior(self, *batch_dims):
    if self.diffusion == 'absorbing_state':
      matrix = torch.ones(*batch_dims, dtype=torch.int64, device=self.device)
      matrix[:, self.config.sampling.target_length:] = 0
      # matrix[:, self.config.sampling.target_length] = 0

      matrix = matrix * self.mask_index + (1 - matrix) * self.tokenizer.pad_token_id

      # starting_mols = ['N[C@@H1](CS)C(=O)']
      # starting_tokens = self.tokenizer(sf.encoder(starting_mols[0]).replace('][', '] ['), return_tensors='pt')['input_ids'][0, :-1].to(self.device)
      # matrix = matrix * self.mask_index + (1 - matrix) * self.tokenizer.pad_token_id
      # matrix[:, 0:starting_tokens.shape[0]] = matrix[:, :starting_tokens.shape[0]] + starting_tokens.unsqueeze(0) - self.mask_index

      # for i in range(matrix.shape[0]):
      #   with open(f'/data2/tianang/projects/discrete-diffusion-guidance/outputs/visualize_gen_process/try_{i+1}.txt', 'w') as f:
      #     text = self.tokenizer.decode(matrix[i])
      #     f.write(text + '\n')

      # return matrix * self.mask_index + (1 - matrix) * self.tokenizer.pad_token_id
      return matrix
      # return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64, device=self.device)
    if self.diffusion == 'uniform':
      return torch.randint(
        0, self.vocab_size, batch_dims, dtype=torch.int64,
        device=self.device)
    elif self.diffusion == 'uniform_data_marginals':
      if self.limiting_distribution.squeeze().ndim == 2:
        batch_dims = (batch_dims[0],)
      return torch.distributions.Categorical(
        self.limiting_distribution.squeeze()).sample(
        sample_shape=torch.Size(batch_dims))
    raise NotImplementedError(
      f'Diffusion type {self.diffusion} not '
      'implemented.')

  def sample(
    self,
    eps=1e-5):  # Note: differs from self.config.training.sampling_eps
    """Generate samples from (ema) model.

      Supports both AR and diffusion sampling.
      Supports:
        - standard decoding,
        - classifier-free guidance,
        - classifier-based guidance
          - CBG / FUDGE,
          - NOS / PPLM.
    """
    # WARNING: Lightning auto-casting is not working in this method.
    if not self.config.eval.disable_ema:
      self.load_ema_params()
    if getattr(self.config, 'guidance', None) is not None:
      if self.config.guidance.method == 'cfg':
        cond = (torch.ones(self.config.sampling.batch_size, device=self.device) *
                self.config.guidance.condition).to(torch.long)
      else:
        cond = None
      if ((self.parameterization == 'ar' and self.config.guidance.method in {'fudge', 'pplm'})
          or self.config.guidance.method in {'cbg', 'nos'}):
        classifier_model = classifier.Classifier.load_from_checkpoint(  # classifier_model 在这呢 TODO: 这个也直接改掉
          self.config.guidance.classifier_checkpoint_path,
          tokenizer=self.tokenizer,
          config=self.config, logger=False).to(self.device)
        classifier_model.eval()
      else:
        classifier_model = None
    else:
      classifier_model, cond = None, None

    if self.parameterization == 'ar':
      samples = self._ar_sample(
        classifier_model=classifier_model, cond=cond)
    else:  # Diffusion sampling
      samples = self._diffusion_sample(
        classifier_model=classifier_model, cond=cond,
        eps=eps)
    if not self.config.eval.disable_ema:
      self._restore_non_ema_params()
    return samples

  def sample_AMP(self, eps=1e-5, config=None, tokenizer=None):  # Note: differs from self.config.training.sampling_eps
    """Generate samples from (ema) model.

      Supports both AR and diffusion sampling.
      Supports:
        - standard decoding,
        - classifier-free guidance,
        - classifier-based guidance
          - CBG / FUDGE,
          - NOS / PPLM.
    """
    # WARNING: Lightning auto-casting is not working in this method.
    if not self.config.eval.disable_ema:
      self.load_ema_params()
    if getattr(self.config, 'guidance', None) is not None:
      if self.config.guidance.method == 'cfg':
        cond = (torch.ones(self.config.sampling.batch_size, device=self.device) *
                self.config.guidance.condition).to(torch.long)
      else:
        cond = None
      if ((self.parameterization == 'ar' and self.config.guidance.method in {'fudge', 'pplm'}) or self.config.guidance.method in {'cbg', 'nos', 'cbg_antibiotic', 'nos_antibiotic', 'cbg_antibiotic_remdm_loop', 'cbg_antibiotic_remdm_loop_reg_cls'}):
        classifier_model = classifier.Classifier(
          config,
          tokenizer=tokenizer,
          pretrained_backbone=None)  # pretrained_backbone 是 None 的时候才有手动初始化的机会
        # classifier_model = classifier.Classifier.load_from_checkpoint(  # classifier_model 在这呢 TODO: 这个也直接改掉
        #   self.config.guidance.classifier_checkpoint_path,
        #   tokenizer=self.tokenizer,
        #   config=self.config, logger=False).to(self.device)
        classifier_model.eval()
        classifier_model.classifier_model.load_pretrained_weight()  # TODO: load train好的 regressor 的权重
        classifier_model = classifier_model.to('cuda')
      else:
        classifier_model = None
    else:
      classifier_model, cond = None, None

    if self.parameterization == 'ar':
      samples = self._ar_sample(
        classifier_model=classifier_model, cond=cond)
    else:  # Diffusion sampling
      samples = self._diffusion_sample(
        classifier_model=classifier_model, cond=cond,
        eps=eps)
    if not self.config.eval.disable_ema:
      self._restore_non_ema_params()
    return samples

  @torch.no_grad()
  def _ar_sample(
      self,
      classifier_model: typing.Optional[classifier.Classifier] = None,
      cond: typing.Optional[torch.tensor] = None,
  ):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (self.config.sampling.batch_size, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute Gumbel sampling noise
    if (getattr(self.config, 'guidance', None) is not None
        and self.config.guidance.method == 'fudge'):
      noise = torch.distributions.Gumbel(0, 1).sample(
        (self.config.sampling.batch_size,  # type: ignore
         num_pred_tokens,
         self.config.guidance.topk)).to(self.device)
    else:
      noise = torch.distributions.Gumbel(0, 1).sample(
        (self.config.sampling.batch_size,  # type: ignore
          num_pred_tokens,
          self.vocab_size)).to(self.device)
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    pbar = tqdm(range(num_pred_tokens), desc='AR Sampling', leave=False)
    inference_params = InferenceParams(
      max_seqlen=num_pred_tokens,
      max_batch_size=x.shape[0],
      seqlen_offset=1)
    # For cfg we do 2 forward passes, one for conditional
    # model and one unconditional, so we need 2 copies of
    # inference_params.
    uncond_inference_params = InferenceParams(
      max_seqlen=num_pred_tokens,
      max_batch_size=x.shape[0],
      seqlen_offset=1)
    for i in pbar:
      if getattr(self.config, 'guidance', None) is None:
        if self.config.backbone == 'dimamba':
          log_probs = self.forward(
            x[:, i:i + 1], None, cond=None,
            inference_params=inference_params)
        else:
          log_probs = self.forward(x[:, :i + 1],
                                  None, cond=None)
        if self.config.sampling.use_float64:
          log_probs = log_probs.to(torch.float64)
        next_log_probs = log_probs[:, -1]
        y = (next_log_probs + noise[:, i]).argmax(-1)
      else:
        if self.config.guidance.method == 'cfg':
          if self.config.backbone == 'dimamba':
            next_log_probs = self._ar_cfg_denoise(
              cond=cond,
              gamma=self.config.guidance.gamma,
              x=x[:, i:i + 1],
              i=i,
              inference_params=(inference_params, uncond_inference_params))
          else:
            next_log_probs = self._ar_cfg_denoise(
              cond=cond,
              gamma=self.config.guidance.gamma,
              x=x,
              i=i)
          y = (next_log_probs + noise[:, i]).argmax(-1)
        elif self.config.guidance.method == 'fudge':
          if self.config.backbone == 'dimamba':
            next_log_probs, top_indices = self._ar_fudge_denoise(
              classifier_model=classifier_model,
              guidance_cond=self.config.guidance.condition,
              topk=self.config.guidance.topk,
              gamma=self.config.guidance.gamma,
              x=x[:, i:i + 1],
              i=i,
              inference_params=inference_params)
          else:
            next_log_probs, top_indices = self._ar_fudge_denoise(
              classifier_model=classifier_model,
              guidance_cond=self.config.guidance.condition,
              topk=self.config.guidance.topk,
              gamma=self.config.guidance.gamma,
              x=x,
              i=i)
          y = torch.gather(
            top_indices,
            1,
            (next_log_probs + noise[:, i]).argmax(-1).unsqueeze(1)
          ).squeeze(1)
        elif self.config.guidance.method == 'pplm':
          raise NotImplementedError
        else:
          raise NotImplementedError(
            f"Guidance method {self.config.guidance.method} not implemented.")
      pbar.set_postfix(
        prob_check=(next_log_probs.exp().sum() / x.shape[0]).item(),
        nan_check=bool(next_log_probs.isnan().sum() > 0))
      x[:, i + 1] = y
    return x

  def _ar_cfg_denoise(
      self,
      cond: torch.tensor,
      gamma: float,
      x: torch.tensor,
      i: int,
      **kwargs
  ) -> torch.tensor:
    if self.config.guidance.gamma == 0.0:  # Sample unconditionally
      mask_cond = (torch.ones_like(cond) *
                   self.config.data.num_classes)
      if self.config.backbone == 'dimamba':
        inference_params = kwargs.pop('inference_params')
        log_probs = self.forward(
          x[:, :i + 1],None, cond=mask_cond,
          inference_params=inference_params[1])
      else:
        log_probs = self.forward(
          x[:, :i + 1],None, cond=mask_cond, **kwargs)
    elif gamma == 1.0:  # Sample conditionally
      if self.config.backbone == 'dimamba':
        inference_params = kwargs.pop('inference_params')
        log_probs = self.forward(
          x[:, :i + 1], None, cond=cond,
          inference_params=inference_params[0])
      else:
        log_probs = self.forward(
          x[:, :i + 1], None, cond=cond, **kwargs)
    else:  # Sample from tempered distribution
        mask_cond = (torch.ones_like(cond) *
                     self.config.data.num_classes)
        if self.config.backbone == 'dimamba':
          inference_params = kwargs.pop('inference_params')
          log_probs_cond = self.forward(
            x[:, :i + 1], None, cond=cond,
            inference_params=inference_params[0])
          log_probs_uncond = self.forward(
            x[:, :i + 1],None, cond=mask_cond,
            inference_params=inference_params[1])
        else:
          log_probs_cond = self.forward(
            x[:, :i + 1], None, cond=cond, **kwargs)
          log_probs_uncond = self.forward(
            x[:, :i + 1],None, cond=mask_cond, **kwargs)

        log_probs = gamma * log_probs_cond + (1 - gamma) * log_probs_uncond
        # Gamma > 1.0 causes instability for Mamba, re-normalizing
        log_probs = log_probs.log_softmax(dim=-1)
    return log_probs[:, -1]

  def _ar_fudge_denoise(
      self,
      classifier_model: classifier.Classifier,
      guidance_cond: int,
      topk: int,
      gamma: float,
      x: torch.tensor,
      i: int,
      **kwargs
  ) -> typing.Tuple[torch.tensor, torch.LongTensor]:
    log_probs = self.forward(
      x[:, :i + 1], None, cond=None, **kwargs)
    next_log_probs = log_probs[:, -1]
    top_logits, top_indices = next_log_probs.topk(topk, dim=-1)
    t_candidates = torch.cat(
      [x[:, :i + 1].unsqueeze(1).expand(-1, topk, -1),
        top_indices.unsqueeze(2)],
      dim=2).view(-1, i + 2)  # (B * K), L

    t = torch.zeros(t_candidates.shape[0],
                    device=self.device)
    sigma, dsigma = self.noise(t)
    time_conditioning = sigma[:, None]

    classifier_log_prob = classifier_model.get_log_probs(
      t_candidates, time_conditioning)
    classifier_log_prob = classifier_log_prob[:, i + 1, :].view(
      x.shape[0], topk, -1)[..., guidance_cond]  # (batch, topk)
    next_log_probs = (top_logits + gamma * classifier_log_prob).log_softmax(dim=-1)
    return next_log_probs, top_indices

  def _ar_pplm_denoise(
      self,
      classifier_model: classifier.Classifier,
      guidance_cond: int,
      num_ppl_steps: int,
      pplm_step_size: float,
      pplm_stability_coef: float,
      x: torch.tensor,
      i: int,
  ):
    raise NotImplementedError

  @torch.no_grad()
  def _diffusion_sample(
    self,
    classifier_model: typing.Optional[classifier.Classifier] = None,
    cond: typing.Optional[torch.tensor] = None,
    eps: float = 1e-5,  # Note: differs from self.config.training.sampling_eps
  ):

    xt = self._sample_prior(  # 这个 prior 就是全是 mask
      self.config.sampling.batch_size,
      self.config.model.length
    ).to(self.device)

    timesteps = torch.linspace(1, eps, self.config.sampling.steps + 1, device=self.device)
    dt = (1 - eps) / self.config.sampling.steps

    pbar = tqdm(range(self.config.sampling.steps), desc='Sampling',leave=False)
    NFEs = 0
    cache = None

    for i in pbar:
      t = timesteps[i]
      if self.T > 0:  # t in {1/T,..., 1}, to match training
        t = (t * self.T).to(torch.int)
        t = t / self.T
        t += (1 / self.T)
      t = t * torch.ones(xt.shape[0], 1, device=self.device)
      if cache is None:
        NFEs += 1
      sigma_t, _ = self.noise(t)
      sigma_s, _ = self.noise(t - dt)
      if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
      if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
      assert sigma_t.ndim == 1, sigma_t.shape
      assert sigma_s.ndim == 1, sigma_s.shape
      move_chance_t = 1 - torch.exp(-sigma_t)
      move_chance_s = 1 - torch.exp(-sigma_s)
      move_chance_t = move_chance_t[:, None, None]
      move_chance_s = move_chance_s[:, None, None]
      assert move_chance_t.ndim == 3, move_chance_t.shape

      if getattr(self.config, 'guidance', None) is None:
        xs, q_xs, cache = self._ddpm_denoise(
          xt=xt,
          time_conditioning=sigma_t,
          move_chance_t=move_chance_t,
          move_chance_s=move_chance_s,
          cache=cache)
      else:
        if self.config.guidance.method == 'cfg':
          xs, q_xs, cache = self._cfg_denoise(
            cond=cond,
            gamma=self.config.guidance.gamma,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache)
        elif self.config.guidance.method == 'cbg':  # 这个
          xs, q_xs, cache = self._cbg_denoise(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            gamma=self.config.guidance.gamma,
            use_approx=self.config.guidance.use_approx,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache)
        elif self.config.guidance.method == 'cbg_antibiotic':  # 这个
          xs, q_xs, cache = self._cbg_denoise_antibiotic(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            gamma=self.config.guidance.gamma,
            use_approx=self.config.guidance.use_approx,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache,
            step = i+1)
        elif self.config.guidance.method == 'cbg_antibiotic_remdm_loop':  # 这个
          xs, q_xs, cache = self._cbg_denoise_antibiotic_remdm_loop(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            gamma=self.config.guidance.gamma,
            use_approx=self.config.guidance.use_approx,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache,
            step = i+1,
            t=t,
            dt=dt)
        elif self.config.guidance.method == 'nos':
          xs, q_xs, cache = self._nos_denoise(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            num_nos_steps=self.config.guidance.num_nos_steps,
            nos_step_size=self.config.guidance.nos_step_size,
            nos_stability_coef=self.config.guidance.nos_stability_coef,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s)
        elif self.config.guidance.method == 'nos_antibiotic':
          xs, q_xs, cache = self._nos_denoise_AMP(
            classifier_model=classifier_model,
            conditioning_class=self.config.guidance.condition,
            num_nos_steps=self.config.guidance.num_nos_steps,
            nos_step_size=self.config.guidance.nos_step_size,
            nos_stability_coef=self.config.guidance.nos_stability_coef,
            xt=xt,
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            step = i+1)
        else:
          raise NotImplementedError(
            f"Guidance method {self.config.guidance.method} not implemented.")
      pbar.set_postfix(
        NFEs=NFEs,
        prob_check=(q_xs.sum() / xt.numel()).item(),
        nan_check=bool(q_xs.isnan().sum() > 0))
      if (not self.config.sampling.use_cache or
          not torch.allclose(xs, xt)):
        # Disable caching
        cache = None
      xt = xs

    # 再 decode 一次
    min_t = timesteps[-1].item()
    t = min_t * torch.ones(xt.shape[0], 1, device=self.device)
    unet_conditioning = self.noise(t)[0]
    xt = self.forward(xt, unet_conditioning).argmax(dim=-1)


    return xt

  def _ddpm_denoise(
    self,
    xt: torch.tensor,
    time_conditioning: torch.tensor,
    move_chance_t: torch.tensor,
    move_chance_s: torch.tensor,
    cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    # Compute x_theta
    if cache is not None:
      log_x_theta = cache['log_x_theta']
    else:
      log_x_theta = self.forward(xt, time_conditioning,
                                 cond=None)
      if self.config.sampling.use_float64:
        log_x_theta = log_x_theta.to(torch.float64)
    x_theta = log_x_theta.exp()

    # Compute posterior
    if self.diffusion == 'absorbing_state':
      q_xs = x_theta * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      q_xs /= move_chance_t
    elif self.diffusion == 'uniform':
      q_xs = self._compute_posterior(
        x=x_theta,
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t)
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    # Sample from posterior
    xs = _sample_categorical(q_xs)
    if self.diffusion == 'absorbing_state':
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
      xs = torch.where(copy_flag, xt, xs)

    return xs, q_xs, {'log_x_theta': log_x_theta}

  def _cfg_denoise(
      self,
      cond: torch.tensor,
      gamma: float,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    # Compute log_x_theta
    if cache is not None:
      log_x_theta_uncond = cache['log_x_theta_uncond']
      log_x_theta_cond = cache['log_x_theta_cond']
    else:
      if gamma == 0.0:  # Sample unconditionally
        mask_cond = (torch.ones_like(cond) *
                     self.config.data.num_classes)
        log_x_theta_uncond = self.forward(
          xt, time_conditioning, cond=mask_cond)
        log_x_theta_cond = None
      elif gamma == 1.0:  # Sample conditionally
        log_x_theta_cond = self.forward(xt, time_conditioning,
                                     cond=cond)
        log_x_theta_uncond = None
      else:  # Sample from tempered distribution
        log_x_theta_cond = self.forward(xt, time_conditioning,
                                     cond=cond)
        mask_cond = (torch.ones_like(cond) *
                     self.config.data.num_classes)
        log_x_theta_uncond = self.forward(xt,
                                       time_conditioning,
                                       cond=mask_cond)
    # Compute (weighted) posterior
    if (log_x_theta_cond is None  # gamma == 0
        or log_x_theta_uncond is None):  # or gamma == 1
      log_x_theta = log_x_theta_uncond if log_x_theta_uncond is not None else log_x_theta_cond
      x_theta = log_x_theta.exp()
      if self.diffusion == 'absorbing_state':
        q_xs = x_theta * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        q_xs /= move_chance_t
      elif self.diffusion == 'uniform':
        q_xs = self._compute_posterior(
          x=x_theta,
          xt=xt,
          alpha_s=1 - move_chance_s,
          alpha_t=1 - move_chance_t)
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")
    else:  # gamma != 0 and gamma != 1
      if self.diffusion == 'absorbing_state':
        log_x_theta = (gamma * log_x_theta_cond + (1 - gamma) * log_x_theta_uncond)
        x_theta = log_x_theta.softmax(dim=-1)
        q_xs = x_theta * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        q_xs /= move_chance_t
      elif (self.diffusion == 'uniform'
            or self.diffusion == 'uniform_data_marginals'):
        log_q_xs_uncond = self._compute_posterior(
          x=log_x_theta_uncond.exp(),
          xt=xt,
          alpha_s=1 - move_chance_s,
          alpha_t=1 - move_chance_t).log()
        log_q_xs_cond = self._compute_posterior(
          x=log_x_theta_cond.exp(),
          xt=xt,
          alpha_s=1 - move_chance_s,
          alpha_t=1 - move_chance_t).log()
        log_q_xs = (gamma * log_q_xs_cond +
                    (1 - gamma) * log_q_xs_uncond)
        q_xs = log_q_xs.softmax(dim=-1)
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")

    # Sample from posterior
    xs = _sample_categorical(q_xs)
    if self.diffusion == 'absorbing_state':
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
      xs = torch.where(copy_flag, xt, xs)

    return xs, q_xs, {'log_x_theta_uncond': log_x_theta_uncond,
                      'log_x_theta_cond': log_x_theta_cond}

  def _cbg_denoise(
      self,
      conditioning_class: int,
      gamma: float,
      classifier_model: classifier.Classifier,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      use_approx: bool = False,  # whether to use first-order approximation
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    if cache is not None:
      log_x_theta = cache['log_x_theta']
      classifier_log_prob = cache['classifier_log_prob']
    else:
      # Diffusion model
      log_x_theta = self.forward(xt, time_conditioning, cond=None)
      # Classifier model
      if use_approx:
        xt_one_hot = torch.nn.functional.one_hot(xt, self.vocab_size).to(torch.float)
        with torch.enable_grad():
          xt_one_hot.requires_grad_(True)
          classifier_log_prob_xt = classifier_model.get_log_probs(xt_one_hot, time_conditioning)
          classifier_log_prob_xt[..., conditioning_class].sum().backward()
          grad_log_prob_xt = xt_one_hot.grad

        classifier_log_prob_ratio = (
            grad_log_prob_xt - (xt_one_hot * grad_log_prob_xt).sum(dim=-1, keepdim=True)
        ).detach().requires_grad_(False)

        classifier_log_prob = (
            classifier_log_prob_ratio + classifier_log_prob_xt[..., conditioning_class][..., None, None]
        ).detach().requires_grad_(False)
      else:
        # Copied from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py#L441
        bsz, seq_len = xt.shape
        # Create bsz*seq_len*N copies of input sequences
        # Shape: (bsz, 1, seq_len) -> (bsz, seq_len*N, seq_len)
        # (where N = vocab_size).
        xt_expand = xt.unsqueeze(1).repeat(1, seq_len * self.vocab_size, 1)
        # Flatten batch and transition dimensions
        # Shape: (bsz, seq_len*N, seq_len) -> (bsz*seq_len*N, seq_len)
        xt_expand = xt_expand.view(-1, seq_len)

        # Create indices for all possible transitions
        # Shape: (seq_len*N,) -> (bsz, seq_len*N) -> (bsz*seq_len*N,)
        jump_idx = torch.arange(seq_len * self.vocab_size).to(xt.device)
        jump_idx = jump_idx.repeat(bsz, 1).flatten()

        # Create tensor for states after one transition
        xt_jumps = xt_expand.clone()

        # Calculate which dimension changes for each transition
        # Shape: (bsz*seq_len*N,)
        jump_dims = jump_idx // self.vocab_size

        # Calculate new value for changed dimension
        # Shape: (bsz*seq_len*N,)
        jump_states = jump_idx % self.vocab_size

        # Apply transitions by assigning new values at transition dimensions
        # Shape: (bsz*seq_len*N, seq_len)
        xt_jumps[
          torch.arange(jump_idx.size(0), device=xt.device),
          jump_dims,  # Index the transitioned dimension
        ] = jump_states  # Assign the new state

        classifier_log_prob = classifier_model.get_log_probs(
          xt_jumps, time_conditioning.repeat(seq_len * self.vocab_size)
        )[..., conditioning_class].reshape(bsz, seq_len, self.vocab_size)

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()
    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(
        x=log_x_theta.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
        copy_flag = (xt != self.mask_index)
        guided_log_probs[copy_flag] = self.neg_infinity
        guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")

    guided_probs = guided_log_probs.softmax(dim=-1)
    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag.to(bool), xt, xs)
    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'classifier_log_prob': classifier_log_prob}

  def _cbg_denoise_antibiotic(
      self,
      conditioning_class: int,
      gamma: float,
      classifier_model: classifier.Classifier,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      use_approx: bool = False,  # whether to use first-order approximation
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
      step: int = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    if cache is not None:
      log_x_theta = cache['log_x_theta']
      classifier_log_prob = cache['classifier_log_prob']
    else:
      # Diffusion model
      log_x_theta = self.forward(xt, time_conditioning, cond=None)
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_x_theta.exp()
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
        log_x_theta = torch.log(p_x0 + 1e-8)
      # Classifier model
      if use_approx:
        xt_one_hot = torch.nn.functional.one_hot(xt, self.vocab_size).to(torch.float)
        if self.config.guidance.var_length:
          backup_shape = xt_one_hot.shape
          xt_one_hot_backup = xt_one_hot
          xt_one_hot = xt_one_hot[:, :self.config.sampling.target_length, :]
        with torch.enable_grad():
          xt_one_hot.requires_grad_(True)
          classifier_log_prob_xt = classifier_model.get_log_probs_antibiotic_guaidance(xt_one_hot, time_conditioning, step=step)
          classifier_log_prob_xt.sum().backward()
          grad_log_prob_xt = xt_one_hot.grad
          if self.config.guidance.var_length:
            true_grad = torch.zeros(backup_shape, device=xt_one_hot.device)
            true_grad[:, :self.config.sampling.target_length, :] = grad_log_prob_xt
            grad_log_prob_xt = true_grad
            xt_one_hot = xt_one_hot_backup

        classifier_log_prob_ratio = (
            grad_log_prob_xt - (xt_one_hot * grad_log_prob_xt).sum(dim=-1, keepdim=True)
        ).detach().requires_grad_(False)

        classifier_log_prob = (
            classifier_log_prob_ratio + classifier_log_prob_xt[..., None]
        ).detach().requires_grad_(False)

        print(f' classifier_log_prob_xt min: {classifier_log_prob_xt.min()}')
        print(f' classifier_log_prob_xt max: {classifier_log_prob_xt.max()}')
        print(f' classifier_log_prob_xt mean: {classifier_log_prob_xt.mean()}')

        print(f' classifier_log_prob_ratio min: {classifier_log_prob_ratio.min()}')
        print(f' classifier_log_prob_ratio max: {classifier_log_prob_ratio.max()}')
        print(f' classifier_log_prob_ratio mean: {classifier_log_prob_ratio.mean()}')

        print(f' classifier_log_prob min: {classifier_log_prob.min()}')
        print(f' classifier_log_prob max: {classifier_log_prob.max()}')
        print(f' classifier_log_prob mean: {classifier_log_prob.mean()}')

      else:
        # Copied from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py#L441
        bsz, seq_len = xt.shape
        # Create bsz*seq_len*N copies of input sequences
        # Shape: (bsz, 1, seq_len) -> (bsz, seq_len*N, seq_len)
        # (where N = vocab_size).
        xt_expand = xt.unsqueeze(1).repeat(1, seq_len * self.vocab_size, 1)
        # Flatten batch and transition dimensions
        # Shape: (bsz, seq_len*N, seq_len) -> (bsz*seq_len*N, seq_len)
        xt_expand = xt_expand.view(-1, seq_len)

        # Create indices for all possible transitions
        # Shape: (seq_len*N,) -> (bsz, seq_len*N) -> (bsz*seq_len*N,)
        jump_idx = torch.arange(seq_len * self.vocab_size).to(xt.device)
        jump_idx = jump_idx.repeat(bsz, 1).flatten()

        # Create tensor for states after one transition
        xt_jumps = xt_expand.clone()

        # Calculate which dimension changes for each transition
        # Shape: (bsz*seq_len*N,)
        jump_dims = jump_idx // self.vocab_size

        # Calculate new value for changed dimension
        # Shape: (bsz*seq_len*N,)
        jump_states = jump_idx % self.vocab_size

        # Apply transitions by assigning new values at transition dimensions
        # Shape: (bsz*seq_len*N, seq_len)
        xt_jumps[
          torch.arange(jump_idx.size(0), device=xt.device),
          jump_dims,  # Index the transitioned dimension
        ] = jump_states  # Assign the new state

        classifier_log_prob = classifier_model.get_log_probs_antibiotic_guaidance(
          xt_jumps, time_conditioning.repeat(seq_len * self.vocab_size)
        ).reshape(bsz, seq_len, self.vocab_size)

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()

      print(f' diffusion_log_probs min: {diffusion_log_probs.min()}')
      print(f' diffusion_log_probs max: {diffusion_log_probs.max()}')
      print(f' diffusion_log_probs mean: {diffusion_log_probs.mean()}')

    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(
        x=log_x_theta.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
        copy_flag = (xt != self.mask_index)
        guided_log_probs[copy_flag] = self.neg_infinity
        guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")

    # 禁止生成 '.'
    vocab = self.tokenizer.get_vocab()
    token_id_forbid = [vocab[word] for word in vocab.keys() if '.' in word]
    token_id_forbid += [vocab[word] for word in vocab.keys() if 'I' in word]
    token_id_forbid += [vocab[word] for word in vocab.keys() if 'Sn' in word]
    token_id_forbid += [vocab[word] for word in vocab.keys() if 'Br' in word]
    guided_log_probs[:, :self.config.sampling.target_length, token_id_forbid] = self.neg_infinity

    print(f'\n guided_log_probs min: {guided_log_probs.min()}')
    print(f' guided_log_probs max: {guided_log_probs.max()}')
    print(f' guided_log_probs mean: {guided_log_probs.mean()}')

    guided_probs = guided_log_probs.softmax(dim=-1)

    print(f'\n guided_probs min: {guided_probs.min()}')
    print(f' guided_probs max: {guided_probs.max()}')
    print(f' guided_probs mean: {guided_probs.mean()}')
    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag.to(bool), xt, xs)
    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'classifier_log_prob': classifier_log_prob}

  def _cbg_denoise_antibiotic_remdm_loop(
      self,
      conditioning_class: int,
      gamma: float,
      classifier_model: classifier.Classifier,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      use_approx: bool = False,  # whether to use first-order approximation
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
      step: int = None,
      t=None,
      dt=None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    if cache is not None:
      log_x_theta = cache['log_x_theta']
      classifier_log_prob = cache['classifier_log_prob']
    else:
      # Diffusion model
      log_x_theta = self.forward(xt, time_conditioning, cond=None)
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_x_theta.exp()
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        log_x_theta = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)  # 这个地方直接是概率了不是 log probability
      # Classifier model
      if use_approx:
        xt_one_hot = torch.nn.functional.one_hot(xt, self.vocab_size).to(torch.float)
        if self.config.guidance.var_length:
          backup_shape = xt_one_hot.shape
          xt_one_hot_backup = xt_one_hot
          xt_one_hot = xt_one_hot[:, :self.config.sampling.target_length, :]
        with torch.enable_grad():
          xt_one_hot.requires_grad_(True)
          classifier_log_prob_xt = classifier_model.get_log_probs_antibiotic_guaidance(xt_one_hot, time_conditioning, step=step)
          classifier_log_prob_xt.sum().backward()
          grad_log_prob_xt = xt_one_hot.grad
          if self.config.guidance.var_length:
            true_grad = torch.zeros(backup_shape, device=xt_one_hot.device)
            true_grad[:, :self.config.sampling.target_length, :] = grad_log_prob_xt
            grad_log_prob_xt = true_grad
            xt_one_hot = xt_one_hot_backup

        classifier_log_prob_ratio = (
            grad_log_prob_xt - (xt_one_hot * grad_log_prob_xt).sum(dim=-1, keepdim=True)
        ).detach().requires_grad_(False)

        classifier_log_prob = (
            classifier_log_prob_ratio + classifier_log_prob_xt[..., None]
        ).detach().requires_grad_(False)

        print(f' classifier_log_prob_xt min: {classifier_log_prob_xt.min()}')
        print(f' classifier_log_prob_xt max: {classifier_log_prob_xt.max()}')
        print(f' classifier_log_prob_xt mean: {classifier_log_prob_xt.mean()}')

        print(f' classifier_log_prob_ratio min: {classifier_log_prob_ratio.min()}')
        print(f' classifier_log_prob_ratio max: {classifier_log_prob_ratio.max()}')
        print(f' classifier_log_prob_ratio mean: {classifier_log_prob_ratio.mean()}')

        print(f' classifier_log_prob min: {classifier_log_prob.min()}')
        print(f' classifier_log_prob max: {classifier_log_prob.max()}')
        print(f' classifier_log_prob mean: {classifier_log_prob.mean()}')

      else:
        # Copied from https://github.com/hnisonoff/discrete_guidance/blob/main/src/fm_utils.py#L441
        bsz, seq_len = xt.shape
        # Create bsz*seq_len*N copies of input sequences
        # Shape: (bsz, 1, seq_len) -> (bsz, seq_len*N, seq_len)
        # (where N = vocab_size).
        xt_expand = xt.unsqueeze(1).repeat(1, seq_len * self.vocab_size, 1)
        # Flatten batch and transition dimensions
        # Shape: (bsz, seq_len*N, seq_len) -> (bsz*seq_len*N, seq_len)
        xt_expand = xt_expand.view(-1, seq_len)

        # Create indices for all possible transitions
        # Shape: (seq_len*N,) -> (bsz, seq_len*N) -> (bsz*seq_len*N,)
        jump_idx = torch.arange(seq_len * self.vocab_size).to(xt.device)
        jump_idx = jump_idx.repeat(bsz, 1).flatten()

        # Create tensor for states after one transition
        xt_jumps = xt_expand.clone()

        # Calculate which dimension changes for each transition
        # Shape: (bsz*seq_len*N,)
        jump_dims = jump_idx // self.vocab_size

        # Calculate new value for changed dimension
        # Shape: (bsz*seq_len*N,)
        jump_states = jump_idx % self.vocab_size

        # Apply transitions by assigning new values at transition dimensions
        # Shape: (bsz*seq_len*N, seq_len)
        xt_jumps[
          torch.arange(jump_idx.size(0), device=xt.device),
          jump_dims,  # Index the transitioned dimension
        ] = jump_states  # Assign the new state

        classifier_log_prob = classifier_model.get_log_probs_antibiotic_guaidance(
          xt_jumps, time_conditioning.repeat(seq_len * self.vocab_size)
        ).reshape(bsz, seq_len, self.vocab_size)

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':

      if self.config.sampling.nucleus_p < 1:  # 上面可能直接出现就是概率的而不是 log probability 的 log_x_theta
        p_x0 = log_x_theta
      else:
        p_x0 = log_x_theta.exp()

      if t.ndim > 1:
        t = t.squeeze(-1)  # shape: [batch_size, ]
      time = t[0].item()
      # compute alpha_t and alpha_s
      if time > self.config.sampling.remdm.t_on:
        gamma = self.config.guidance.var_gamma.gamma_l
        self.config.guidance.reg_guide_weight = 1.0
        self.config.guidance.cls_guide_weight = 0.0
        # self.config.guidance.reg_guide_weight = 0.5
        # self.config.guidance.cls_guide_weight = 0.5
        move_chance_t = (1 - (1 - t) * self.config.sampling.remdm.alpha_on / (1 - self.config.sampling.remdm.t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * self.config.sampling.remdm.alpha_on / (1 - self.config.sampling.remdm.t_on))[:, None, None]
      elif time <= self.config.sampling.remdm.t_off:
        gamma = self.config.guidance.var_gamma.gamma_l
        self.config.guidance.reg_guide_weight = 1.0
        self.config.guidance.cls_guide_weight = 0.0
        # self.config.guidance.reg_guide_weight = 0.5
        # self.config.guidance.cls_guide_weight = 0.5
        move_chance_t = (t * (1 - self.config.sampling.remdm.alpha_on) / self.config.sampling.remdm.t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - self.config.sampling.remdm.alpha_on) / self.config.sampling.remdm.t_off)[:, None, None]
      else:
        gamma = self.config.guidance.var_gamma.gamma_s
        self.config.guidance.reg_guide_weight = 0.0
        self.config.guidance.cls_guide_weight = 1.0
        # self.config.guidance.reg_guide_weight = 0.5
        # self.config.guidance.cls_guide_weight = 0.5
        move_chance_t, move_chance_s = None, None
      # use MDLM
      if time > self.config.sampling.remdm.t_on or time <= self.config.sampling.remdm.t_off:
        # q_xs = p_x0 * (move_chance_t - move_chance_s)
        # q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        q_xs = p_x0 * (1. - (move_chance_s / move_chance_t))
        q_xs[:, :, self.mask_index] = (move_chance_s / move_chance_t)[:, :, 0]
        diffusion_log_probs = torch.log(q_xs + 1e-8)
        # _x = _sample_categorical(q_xs)
        # copy_flag = (x != self.mask_index).to(x.dtype)
        # xs = copy_flag * x + (1 - copy_flag) * _x
      else:  # use ReMDM
        sigma = self.config.sampling.remdm.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((self.config.sampling.remdm.alpha_on - (1 - sigma) * self.config.sampling.remdm.alpha_on) / (
                1 - self.config.sampling.remdm.alpha_on))
        q_xs_2[..., self.mask_index] = (1 - self.config.sampling.remdm.alpha_on - self.config.sampling.remdm.alpha_on * sigma) / (
                1 - self.config.sampling.remdm.alpha_on)
        copy_flag = (xt != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        diffusion_log_probs = torch.log(q_xs + 1e-8)
        # xs = _sample_categorical(q_xs)

      # diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      # diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      # diffusion_log_probs.detach()

      print(f' diffusion_log_probs min: {diffusion_log_probs.min()}')
      print(f' diffusion_log_probs max: {diffusion_log_probs.max()}')
      print(f' diffusion_log_probs mean: {diffusion_log_probs.mean()}')

    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(
        x=log_x_theta.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    # Apply guidance
    with torch.no_grad():
      if self.diffusion == 'absorbing_state':

        # use MDLM
        if time > self.config.sampling.remdm.t_on or time <= self.config.sampling.remdm.t_off:
          guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
          copy_flag = (xt != self.mask_index)
          guided_log_probs[copy_flag] = self.neg_infinity
          guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
        # use ReMDM
        else:
          guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
          copy_flag = (xt != self.mask_index)
          guided_log_probs_copy = guided_log_probs.clone()
          guided_log_probs[copy_flag] = self.neg_infinity
          guided_log_probs[copy_flag, xt[copy_flag]] = guided_log_probs_copy[copy_flag, xt[copy_flag]]
          guided_log_probs[copy_flag, self.mask_index * torch.ones_like(xt[copy_flag])] = guided_log_probs_copy[copy_flag, self.mask_index * torch.ones_like(xt[copy_flag])]

          # 防止padding全都被remask了
          guided_log_probs[:, self.config.sampling.target_length:, :] = self.neg_infinity
          guided_log_probs[:, self.config.sampling.target_length:, self.tokenizer.pad_token_id] = 0.0

        # 禁止生成 '.'
        vocab = self.tokenizer.get_vocab()
        token_id_forbid = []
        # forbidden_list = ['.', 'He', 'Li', 'Be', 'B', 'Ne',
        #                   'Na','Mg','Al','Si', 'Ar','K', 'Ca',
        #                   'Sc','Ti','V', 'Cr','Mn','Fe','Co','Ni','Cu','Zn',
        #                   'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y', 'Zr',
        #                   'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
        #                   'Sb','Te','I', 'Xe','Cs','Ba','La','Ce','Pr','Nd',
        #                   'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
        #                   'Lu','Hf','Ta','W', 'Re','Os','Ir','Pt','Au','Hg',
        #                   'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',
        #                   'Pa','U', 'Np','Pu','Am','Cm','Bk','Cf','Es','Fm',
        #                   'Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',
        #                   'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

        forbidden_list = ['.', 'I', 'Sn', 'Br']
        for forbiden_id in forbidden_list:
          token_id_forbid += [vocab[word] for word in vocab.keys() if forbiden_id in word]
        guided_log_probs[:, :self.config.sampling.target_length, token_id_forbid] = self.neg_infinity

        # valid_ids = ['[CLS]', '[SEP]', '[/N]', '[C@]', '[Branch1]', '[=Branch2]', '[/C]', '[SH1]', '[#Branch1]',
        #              '[F]', '[=Ring2]', '[=Ring1]', '[\\O]', '[P]', '[Ring2]', '[\\N]', '[C@@]',
        #             '[=S]', '[C@H1]', '[\\C]', '[OH0]', '[Ring1]', '[=C]', '[OH1]', '[CH1-1]',
        #              '[N+1]', '[/C@H1]', '[=N]', '[2H]', '[S]', '[\\S]', '[O-1]',
        #              '[=Branch1]', '[#Branch2]', '[#C]', '[Cl]', '[C-1]', '[Branch2]', '[/S]', '[NH3+1]', '[\\-Ring2]',
        #              '[NH1]', '[N]', '[B]', '[C]', '[O]', '[=O]', '[C@@H1]', '[Branch3]',
        #              '[Ring3]', '[/-Ring2]', '[=Branch3]']
        # token_id_forbid = [vocab[word] for word in vocab.keys() if word not in valid_ids]
        # guided_log_probs[:, :self.config.sampling.target_length, token_id_forbid] = self.neg_infinity


      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * classifier_log_prob) + diffusion_log_probs
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")

    print(f'\n guided_log_probs min: {guided_log_probs.min()}')
    print(f' guided_log_probs max: {guided_log_probs.max()}')
    print(f' guided_log_probs mean: {guided_log_probs.mean()}')

    guided_probs = guided_log_probs.softmax(dim=-1)

    print(f'\n guided_probs min: {guided_probs.min()}')
    print(f' guided_probs max: {guided_probs.max()}')
    print(f' guided_probs mean: {guided_probs.mean()}')

    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    xs[:, 0] = self.tokenizer.cls_token_id
    if self.diffusion == 'absorbing_state':
      if time > self.config.sampling.remdm.t_on or time <= self.config.sampling.remdm.t_off:
        xs = torch.where(copy_flag.to(bool), xt, xs)
    # with open('/data2/tianang/projects/discrete-diffusion-guidance/outputs/visualize_gen_process/try_1.txt', 'a') as f:
    #   text = self.tokenizer.decode(xs[0])
    #   f.write(text+'\n')

    # for i in range(xs.shape[0]):
    #   with open(f'/data2/tianang/projects/discrete-diffusion-guidance/outputs/visualize_gen_process/try_{i + 1}.txt',
    #             'a') as f:
    #     text = self.tokenizer.decode(xs[i])
    #     f.write(text + '\n')

    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'classifier_log_prob': classifier_log_prob}

  def _nos_denoise(
      self,
      classifier_model: classifier.Classifier,
      num_nos_steps: int,
      nos_step_size: float,
      nos_stability_coef: float,
      conditioning_class: int,
      xt: torch.Tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
  ) -> typing.Tuple[torch.tensor, torch.tensor, None]:
    # Compute original diffusion_log_probs and hidden states
    copy_flag = (xt != self.mask_index).to(torch.bool)
    with torch.no_grad():
      time_conditioning = self._process_sigma(time_conditioning)
      with torch.cuda.amp.autocast(dtype=torch.float32):
        logits, hidden_states = self.backbone(xt, time_conditioning, cond=None, return_hidden_states=True)
        if self.parameterization == 'subs':
          log_x_theta = self._subs_parameterization(logits=logits, xt=xt)
        elif self.parameterization == 'd3pm':
          # returns log_probs
          if self.subs_masking:  # Can use "zero masking prob"
            logits[:, :,
            self.mask_index] += self.neg_infinity
          log_x_theta = logits.log_softmax(dim=-1)
        else:
          raise NotImplementedError(
            f"Parameterization {self.parameterization} not implemented for NOS guidance.")
        if self.diffusion == 'absorbing_state':
          diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
          diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
          diffusion_log_probs[copy_flag] = self.neg_infinity
          diffusion_log_probs[copy_flag, xt[copy_flag]] = 0.0
        elif self.diffusion == 'uniform':
          diffusion_log_probs = self._compute_posterior(
            x=log_x_theta.exp(),
            xt=xt,
            alpha_s=1 - move_chance_s,
            alpha_t=1 - move_chance_t).log()

    # Perform NOS steps
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean',
                                 log_target=True)
    delta = torch.nn.Parameter(
      torch.zeros_like(hidden_states[-1]),
      requires_grad=True)
    optimizer = torch.optim.Adagrad([delta], lr=nos_step_size)
    with torch.enable_grad():
      for _ in tqdm(range(num_nos_steps),desc='NOS', leave=False):
        h_current = hidden_states[-1] + delta
        target_loss = classifier_model.get_log_probs(xt, time_conditioning, x_emb=h_current)[..., conditioning_class].sum()
        with torch.cuda.amp.autocast(dtype=torch.float32):
          new_logits = self.forward(xt, time_conditioning,
                                    cond=None,
                                    x_emb=h_current)
        if self.diffusion == 'absorbing_state':
          adjusted_log_probs = new_logits + torch.log(1. - (move_chance_s / move_chance_t))
          adjusted_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
          adjusted_log_probs[copy_flag] = self.neg_infinity
          adjusted_log_probs[copy_flag, xt[copy_flag]] = 0.0
        elif self.diffusion == 'uniform':
          adjusted_log_probs = self._compute_posterior(
            x=new_logits.exp(),
            xt=xt,
            alpha_s=1 - move_chance_s,
            alpha_t=1 - move_chance_t).log()
        kl = kl_loss(adjusted_log_probs, diffusion_log_probs)
        loss = -target_loss + nos_stability_coef * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.cuda.amp.autocast(dtype=torch.float32):
      guided_logits = self.forward(
        xt, time_conditioning,
        cond=None,
        x_emb=hidden_states[-1] + delta.data)
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = guided_logits + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()
      guided_probs = diffusion_log_probs.exp()
    elif self.diffusion == 'uniform':
      guided_probs = self._compute_posterior(
        x=guided_logits.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).detach()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag, xt, xs)

    return xs, guided_probs, None

  def _nos_denoise_AMP(
      self,
      classifier_model: classifier.Classifier,
      num_nos_steps: int,
      nos_step_size: float,
      nos_stability_coef: float,
      conditioning_class: int,
      xt: torch.Tensor,
      time_conditioning: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      step: int = None,
  ) -> typing.Tuple[torch.tensor, torch.tensor, None]:
    # Compute original diffusion_log_probs and hidden states
    copy_flag = (xt != self.mask_index).to(torch.bool)
    with torch.no_grad():
      time_conditioning = self._process_sigma(time_conditioning)
      with torch.cuda.amp.autocast(dtype=torch.float32):
        logits, hidden_states = self.backbone(xt, time_conditioning, cond=None, return_hidden_states=True)
        if self.parameterization == 'subs':
          log_x_theta = self._subs_parameterization(logits=logits, xt=xt)
        elif self.parameterization == 'd3pm':
          # returns log_probs
          if self.subs_masking:  # Can use "zero masking prob"
            logits[:, :,
            self.mask_index] += self.neg_infinity
          log_x_theta = logits.log_softmax(dim=-1)
        else:
          raise NotImplementedError(
            f"Parameterization {self.parameterization} not implemented for NOS guidance.")
        if self.diffusion == 'absorbing_state':
          diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
          diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
          diffusion_log_probs[copy_flag] = self.neg_infinity
          diffusion_log_probs[copy_flag, xt[copy_flag]] = 0.0
        elif self.diffusion == 'uniform':
          diffusion_log_probs = self._compute_posterior(
            x=log_x_theta.exp(),
            xt=xt,
            alpha_s=1 - move_chance_s,
            alpha_t=1 - move_chance_t).log()

    # Perform NOS steps
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean',
                                 log_target=True)
    delta = torch.nn.Parameter(
      torch.zeros_like(hidden_states[-1]),
      requires_grad=True)
    optimizer = torch.optim.Adagrad([delta], lr=nos_step_size)
    with torch.enable_grad():
      for _ in tqdm(range(num_nos_steps),desc='NOS', leave=False):
        h_current = hidden_states[-1] + delta
        target_loss = classifier_model.get_log_probs_antibiotic_guaidance(xt, time_conditioning, x_emb=h_current, step=step).sum()
        with torch.cuda.amp.autocast(dtype=torch.float32):
          new_logits = self.forward(xt, time_conditioning,
                                    cond=None,
                                    x_emb=h_current)
        if self.diffusion == 'absorbing_state':
          adjusted_log_probs = new_logits + torch.log(1. - (move_chance_s / move_chance_t))
          adjusted_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
          adjusted_log_probs[copy_flag] = self.neg_infinity
          adjusted_log_probs[copy_flag, xt[copy_flag]] = 0.0
        elif self.diffusion == 'uniform':
          adjusted_log_probs = self._compute_posterior(
            x=new_logits.exp(),
            xt=xt,
            alpha_s=1 - move_chance_s,
            alpha_t=1 - move_chance_t).log()
        kl = kl_loss(adjusted_log_probs, diffusion_log_probs)
        loss = -target_loss + nos_stability_coef * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.cuda.amp.autocast(dtype=torch.float32):
      guided_logits = self.forward(
        xt, time_conditioning,
        cond=None,
        x_emb=hidden_states[-1] + delta.data)
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = guided_logits + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()
      guided_probs = diffusion_log_probs.exp()
    elif self.diffusion == 'uniform':
      guided_probs = self._compute_posterior(
        x=guided_logits.exp(),
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t).detach()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag, xt, xs)

    return xs, guided_probs, None

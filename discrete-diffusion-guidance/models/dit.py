from collections import OrderedDict

import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from . import antibiotic_classifier
import numpy as np
from pathlib import Path
from torch.distributions import Normal

from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: typing.Optional[torch.Tensor],
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is None:
    out = F.dropout(x, p=prob, training=training)
  else:
    out = F.dropout(x + bias, p=prob, training=training)
  if scale is not None:
    out = scale * out
  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: typing.Optional[torch.Tensor],
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: typing.Optional[torch.Tensor],
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(
        x.shape[seq_dim],
        device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t,
                           self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv,
                                                        cos,
                                                        sin)

# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


############################################################
#                        Layers                            #
############################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


############################################################
#     Embedding Layers for Timesteps and Class Labels      #
############################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size,
               frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size,
                bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, 1 / batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the
      embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat(
      [torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations."""
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes,
                                        cond_size)
    self.num_classes = num_classes

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    # embeddings = self.mlp(embeddings)
    return embeddings


############################################################
#                      Core Model                          #
############################################################


class DDiTBlock(nn.Module):
  def __init__(
      self,
      dim,
      n_heads,
      cond_dim,
      mlp_ratio=4,
      dropout=0.1,
      causal=False,
      use_adaLN=True,
  ):
    super().__init__()
    self.n_heads = n_heads
    self.causal = causal

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.use_adaLN = use_adaLN
    if use_adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    if self.use_adaLN:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(
        c)[:, None].chunk(6, dim=2)
    else:
      (shift_msa, scale_msa, gate_msa,
       shift_mlp, scale_mlp, gate_mlp) = (
        None, None, None, None, None, None)

    # attention operation
    x_skip = x
    if self.use_adaLN:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    qkv = self.attn_qkv(x)
    qkv = rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    if seqlens is None:
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
    else:
      cu_seqlens = seqlens.cumsum(-1)
    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      qkv, cu_seqlens, seq_len, 0., causal=self.causal)

    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x_skip = x
    if self.use_adaLN:
      x = modulate_fused(self.norm2(x), shift_mlp, scale_mlp)
    else:
      x = self.norm2(x)
    return bias_dropout_scale_fn(
        self.mlp(x),None, gate_mlp, x_skip, self.dropout)


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(
      torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(
      self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(
      self, hidden_size, out_channels, cond_dim,
      use_adaLN=True):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    if use_adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    else:
      self.adaLN_modulation = None

  def forward(self, x, c):
    if self.adaLN_modulation is not None:
      shift, scale = self.adaLN_modulation(
        c)[:, None].chunk(2, dim=2)
      x = modulate_fused(self.norm_final(x), shift, scale)
      x = self.linear(x)
      return x
    return self.linear(self.norm_final(x))


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = config.parameterization == 'ar'

    self.vocab_embed = EmbeddingLayer(
      config.model.hidden_size, vocab_size)

    if self.causal:
      self.sigma_map = None  # no timestep embedding for AR
    else:
      self.sigma_map = TimestepEmbedder(config.model.cond_dim)

    if (config.training.guidance is not None or  # Training for / using CFG
        (hasattr(config, 'guidance')
         and config.guidance is not None
         and config.guidance.method == 'cfg')):
      self.cond_map = LabelEmbedder(
        config.data.num_classes + 1,  # +1 for mask
        config.model.cond_dim)
    else:
      self.cond_map = None
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    use_adaLN = (config.parameterization != 'ar' or
                 self.cond_map is not None)
    for _ in range(config.model.n_blocks):
      blocks.append(
        DDiTBlock(
          config.model.hidden_size,
          config.model.n_heads,
          config.model.cond_dim,
          dropout=config.model.dropout,
          causal=self.causal,
          use_adaLN=use_adaLN,))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim,
      use_adaLN=use_adaLN)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma,
              cond=None,
              x_emb=None,
              return_hidden_states=False):
    if return_hidden_states:
      hidden_states = []

    if self.causal:
      c = None
    else:
      c = F.silu(self.sigma_map(sigma))
    if cond is not None:
      if self.cond_map is None:
        raise ValueError("Conditioning variable provided, "
                         "but Model was not initialized "
                         "with condition embedding layer.")
      else:
        if c is None:  # AR (self.causal is True)
          c = F.silu(self.cond_map(cond))
        else:
          c = c + F.silu(self.cond_map(cond))

    if x_emb is None:
      x = self.vocab_embed(indices)
      if return_hidden_states:
        hidden_states.append(x)
      rotary_cos_sin = self.rotary_emb(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
          if return_hidden_states:
            hidden_states.append(x)
    else:
      x = x_emb

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      x = self.output_layer(x, c)

    if return_hidden_states:
      return x, hidden_states
    return x

class DDiTBlock_non_pad(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c, seqlens=None, attnmask = None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)  # dim -> 3 * dim
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=True):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')

    # --------------------------------
    mask_flat = attnmask.reshape(-1)
    qkv = qkv[mask_flat]
    seqlens = attnmask.sum(dim=1)
    pad_seq_len = torch.zeros(len(seqlens)+1, dtype=torch.int32, device=qkv.device)
    pad_seq_len[1:] = seqlens
    seqlens = pad_seq_len
    # cu_seqlens = pad_seq_len.cumsum(-1)
    # x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
    #   qkv, cu_seqlens, seq_len, 0., causal=False)
    # --------------------------------

    if seqlens is None:
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
    else:
      cu_seqlens = seqlens.cumsum(-1).to(torch.int32)

    assert cu_seqlens.min() == 0, "cu_seqlens 最小值必须等于 0"
    assert qkv.size(0) == cu_seqlens[-1], "token 总数和 cu_seqlens 不符"

    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      qkv, cu_seqlens, seq_len, 0., causal=False)

    # --------------------------------
    out_flat = torch.zeros([batch_size*seq_len, x.shape[1], x.shape[2]]).to(x.device).to(x.dtype)
    out_flat[mask_flat] = x
    x = out_flat
    # --------------------------------

    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x

class DIT_non_pad(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock_non_pad(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma, attnmask):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None, attnmask=attnmask)
      x = self.output_layer(x, c)

    return x

class DITClassifier(nn.Module):
  def __init__(self, config, vocab_size):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = config.parameterization == 'ar'

    self.vocab_embed = EmbeddingLayer(
      config.classifier_model.hidden_size, vocab_size)

    if self.causal:
      self.sigma_map = None
    else:
      self.sigma_map = TimestepEmbedder(config.classifier_model.cond_dim)

    self.rotary_emb = Rotary(
      config.classifier_model.hidden_size // config.classifier_model.n_heads)

    blocks = []
    use_adaLN = config.parameterization != 'ar'
    for _ in range(config.classifier_model.n_blocks):
      blocks.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  causal=self.causal,
                  use_adaLN=use_adaLN))
    self.blocks = nn.ModuleList(blocks)

    self.scale_by_sigma = config.classifier_model.scale_by_sigma

    self.pooling = getattr(config.classifier_model, 'pooling', 'mean')
    self.output_layer = nn.Linear(
      config.classifier_model.hidden_size,
      config.classifier_model.num_classes)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices_or_one_hots, sigma, x_emb=None, attention_mask=None):
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float),
                     self.vocab_embed.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map(sigma))

      rotary_cos_sin = self.rotary_emb(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      x = x_emb

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if self.pooling == 'mean':
        x = x.mean(dim=1)
      elif self.pooling == 'max':
        x = x.max(dim=1)
      elif self.pooling == 'cls':
        x = x[..., 0]
      elif self.pooling == 'last':
        x = x[..., -1]
      elif self.pooling == 'no_pooling':  # for ar_fudge
        pass
      elif self.pooling == 'attention_mean':  # for ar_pplm
        masked_x = x * attention_mask.unsqueeze(2)
        x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
      else:
        raise NotImplementedError(
          f"`{self.pooling}` method not implemented.")
      x = self.output_layer(x)
    return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks

class DITClassifier_AMP(nn.Module):
  def __init__(self, config, vocab_size):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = config.parameterization == 'ar'

    self.vocab_embed = EmbeddingLayer(config.classifier_model.hidden_size, vocab_size)

    if self.causal:
      self.sigma_map = None
    else:
      self.sigma_map = TimestepEmbedder(config.classifier_model.cond_dim)

    self.rotary_emb = Rotary(config.classifier_model.hidden_size // config.classifier_model.n_heads)

    blocks = []
    use_adaLN = config.parameterization != 'ar'
    for _ in range(config.classifier_model.n_blocks):
      blocks.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  causal=self.causal,
                  use_adaLN=use_adaLN))
    self.blocks = nn.ModuleList(blocks)

    self.co_cross_attn_genome = antibiotic_classifier.FirstTokenAttention_genome(config.classifier_model.hidden_size,
                                                                                 8192,
                                                                                 4,
                                                                                 0.1)
    self.co_cross_attn_text = antibiotic_classifier.FirstTokenAttention_genome(config.classifier_model.hidden_size,
                                                                               4096,
                                                                               4,
                                                                               0.1)
    self.reg_head = antibiotic_classifier.RegressionHead(8192+4096, (8192+4096)//4, 128, 1, 0.2)
    self.learnable_embedding_weight = nn.Parameter(torch.randn(1, 8192))

    self.scale_by_sigma = config.classifier_model.scale_by_sigma

    self.ATCC_genome_emb_dict, self.ATCC_text_emb_dict, self.text_only_emb_dict, self.strain_cond = self.load_genome_test_embedding()

    # self.pooling = getattr(config.classifier_model, 'pooling', 'mean')
    # self.output_layer = nn.Linear(
    #   config.classifier_model.hidden_size,
    #   config.classifier_model.num_classes)

  def load_pretrained_weight(self):
    """
    在使用 forward 之前进行，load 自己要用的权重
    """
    regressor_ckpt_path = self.config.guidance.regressor_checkpoint_path
    checkpoint = torch.load(regressor_ckpt_path, map_location='cuda')
    new_sd = OrderedDict()
    for k, v in checkpoint['mdlm_model_state_dict'].items():
      if k.startswith('backbone.'):
        new_key = k[len('backbone.'):]
      else:
        new_key = k
      new_sd[new_key] = v
    self.load_state_dict(new_sd, strict=False)
    self.reg_head.load_state_dict(checkpoint['re_head_state_dict'])
    self.co_cross_attn_genome.load_state_dict(checkpoint['co_cross_attn_genome'])
    self.co_cross_attn_text.load_state_dict(checkpoint['co_cross_attn_text'])
    self.learnable_embedding_weight = checkpoint['learnable_embedding_weight']


  def load_genome_test_embedding(self):
    ATCC_genome_emb_dict = antibiotic_classifier.load_all_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.ATCC_genome_emb),
                                                                            1e14,
                                                                            'cpu',
                                                                            'ATCC genome embedding')
    ATCC_text_emb_dict = antibiotic_classifier.load_all_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.ATCC_text_emb),
                                                                            1,
                                                                            'cpu',
                                                                            'ATCC text embedding')
    text_only_emb_dict = antibiotic_classifier.load_text_wo_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.only_text_emb),
                                                                            1,
                                                                            'cpu',
                                                                            'text only embedding')
    strain_cond = str(self.config.sampling.strain)
    return ATCC_genome_emb_dict, ATCC_text_emb_dict, text_only_emb_dict, strain_cond


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def comp_reg_log(self, target, pred, step = None):
    # target = torch.tensor(target)[None, ...].expand(pred.shape[0], -1).to('cuda')
    target = self.config.sampling.target_MIC_max - (self.config.sampling.target_MIC_max - target) / self.config.sampling.steps * step
    gaussian_sigma = self.config.sampling.gaussian_sigma_max - (self.config.sampling.gaussian_sigma_max - self.config.sampling.gaussian_sigma_min) / self.config.sampling.steps * step

    print(f'\n gaussian_sigma: {gaussian_sigma}')
    print(f'\n target: {target}')

    target = torch.tensor(target)
    target_trans = -torch.log10(target / 10)[None, ...].expand(pred.shape[0], -1).to('cuda')
    # pred = 10**(-pred)*10
    sigma = torch.ones_like(target).to('cuda') * gaussian_sigma
    dist = Normal(target_trans, sigma)
    cdf_pred = dist.cdf(pred)
    mask = cdf_pred > 0.5
    cdf_pred[mask] = 1 - cdf_pred[mask]
    log_pdf = torch.log(cdf_pred * 2 + 1e-9)

    # log_pdf = torch.log(F.sigmoid(-torch.exp(torch.abs(target_trans - pred))))

    # log_pdf = dist.log_prob(pred)
    return log_pdf

  def forward(self, indices_or_one_hots, sigma, x_emb=None, attention_mask=None, step=None):
    # step = x_emb
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float), self.vocab_embed.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map(sigma))

      rotary_cos_sin = self.rotary_emb(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      x = x_emb

    mol_cls_embedding = x[:, 0, :] # cls

    # 加载 genome embedding 和 text embedding
    if self.strain_cond in self.ATCC_genome_emb_dict.keys():
      padded_genome_embeddings = self.ATCC_genome_emb_dict[self.strain_cond].to('cuda')
      padded_text_embeddings = self.ATCC_text_emb_dict[self.strain_cond].to('cuda')
      padded_genome_embeddings = padded_genome_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
      genome_attn_masks = torch.ones(padded_genome_embeddings.shape[0], padded_genome_embeddings.shape[1]).to('cuda')
    else:
      padded_genome_embeddings = None
      padded_text_embeddings = self.text_only_emb_dict[self.strain_cond].to('cuda')
    padded_text_embeddings = padded_text_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
    text_attn_masks = torch.ones(padded_text_embeddings.shape[0], padded_text_embeddings.shape[1]).to('cuda')

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if padded_genome_embeddings is not None:
        mol_cls_embedding_genome = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
      else:
        padded_genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
        genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1)
        mol_cls_embedding_genome = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
      mol_cls_embedding_text = self.co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
      mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
      reg_logits = self.reg_head(mol_cls_embedding)

    print(f' MIC: {(10 ** (1 - reg_logits)).mean()}')

    reg_log_prob = self.comp_reg_log(self.config.sampling.target_MIC, reg_logits, step)

    # print(f' reg_log_prob: {reg_log_prob}')

    return reg_log_prob # reg_log_prob

    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #   if self.pooling == 'mean':
    #     x = x.mean(dim=1)
    #   elif self.pooling == 'max':
    #     x = x.max(dim=1)
    #   elif self.pooling == 'cls':
    #     x = x[..., 0]
    #   elif self.pooling == 'last':
    #     x = x[..., -1]
    #   elif self.pooling == 'no_pooling':  # for ar_fudge
    #     pass
    #   elif self.pooling == 'attention_mean':  # for ar_pplm
    #     masked_x = x * attention_mask.unsqueeze(2)
    #     x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
    #   else:
    #     raise NotImplementedError(
    #       f"`{self.pooling}` method not implemented.")
    #   x = self.output_layer(x)
    # return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks


class DIT_Reg_Cls_AMP(nn.Module):
  def __init__(self, config, vocab_size):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = config.parameterization == 'ar'

    self.vocab_embed_reg = EmbeddingLayer(config.classifier_model.hidden_size, vocab_size)
    self.vocab_embed_cls = EmbeddingLayer(config.classifier_model.hidden_size, vocab_size)

    if self.causal:
      self.sigma_map = None
    else:
      self.sigma_map_reg = TimestepEmbedder(config.classifier_model.cond_dim)
      self.sigma_map_cls = TimestepEmbedder(config.classifier_model.cond_dim)

    self.rotary_emb_reg = Rotary(config.classifier_model.hidden_size // config.classifier_model.n_heads)
    self.rotary_emb_cls = Rotary(config.classifier_model.hidden_size // config.classifier_model.n_heads)

    # 构建回归用的
    blocks_reg = []
    use_adaLN = config.parameterization != 'ar'
    for _ in range(config.classifier_model.n_blocks):
      blocks_reg.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  causal=self.causal,
                  use_adaLN=use_adaLN))
    self.blocks_reg = nn.ModuleList(blocks_reg)

    blocks_cls = []
    use_adaLN = config.parameterization != 'ar'
    for _ in range(config.classifier_model.n_blocks):
      blocks_cls.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  causal=self.causal,
                  use_adaLN=use_adaLN))
    self.blocks_cls = nn.ModuleList(blocks_cls)

    self.co_cross_attn_genome = antibiotic_classifier.FirstTokenAttention_genome(config.classifier_model.hidden_size,
                                                                                 8192,
                                                                                 4,
                                                                                 0.1)
    self.co_cross_attn_text = antibiotic_classifier.FirstTokenAttention_genome(config.classifier_model.hidden_size,
                                                                               4096,
                                                                               4,
                                                                               0.1)
    self.reg_head = antibiotic_classifier.RegressionHead(8192+4096, (8192+4096)//4, 128, 1, 0.2)
    self.learnable_embedding_weight = nn.Parameter(torch.randn(1, 8192))

    self.ClsHead = ClsHead(input_dim=768, num_targets=1)

    self.scale_by_sigma = config.classifier_model.scale_by_sigma

    self.ATCC_genome_emb_dict, self.ATCC_text_emb_dict, self.text_only_emb_dict, self.strain_cond = self.load_genome_test_embedding()

    # self.pooling = getattr(config.classifier_model, 'pooling', 'mean')
    # self.output_layer = nn.Linear(
    #   config.classifier_model.hidden_size,
    #   config.classifier_model.num_classes)

  def load_pretrained_weight(self):
    """
    在使用 forward 之前进行，load 自己要用的权重
    """

    # 加载 regressor 的权重
    regressor_ckpt_path = self.config.guidance.regressor_checkpoint_path
    checkpoint_reg = torch.load(regressor_ckpt_path, map_location='cuda')
    new_sd = OrderedDict()
    for k, v in checkpoint_reg['mdlm_model_state_dict'].items():
      if k.startswith('backbone.'):
        new_key = k[len('backbone.'):]
        if 'blocks' in new_key:
          new_key = new_key.replace('blocks', 'blocks_reg')
        elif 'vocab_embed' in new_key:
          new_key = new_key.replace('vocab_embed', 'vocab_embed_reg')
        elif 'sigma_map' in new_key:
          new_key = new_key.replace('sigma_map', 'sigma_map_reg')
        elif 'rotary_emb' in new_key:
          new_key = new_key.replace('rotary_emb', 'rotary_emb_reg')
      else:
        new_key = k
      new_sd[new_key] = v
    self.load_state_dict(new_sd, strict=False)

    # 加载 classifier 的权重
    classifier_ckpt_path = self.config.guidance.classifier_checkpoint_path
    checkpoint_cls = torch.load(classifier_ckpt_path, map_location='cuda')
    new_sd = OrderedDict()
    for k, v in checkpoint_cls['state_dict'].items():
      if k.startswith('backbone.backbone.'):
        new_key = k[len('backbone.backbone.'):]
        if 'blocks' in new_key:
          new_key = new_key.replace('blocks', 'blocks_cls')
        elif 'vocab_embed' in new_key:
          new_key = new_key.replace('vocab_embed', 'vocab_embed_cls')
        elif 'sigma_map' in new_key:
          new_key = new_key.replace('sigma_map', 'sigma_map_cls')
        elif 'rotary_emb' in new_key:
          new_key = new_key.replace('rotary_emb', 'rotary_emb_cls')
      else:
        new_key = k  # 这个地方应该已经把 ClsHead 包进去了
      new_sd[new_key] = v
    self.load_state_dict(new_sd, strict=False)

    self.reg_head.load_state_dict(checkpoint_reg['re_head_state_dict'])
    self.co_cross_attn_genome.load_state_dict(checkpoint_reg['co_cross_attn_genome'])
    self.co_cross_attn_text.load_state_dict(checkpoint_reg['co_cross_attn_text'])
    self.learnable_embedding_weight = checkpoint_reg['learnable_embedding_weight']

    # self.cls_head.load_state_dict(checkpoint_cls['ClsHead'])


  def load_genome_test_embedding(self):
    ATCC_genome_emb_dict = antibiotic_classifier.load_all_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.ATCC_genome_emb),
                                                                            1e14,
                                                                            'cpu',
                                                                            'ATCC genome embedding')
    ATCC_text_emb_dict = antibiotic_classifier.load_all_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.ATCC_text_emb),
                                                                            1,
                                                                            'cpu',
                                                                            'ATCC text embedding')
    text_only_emb_dict = antibiotic_classifier.load_text_wo_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.only_text_emb),
                                                                            1,
                                                                            'cpu',
                                                                            'text only embedding')
    strain_cond = str(self.config.sampling.strain)
    return ATCC_genome_emb_dict, ATCC_text_emb_dict, text_only_emb_dict, strain_cond


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def comp_reg_log(self, target, pred, step = None):
    # target = torch.tensor(target)[None, ...].expand(pred.shape[0], -1).to('cuda')
    target = self.config.sampling.target_MIC_max - (self.config.sampling.target_MIC_max - target) / self.config.sampling.steps * step
    gaussian_sigma = self.config.sampling.gaussian_sigma_max - (self.config.sampling.gaussian_sigma_max - self.config.sampling.gaussian_sigma_min) / self.config.sampling.steps * step

    print(f'\n gaussian_sigma: {gaussian_sigma}')
    print(f'\n target: {target}')

    target = torch.tensor(target)
    target_trans = -torch.log10(target / 10)[None, ...].expand(pred.shape[0], -1).to('cuda')
    # pred = 10**(-pred)*10
    sigma = torch.ones_like(target).to('cuda') * gaussian_sigma
    dist = Normal(target_trans, sigma)
    cdf_pred = dist.cdf(pred)
    mask = cdf_pred > 0.5
    cdf_pred[mask] = 1 - cdf_pred[mask]
    log_pdf = torch.log(cdf_pred * 2 + 1e-9)

    # log_pdf = torch.log(F.sigmoid(-torch.exp(torch.abs(target_trans - pred))))

    # log_pdf = dist.log_prob(pred)
    return log_pdf

  def forward(self, indices_or_one_hots, sigma, x_emb=None, attention_mask=None, step=None):
    # step = x_emb
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed_reg(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float), self.vocab_embed_reg.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map_reg(sigma))

      rotary_cos_sin = self.rotary_emb_reg(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks_reg)):
          x = self.blocks_reg[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      x = x_emb

    mol_cls_embedding = x[:, 0, :] # cls 这个都是正在生成的分子的 embedding
    # mol_cls_embedding_2 =

    # 加载 genome embedding 和 text embedding
    if self.strain_cond in self.ATCC_genome_emb_dict.keys():
      padded_genome_embeddings = self.ATCC_genome_emb_dict[self.strain_cond].to('cuda')
      padded_text_embeddings = self.ATCC_text_emb_dict[self.strain_cond].to('cuda')
      padded_genome_embeddings = padded_genome_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
      genome_attn_masks = torch.ones(padded_genome_embeddings.shape[0], padded_genome_embeddings.shape[1]).to('cuda')
    else:
      padded_genome_embeddings = None
      padded_text_embeddings = self.text_only_emb_dict[self.strain_cond].to('cuda')
    padded_text_embeddings = padded_text_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
    text_attn_masks = torch.ones(padded_text_embeddings.shape[0], padded_text_embeddings.shape[1]).to('cuda')

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if padded_genome_embeddings is not None:
        mol_cls_embedding_genome = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
      else:
        padded_genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
        genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1)
        mol_cls_embedding_genome = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
      mol_cls_embedding_text = self.co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
      mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
      reg_logits = self.reg_head(mol_cls_embedding)

    print(f' MIC: {(10 ** (1 - reg_logits)).mean()}')

    reg_log_prob = self.comp_reg_log(self.config.sampling.target_MIC, reg_logits, step)


    # 计算 pep_SM_cls 分类概率
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed_cls(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float), self.vocab_embed_cls.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map_cls(sigma))

      rotary_cos_sin = self.rotary_emb_cls(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks_cls)):
          x = self.blocks_cls[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      x = x_emb

    mol_cls_embedding = x[:, 0, :] # cls

    cls_log_prob = torch.log(torch.sigmoid(self.ClsHead(mol_cls_embedding)))

    # print(f' reg_log_prob: {reg_log_prob}')

    return self.config.guidance.reg_guide_weight * reg_log_prob + self.config.guidance.cls_guide_weight * cls_log_prob # reg_log_prob

    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #   if self.pooling == 'mean':
    #     x = x.mean(dim=1)
    #   elif self.pooling == 'max':
    #     x = x.max(dim=1)
    #   elif self.pooling == 'cls':
    #     x = x[..., 0]
    #   elif self.pooling == 'last':
    #     x = x[..., -1]
    #   elif self.pooling == 'no_pooling':  # for ar_fudge
    #     pass
    #   elif self.pooling == 'attention_mean':  # for ar_pplm
    #     masked_x = x * attention_mask.unsqueeze(2)
    #     x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
    #   else:
    #     raise NotImplementedError(
    #       f"`{self.pooling}` method not implemented.")
    #   x = self.output_layer(x)
    # return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks



class DIT_Syn_Cls_Pep_Cls_AMP(nn.Module):
  def __init__(self, config, vocab_size):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = config.parameterization == 'ar'

    self.vocab_embed_reg = EmbeddingLayer(config.classifier_model.hidden_size, vocab_size)
    self.vocab_embed_cls = EmbeddingLayer(config.classifier_model.hidden_size, vocab_size)

    if self.causal:
      self.sigma_map = None
    else:
      self.sigma_map_reg = TimestepEmbedder(config.classifier_model.cond_dim)
      self.sigma_map_cls = TimestepEmbedder(config.classifier_model.cond_dim)

    self.rotary_emb_reg = Rotary(config.classifier_model.hidden_size // config.classifier_model.n_heads)
    self.rotary_emb_cls = Rotary(config.classifier_model.hidden_size // config.classifier_model.n_heads)

    lora_r_other = 64
    lora_config_co_cross = LoraConfig(
      r=lora_r_other,
      lora_alpha=32,
      target_modules=["mol_to_genome_dim", "key_value_projection", "mha.out_proj", "ffn.0", "ffn.2"],
      # 也可以包含 "dense" 等其他线性层, 看你想插在哪些层
      task_type=TaskType.FEATURE_EXTRACTION,  # 不走任何特定任务逻辑，最通用的方式
      lora_dropout=0.1,
      bias="none"
    )

    lora_r_other = 64
    lora_config_reg = LoraConfig(
      r=lora_r_other,
      lora_alpha=32,
      target_modules=['dense_1', 'dense_2', "out_proj"],  # 也可以包含 "dense" 等其他线性层, 看你想插在哪些层
      task_type=TaskType.FEATURE_EXTRACTION,  # 不走任何特定任务逻辑，最通用的方式
      lora_dropout=0.1,
      bias="none"
    )

    # 构建回归用的
    blocks_reg = []
    use_adaLN = config.parameterization != 'ar'
    for _ in range(config.classifier_model.n_blocks):
      blocks_reg.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  causal=self.causal,
                  use_adaLN=use_adaLN))
    self.blocks_reg = nn.ModuleList(blocks_reg)

    blocks_cls = []
    use_adaLN = config.parameterization != 'ar'
    for _ in range(config.classifier_model.n_blocks):
      blocks_cls.append(
        DDiTBlock(config.classifier_model.hidden_size,
                  config.classifier_model.n_heads,
                  config.classifier_model.cond_dim,
                  dropout=config.classifier_model.dropout,
                  causal=self.causal,
                  use_adaLN=use_adaLN))
    self.blocks_cls = nn.ModuleList(blocks_cls)

    self.co_cross_attn_genome = antibiotic_classifier.FirstTokenAttention_genome(config.classifier_model.hidden_size,
                                                                                 8192,
                                                                                 4,
                                                                                 0.1)
    self.co_cross_attn_genome = get_peft_model(self.co_cross_attn_genome, lora_config_co_cross)
    self.co_cross_attn_text = antibiotic_classifier.FirstTokenAttention_genome(config.classifier_model.hidden_size,
                                                                               4096,
                                                                               4,
                                                                               0.1)
    self.co_cross_attn_text = get_peft_model(self.co_cross_attn_text, lora_config_co_cross)
    self.reg_head = antibiotic_classifier.RegressionHead((8192+4096)*2, (8192+4096)//4, 128, 1, 0.2)
    self.learnable_embedding_weight = nn.Parameter(torch.randn(1, 8192))

    self.ClsHead = ClsHead(input_dim=768, num_targets=1)

    self.scale_by_sigma = config.classifier_model.scale_by_sigma

    self.ATCC_genome_emb_dict, self.ATCC_text_emb_dict, self.text_only_emb_dict, self.strain_cond = self.load_genome_test_embedding()

    self.synergy_mol_emb_dict = torch.load(self.config.sampling.synergy_mol_emb_dict_path)
    self.synergy_mol_emb = self.synergy_mol_emb_dict[self.config.sampling.synergy_mol_name] # TODO: 弄到这里了

    model_name = "ibm-research/materials.selfies-ted"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    #
    # self.synergy_mol_input_ids = tokenizer(self.synergy_mol_SMILES, return_tensors='pt')['input_ids']

    # self.pooling = getattr(config.classifier_model, 'pooling', 'mean')
    # self.output_layer = nn.Linear(
    #   config.classifier_model.hidden_size,
    #   config.classifier_model.num_classes)

  def load_pretrained_weight(self):
    """
    在使用 forward 之前进行，load 自己要用的权重
    """

    # 加载 regressor 的权重
    regressor_ckpt_path = self.config.guidance.syn_classifier_checkpoint_path
    checkpoint_reg = torch.load(regressor_ckpt_path, map_location='cuda')
    new_sd = OrderedDict()
    for k, v in checkpoint_reg['mdlm_model_state_dict'].items():
      if k.startswith('backbone.'):
        new_key = k[len('backbone.'):]
        if 'blocks' in new_key:
          new_key = new_key.replace('blocks', 'blocks_reg')
        elif 'vocab_embed' in new_key:
          new_key = new_key.replace('vocab_embed', 'vocab_embed_reg')
        elif 'sigma_map' in new_key:
          new_key = new_key.replace('sigma_map', 'sigma_map_reg')
        elif 'rotary_emb' in new_key:
          new_key = new_key.replace('rotary_emb', 'rotary_emb_reg')
      else:
        new_key = k
      new_sd[new_key] = v
    self.load_state_dict(new_sd, strict=False)

    # 加载 classifier 的权重
    classifier_ckpt_path = self.config.guidance.pep_classifier_checkpoint_path
    checkpoint_cls = torch.load(classifier_ckpt_path, map_location='cuda')
    new_sd = OrderedDict()
    for k, v in checkpoint_cls['state_dict'].items():
      if k.startswith('backbone.backbone.'):
        new_key = k[len('backbone.backbone.'):]
        if 'blocks' in new_key:
          new_key = new_key.replace('blocks', 'blocks_cls')
        elif 'vocab_embed' in new_key:
          new_key = new_key.replace('vocab_embed', 'vocab_embed_cls')
        elif 'sigma_map' in new_key:
          new_key = new_key.replace('sigma_map', 'sigma_map_cls')
        elif 'rotary_emb' in new_key:
          new_key = new_key.replace('rotary_emb', 'rotary_emb_cls')
      else:
        new_key = k  # 这个地方应该已经把 ClsHead 包进去了
      new_sd[new_key] = v
    self.load_state_dict(new_sd, strict=False)

    self.reg_head.load_state_dict(checkpoint_reg['re_head_state_dict'])  # 这里 reg_head 实际上还是用来做分类的
    self.co_cross_attn_genome.load_state_dict(checkpoint_reg['co_cross_attn_genome'])
    self.co_cross_attn_text.load_state_dict(checkpoint_reg['co_cross_attn_text'])
    self.learnable_embedding_weight = checkpoint_reg['learnable_embedding_weight']  # 这个只需要一个，peptide cls 只需要用到 mdlm_model 就行了

    # self.ClsHead.load_state_dict(checkpoint_cls['ClsHead'])


  def load_genome_test_embedding(self):
    ATCC_genome_emb_dict = antibiotic_classifier.load_all_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.ATCC_genome_emb),
                                                                            1e14,
                                                                            'cpu',
                                                                            'ATCC genome embedding')
    ATCC_text_emb_dict = antibiotic_classifier.load_all_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.ATCC_text_emb),
                                                                            1,
                                                                            'cpu',
                                                                            'ATCC text embedding')
    text_only_emb_dict = antibiotic_classifier.load_text_wo_genome_embeddings(Path(self.config.sampling.genome_test_emb_dir_path.only_text_emb),
                                                                            1,
                                                                            'cpu',
                                                                            'text only embedding')
    strain_cond = str(self.config.sampling.strain)
    return ATCC_genome_emb_dict, ATCC_text_emb_dict, text_only_emb_dict, strain_cond


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def comp_reg_log(self, target, pred, step = None):
    # target = torch.tensor(target)[None, ...].expand(pred.shape[0], -1).to('cuda')
    target = self.config.sampling.target_MIC_max - (self.config.sampling.target_MIC_max - target) / self.config.sampling.steps * step
    gaussian_sigma = self.config.sampling.gaussian_sigma_max - (self.config.sampling.gaussian_sigma_max - self.config.sampling.gaussian_sigma_min) / self.config.sampling.steps * step

    print(f'\n gaussian_sigma: {gaussian_sigma}')
    print(f'\n target: {target}')

    target = torch.tensor(target)
    target_trans = -torch.log10(target / 10)[None, ...].expand(pred.shape[0], -1).to('cuda')
    # pred = 10**(-pred)*10
    sigma = torch.ones_like(target).to('cuda') * gaussian_sigma
    dist = Normal(target_trans, sigma)
    cdf_pred = dist.cdf(pred)
    mask = cdf_pred > 0.5
    cdf_pred[mask] = 1 - cdf_pred[mask]
    log_pdf = torch.log(cdf_pred * 2 + 1e-9)

    # log_pdf = torch.log(F.sigmoid(-torch.exp(torch.abs(target_trans - pred))))

    # log_pdf = dist.log_prob(pred)
    return log_pdf

  def forward(self, indices_or_one_hots, sigma, x_emb=None, attention_mask=None, step=None):
    # step = x_emb
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed_reg(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float), self.vocab_embed_reg.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map_reg(sigma))

      rotary_cos_sin = self.rotary_emb_reg(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks_reg)):
          x = self.blocks_reg[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      x = x_emb

    mol_cls_embedding = x[:, 0, :] # cls 生成分子的 embedding

    # x = self.vocab_embed_reg(self.synergy_mol_input_ids)
    mol_cls_embedding_2 = self.synergy_mol_emb.expand(mol_cls_embedding.shape).to('cuda')


    # 加载 genome embedding 和 text embedding
    if self.strain_cond in self.ATCC_genome_emb_dict.keys():
      padded_genome_embeddings = self.ATCC_genome_emb_dict[self.strain_cond].to('cuda')
      padded_text_embeddings = self.ATCC_text_emb_dict[self.strain_cond].to('cuda')
      padded_genome_embeddings = padded_genome_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
      genome_attn_masks = torch.ones(padded_genome_embeddings.shape[0], padded_genome_embeddings.shape[1]).to('cuda')
    else:
      padded_genome_embeddings = None
      padded_text_embeddings = self.text_only_emb_dict[self.strain_cond].to('cuda')
    padded_text_embeddings = padded_text_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
    text_attn_masks = torch.ones(padded_text_embeddings.shape[0], padded_text_embeddings.shape[1]).to('cuda')

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if padded_genome_embeddings is not None:
        mol_cls_embedding_genome_1 = self.co_cross_attn_genome(mol_cls_emb = mol_cls_embedding, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
        mol_cls_embedding_genome_2 = self.co_cross_attn_genome(mol_cls_emb = mol_cls_embedding_2, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
      else:
        padded_genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
        genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1)
        mol_cls_embedding_genome_1 = self.co_cross_attn_genome(mol_cls_emb = mol_cls_embedding, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
        mol_cls_embedding_genome_2 = self.co_cross_attn_genome(mol_cls_emb = mol_cls_embedding_2, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
      mol_cls_embedding_text_1 = self.co_cross_attn_text(mol_cls_emb = mol_cls_embedding, genome_embs = padded_text_embeddings, key_padding_mask = 1 - text_attn_masks)
      mol_cls_embedding_text_2 = self.co_cross_attn_text(mol_cls_emb = mol_cls_embedding_2, genome_embs = padded_text_embeddings, key_padding_mask = 1 - text_attn_masks)
      mol_cls_embedding_1 = torch.cat((mol_cls_embedding_genome_1.reshape(-1, 8192), mol_cls_embedding_text_1.reshape(-1, 4096)), dim=1)
      mol_cls_embedding_2 = torch.cat((mol_cls_embedding_genome_2.reshape(-1, 8192), mol_cls_embedding_text_2.reshape(-1, 4096)), dim=1)

      FICI_input_1 = torch.cat((mol_cls_embedding_1, mol_cls_embedding_2), dim=1)
      FICI_input_2 = torch.cat((mol_cls_embedding_2, mol_cls_embedding_1), dim=1)
      logits_1 = self.reg_head(FICI_input_1)
      logits_2 = self.reg_head(FICI_input_2)
      logits = (logits_1 + logits_2) / 2
      # reg_logits = self.reg_head(mol_cls_embedding)

    print(f' Synergy probability: {torch.mean(torch.sigmoid(logits))}')

    # reg_log_prob = self.comp_reg_log(self.config.sampling.target_MIC, reg_logits, step)
    reg_log_prob = torch.log(torch.sigmoid(logits))


    # 计算 pep_SM_cls 分类概率
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed_cls(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float), self.vocab_embed_cls.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map_cls(sigma))

      rotary_cos_sin = self.rotary_emb_cls(x)

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for i in range(len(self.blocks_cls)):
          x = self.blocks_cls[i](x, rotary_cos_sin, c, seqlens=None)
    else:
      x = x_emb

    mol_cls_embedding = x[:, 0, :] # cls

    cls_log_prob = torch.log(torch.sigmoid(self.ClsHead(mol_cls_embedding)))

    # print(f' reg_log_prob: {reg_log_prob}')

    return self.config.guidance.reg_guide_weight * reg_log_prob + self.config.guidance.cls_guide_weight * cls_log_prob # reg_log_prob

    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #   if self.pooling == 'mean':
    #     x = x.mean(dim=1)
    #   elif self.pooling == 'max':
    #     x = x.max(dim=1)
    #   elif self.pooling == 'cls':
    #     x = x[..., 0]
    #   elif self.pooling == 'last':
    #     x = x[..., -1]
    #   elif self.pooling == 'no_pooling':  # for ar_fudge
    #     pass
    #   elif self.pooling == 'attention_mean':  # for ar_pplm
    #     masked_x = x * attention_mask.unsqueeze(2)
    #     x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
    #   else:
    #     raise NotImplementedError(
    #       f"`{self.pooling}` method not implemented.")
    #   x = self.output_layer(x)
    # return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks


class ClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            hidden_dim_1=384,
            hidden_dim_2=128,
            num_targets=19,
            pooler_dropout: float = 0.2,
    ):
      """
      Initialize the classification head.

      :param input_dim: Dimension of input features.
      :param inner_dim: Dimension of the inner layer.
      :param num_classes: Number of classes for classification.
      :param activation_fn: Activation function name.
      :param pooler_dropout: Dropout rate for the pooling layer.
      """
      super().__init__()
      self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
      self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
      self.activation_fn = nn.GELU()
      self.dropout = nn.Dropout(p=pooler_dropout)
      self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features, **kwargs):
      """
      Forward pass for the classification head.

      :param features: Input features for classification.

      :return: Output from the classification head.
      """
      x = self.dense_1(features)
      x = self.activation_fn(x)
      x = self.dropout(x)

      x = self.dense_2(x)
      x = self.activation_fn(x)
      x = self.dropout(x)

      x = self.out_proj(x)
      # x = torch.sigmoid(x)
      return x
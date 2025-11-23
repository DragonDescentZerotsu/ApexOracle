import math
from functools import partial
from typing import Optional, Tuple, Union

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_conv1d import (
  causal_conv1d_fn,
  causal_conv1d_update,
)
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import (
  mamba_inner_fn,
  selective_scan_fn,
)
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
  BaseModelOutputWithNoAttention,
  MaskedLMOutput,
)

try:
  # Legacy mamba-ssm v1 file structure
  from mamba_ssm.ops.triton.layernorm import (
    RMSNorm, layer_norm_fn, rms_norm_fn
  )
except ImportError:
  try:
    # mamba-ssm v2 file structure
    from mamba_ssm.ops.triton.layer_norm import (
      RMSNorm, layer_norm_fn, rms_norm_fn
    )
  except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from mamba_ssm.ops.triton.selective_state_update import (
  selective_state_update,
)
from mamba_ssm.utils.generation import InferenceParams

from models.dit import (
  LabelEmbedder,
  TimestepEmbedder,
  bias_dropout_add_scale_fused_inference,
  bias_dropout_add_scale_fused_train,
  modulate_fused,
)

class Mamba(nn.Module):
  def __init__(
    self,
    d_model,
    d_state=16,
    d_conv=4,
    expand=2,
    dt_rank='auto',
    dt_min=0.001,
    dt_max=0.1,
    dt_init='random',
    dt_scale=1.0,
    dt_init_floor=1e-4,
    conv_bias=True,
    bias=False,
    use_fast_path=True,  # Fused kernel options
    layer_idx=None,
    device=None,
    dtype=None,
  ):
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.d_model = d_model
    self.d_state = d_state
    self.d_conv = d_conv
    self.expand = expand
    self.d_inner = int(self.expand * self.d_model)
    self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank
    self.use_fast_path = use_fast_path
    self.layer_idx = layer_idx

    self.in_proj = nn.Linear(
      self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
    )

    self.conv1d = nn.Conv1d(
      in_channels=self.d_inner,
      out_channels=self.d_inner,
      bias=conv_bias,
      kernel_size=d_conv,
      groups=self.d_inner,
      padding=d_conv - 1,
      **factory_kwargs,
    )

    self.activation = 'silu'
    self.act = nn.SiLU()

    self.x_proj = nn.Linear(
      self.d_inner, self.dt_rank + self.d_state * 2,
      bias=False, **factory_kwargs)
    self.dt_proj = nn.Linear(
      self.dt_rank, self.d_inner,
      bias=True, **factory_kwargs)

    # Initialize special dt projection to preserve variance at initialization
    dt_init_std = self.dt_rank**-0.5 * dt_scale
    if dt_init == 'constant':
      nn.init.constant_(self.dt_proj.weight, dt_init_std)
    elif dt_init == 'random':
      nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
    else:
      raise NotImplementedError

    # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
    dt = torch.exp(
      torch.rand(self.d_inner, **factory_kwargs)
      * (math.log(dt_max) - math.log(dt_min))
      + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
      self.dt_proj.bias.copy_(inv_dt)
    # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
    self.dt_proj.bias._no_reinit = True

    # S4D real initialization
    A = repeat(
      torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
      'n -> d n',
      d=self.d_inner,
    ).contiguous()
    A_log = torch.log(A)  # Keep A_log in fp32
    self.A_log = nn.Parameter(A_log)
    self.A_log._no_weight_decay = True

    # D 'skip' parameter
    self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
    self.D._no_weight_decay = True

    self.out_proj = nn.Linear(
      self.d_inner, self.d_model, bias=bias, **factory_kwargs
    )

  def forward(self, hidden_states, inference_params=None):
    """
    hidden_states: (B, L, D)
    Returns: same shape as hidden_states
    """
    batch, seqlen, dim = hidden_states.shape

    conv_state, ssm_state = None, None
    if inference_params is not None:
      conv_state, ssm_state = self._get_states_from_cache(
        inference_params, batch)
      if inference_params.seqlen_offset > 0:
        # The states are updated inplace
        out, _, _ = self.step(
          hidden_states, conv_state, ssm_state)
        return out

    # We do matmul and transpose BLH -> HBL at the same time
    xz = rearrange(
      self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'),
      'd (b l) -> b d l',
      l=seqlen,
    )
    if self.in_proj.bias is not None:
      xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), 'd -> d 1')

    A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
    # In the backward pass we write dx and dz next to each other to avoid torch.cat

    if (
      self.use_fast_path
      and causal_conv1d_fn is not None
      and inference_params is None
    ):  # Doesn't support outputting the states
      out = mamba_inner_fn(
        xz,
        self.conv1d.weight,
        self.conv1d.bias,
        self.x_proj.weight,
        self.dt_proj.weight,
        self.out_proj.weight,
        self.out_proj.bias,
        A,
        None,  # input-dependent B
        None,  # input-dependent C
        self.D.float(),
        delta_bias=self.dt_proj.bias.float(),
        delta_softplus=True,
      )

    else:
      x, z = xz.chunk(2, dim=1)
      # Compute short convolution
      if conv_state is not None:
        # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
        # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
        conv_state.copy_(
            F.pad(x, (self.d_conv - x.shape[-1], 0))
        )  # Update state (B D W)
      if causal_conv1d_fn is None:
        x = self.act(self.conv1d(x)[..., :seqlen])
      else:
        assert self.activation in ['silu', 'swish']
        x = causal_conv1d_fn(
          x=x,
          weight=rearrange(self.conv1d.weight, 'd 1 w -> d w'),
          bias=self.conv1d.bias,
          activation=self.activation,
          state=conv_state,)

      # We're careful here about the layout, to avoid extra transposes.
      # We want dt to have d as the slowest moving dimension
      # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
      x_dbl = self.x_proj(rearrange(x, 'b d l -> (b l) d'))  # (bl d)
      dt, B, C = torch.split(
        x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
      )
      dt = self.dt_proj.weight @ dt.t()
      dt = rearrange(dt, 'd (b l) -> b d l', l=seqlen)
      B = rearrange(B, '(b l) dstate -> b dstate l', l=seqlen).contiguous()
      C = rearrange(C, '(b l) dstate -> b dstate l', l=seqlen).contiguous()

      assert self.activation in ['silu', 'swish']

      y = selective_scan_fn(
        x,
        dt,
        A,
        B,
        C,
        self.D.float(),
        z=z,
        delta_bias=self.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=ssm_state is not None,
      )

      if ssm_state is not None:
        y, last_state = y
        ssm_state.copy_(last_state)
      y = rearrange(y, 'b d l -> b l d')

      out = self.out_proj(y)
    return out

  def step(self, hidden_states, conv_state, ssm_state):
    dtype = hidden_states.dtype
    assert (
      hidden_states.shape[1] == 1
    ), 'Only support decoding with 1 token at a time for now'
    xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
    x, z = xz.chunk(2, dim=-1)  # (B D)

    # Conv step
    if causal_conv1d_update is None:
      conv_state.copy_(
        torch.roll(conv_state, shifts=-1, dims=-1)
      )  # Update state (B D W)
      conv_state[:, :, -1] = x
      x = torch.sum(
        conv_state * rearrange(self.conv1d.weight, 'd 1 w -> d w'), dim=-1
      )  # (B D)
      if self.conv1d.bias is not None:
        x = x + self.conv1d.bias
      x = self.act(x).to(dtype=dtype)
    else:
      x = causal_conv1d_update(
        x.to(dtype),
        conv_state.to(dtype),
        rearrange(self.conv1d.weight, 'd 1 w -> d w'),
        self.conv1d.bias,
        self.activation,
      )

    x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
    dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    # Don't add dt_bias here
    dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
    A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

    # SSM step
    if selective_state_update is None:
      # Discretize A and B
      dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
      dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A))
      dB = torch.einsum('bd,bn->bdn', dt, B)
      ssm_state.copy_(ssm_state * dA + rearrange(x, 'b d -> b d 1') * dB)
      y = torch.einsum('bdn,bn->bd', ssm_state.to(dtype), C)
      y = y + self.D.to(dtype) * x
      y = y * self.act(z)  # (B D)
    else:
      y = selective_state_update(
        ssm_state,
        x,
        dt,
        A,
        B,
        C,
        self.D,
        z=z,
        dt_bias=self.dt_proj.bias,
        dt_softplus=True,
      )

    out = self.out_proj(y)
    return out.unsqueeze(1), conv_state, ssm_state

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    device = self.out_proj.weight.device
    conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
    conv_state = torch.zeros(
      batch_size,
      self.d_model * self.expand,
      self.d_conv,
      device=device,
      dtype=conv_dtype,
    )
    ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
    # ssm_dtype = torch.float32
    ssm_state = torch.zeros(
      batch_size,
      self.d_model * self.expand,
      self.d_state,
      device=device,
      dtype=ssm_dtype,
    )
    return conv_state, ssm_state

  def _get_states_from_cache(
    self, inference_params, batch_size, initialize_states=False
  ):
    assert self.layer_idx is not None
    if self.layer_idx not in inference_params.key_value_memory_dict:
      batch_shape = (batch_size,)
      conv_state = torch.zeros(
        batch_size,
        self.d_model * self.expand,
        self.d_conv,
        device=self.conv1d.weight.device,
        dtype=self.conv1d.weight.dtype,
      )
      ssm_state = torch.zeros(
        batch_size,
        self.d_model * self.expand,
        self.d_state,
        device=self.dt_proj.weight.device,
        dtype=self.dt_proj.weight.dtype,
        # dtype=torch.float32,
      )
      inference_params.key_value_memory_dict[self.layer_idx] = (
        conv_state,
        ssm_state,
      )
    else:
      conv_state, ssm_state = inference_params.key_value_memory_dict[
        self.layer_idx
      ]
      # TODO: What if batch size changes between generation, and we reuse the same states?
      if initialize_states:
        conv_state.zero_()
        ssm_state.zero_()
    return conv_state, ssm_state


class Block(nn.Module):
  def __init__(
    self,
    dim,
    mixer_cls,
    norm_cls=nn.LayerNorm,
    fused_add_norm=False,
    residual_in_fp32=False,
    use_adaLN=False,
    cond_dim=0,
  ):
    """
    Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection'

    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    [Ref: https://arxiv.org/abs/2002.04745]
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    """
    super().__init__()
    self.residual_in_fp32 = residual_in_fp32
    self.fused_add_norm = fused_add_norm
    self.mixer = mixer_cls(dim)
    self.norm = norm_cls(dim)

    if self.fused_add_norm:
      assert RMSNorm is not None, 'RMSNorm import fails'
      assert isinstance(
          self.norm, (nn.LayerNorm, RMSNorm)
      ), 'Only LayerNorm and RMSNorm are supported for fused_add_norm'

    self.dropout = 0.1

    self.use_adaLN = use_adaLN
    self.cond_dim = cond_dim
    if use_adaLN:
      self.adaLN_modulation = nn.Linear(
        cond_dim, 3 * dim, bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

  def forward(
    self,
    hidden_states: Tensor,
    residual: Optional[Tensor] = None,
    cond_embeds: Optional[Tensor] = None,
    inference_params: Optional[InferenceParams] = None,
  ):
    r"""Pass the input through the encoder layer.

    Args:
      hidden_states: the sequence to the encoder layer (required).
      residual: hidden_states = Mixer(LN(residual))
      cond_embeds: conditional embeddings for modulation (optional).
      inference_params: parameters for inference (optional).
    """
    if not self.fused_add_norm:
      residual = (
        (hidden_states + residual)
        if residual is not None
        else hidden_states
      )

      hidden_states = self.norm(
        residual.to(dtype=self.norm.weight.dtype))
      if self.residual_in_fp32:
        residual = residual.to(torch.float32)
    else:
      fused_add_norm_fn = (
        rms_norm_fn
        if isinstance(self.norm, RMSNorm)
        else layer_norm_fn
      )

      hidden_states, residual = fused_add_norm_fn(
        hidden_states,
        self.norm.weight,
        self.norm.bias,
        residual=residual,
        prenorm=True,
        residual_in_fp32=self.residual_in_fp32,
        eps=self.norm.eps)

    if self.use_adaLN and cond_embeds is not None:
      (shift_msa,
       scale_msa,
       gate_msa) = self.adaLN_modulation(
          cond_embeds)[:, None].chunk(3, dim=-1)
      hidden_states = modulate_fused(hidden_states,
                                     shift_msa,
                                     scale_msa)
    else:
      gate_msa = None

    mixer_out = self.mixer(hidden_states, inference_params=inference_params)

    hidden_states = mixer_out
    if self.use_adaLN and cond_embeds is not None:
      bias_dropout_scale_fn = self._get_bias_dropout_scale()
      hidden_states = bias_dropout_scale_fn(
        hidden_states,
        None,
        gate_msa,
        residual,
        self.dropout)

    return hidden_states, residual

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    return self.mixer.allocate_inference_cache(
      batch_size, max_seqlen, dtype=dtype, **kwargs)


class BiMambaConfig(PretrainedConfig):
  """Config that extends the original MambaConfig with params relevant to bi-directionality."""

  model_type = 'bimamba'

  def __init__(
    self,
    # From original MambaConfig
    d_model: int = 2560,
    n_layer: int = 64,
    vocab_size: int = 50277,
    ssm_cfg: Optional[dict] = None,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    fused_add_norm: bool = True,
    pad_vocab_size_multiple: int = 8,
    tie_word_embeddings: bool = True,
    # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
    norm_epsilon: float = 1e-5,
    # Used in init_weights
    initializer_cfg: Optional[dict] = None,
    # Caduceus-specific params
    bidirectional: bool = True,
    bidirectional_strategy: Union[str, None] = 'add',
    bidirectional_weight_tie: bool = True,
    use_adaLN: bool = True,
    cond_dim: int = 128,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.n_layer = n_layer
    self.vocab_size = vocab_size
    self.ssm_cfg = ssm_cfg
    self.rms_norm = rms_norm
    self.residual_in_fp32 = residual_in_fp32
    self.fused_add_norm = fused_add_norm
    self.pad_vocab_size_multiple = pad_vocab_size_multiple
    self.tie_word_embeddings = tie_word_embeddings
    self.norm_epsilon = norm_epsilon
    self.initializer_cfg = initializer_cfg
    self.bidirectional = bidirectional
    self.bidirectional_strategy = bidirectional_strategy
    self.bidirectional_weight_tie = bidirectional_weight_tie
    self.use_adaLN = use_adaLN
    self.cond_dim = cond_dim

def create_block(
  d_model,
  ssm_cfg=None,
  norm_epsilon=1e-5,
  rms_norm=False,
  residual_in_fp32=False,
  fused_add_norm=False,
  layer_idx=None,
  bidirectional=True,
  bidirectional_strategy='add',
  bidirectional_weight_tie=True,
  device=None,
  dtype=None,
  use_adaLN=False,
  cond_dim=0,
):
  """Create BiMamba block.

  Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
  """
  if ssm_cfg is None:
    ssm_cfg = {}
  factory_kwargs = {'device': device, 'dtype': dtype}
  bidirectional_kwargs = {
    'bidirectional': bidirectional,
    'bidirectional_strategy': bidirectional_strategy,
    'bidirectional_weight_tie': bidirectional_weight_tie,
  }
  mixer_cls = partial(
    BiMambaWrapper,
    layer_idx=layer_idx,
    **ssm_cfg,
    **bidirectional_kwargs,
    **factory_kwargs,
  )
  norm_cls = partial(
    nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
  )
  block_cls = Block
  block = block_cls(
    d_model,
    mixer_cls,
    norm_cls=norm_cls,
    fused_add_norm=fused_add_norm,
    residual_in_fp32=residual_in_fp32,
    use_adaLN=use_adaLN,
    cond_dim=cond_dim,
  )
  block.layer_idx = layer_idx

  return block


class BiMambaWrapper(nn.Module):
  """Thin wrapper around Mamba to support bi-directionality."""

  def __init__(
    self,
    d_model: int,
    bidirectional: bool = True,
    bidirectional_strategy: Optional[str] = 'add',
    bidirectional_weight_tie: bool = True,
    **mamba_kwargs,
  ):
    super().__init__()
    if bidirectional and bidirectional_strategy is None:
      bidirectional_strategy = 'add'  # Default strategy: `add`
    if bidirectional and bidirectional_strategy not in ['add', 'ew_multiply']:
      raise NotImplementedError(
        f'`{bidirectional_strategy}` strategy for bi-directionality is not implemented!'
      )
    self.bidirectional = bidirectional
    self.bidirectional_strategy = bidirectional_strategy

    self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)

    self.mamba_rev = None
    if bidirectional:
      self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
      if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
        self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
        self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
        self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
        self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
    else:
      self.mamba_rev = None

  def forward(self, hidden_states, inference_params=None):
    """Bidirectional-enabled forward pass

    hidden_states: (B, L, D)
    Returns: same shape as hidden_states
    """

    out = self.mamba_fwd(
    hidden_states, inference_params=inference_params,)

    if self.bidirectional:
      if inference_params is not None:
        raise NotImplementedError(
          'Passing `inference_params` not supported '
          'for bidirectional Mamba.')

      hidden_states_flipped = torch.flip(hidden_states, dims=(1,))

      out_rev = self.mamba_rev(
        hidden_states_flipped,  # Flip along the sequence length dimension
        inference_params=inference_params,)

      out_rev_flipped = torch.flip(out_rev, dims=(1,))
      if self.bidirectional_strategy == 'add':
        out = out + out_rev_flipped  # Flip back for combining with forward hidden states
      elif self.bidirectional_strategy == 'ew_multiply':
        out = out * out_rev_flipped
      else:
        raise NotImplementedError(
          f"`{self.bidirectional_strategy}` for "
          f"bi-directionality not implemented!")
    return out

  def allocate_inference_cache(
    self, batch_size, max_seqlen, dtype=None, **kwargs):
    if self.bidirectional:
      raise NotImplementedError(
        'Allocating inference cache not supported '
        'for bidirectional Mamba.')
    return self.mamba_fwd.allocate_inference_cache(
      batch_size, max_seqlen, dtype=dtype, **kwargs)


class BiMambaEmbeddings(nn.Module):
  def __init__(
      self,
      config: BiMambaConfig,
      input_dim=None,
      device=None,
      dtype=None,
  ):
    super().__init__()
    factory_kwargs = {'device': device, 'dtype': dtype}
    if input_dim is None:
        input_dim = config.vocab_size
    self.word_embeddings = nn.Embedding(
        input_dim, config.d_model, **factory_kwargs
    )

  def forward(self, input_ids):
    """
    input_ids: (batch, seqlen)
    """
    return self.word_embeddings(input_ids)


class BiMambaMixerModel(nn.Module):
  def __init__(
    self,
    config: BiMambaConfig,
    device=None,
    dtype=None,
  ) -> None:
    super().__init__()
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.config = config
    input_dim = config.vocab_size
    d_model = config.d_model

    self.fused_add_norm = config.fused_add_norm
    self.residual_in_fp32 = config.residual_in_fp32

    self.embeddings = BiMambaEmbeddings(
      config, input_dim=input_dim, **factory_kwargs)

    # Mamba changes the order of residual and layer norm:
    # Instead of LN -> Attn / MLP -> Add, we do:
    # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
    # the main branch (output of MLP / Mixer). The model definition is unchanged.
    # This is for performance reason: we can fuse add + layer_norm.
    if config.fused_add_norm:
      if layer_norm_fn is None or rms_norm_fn is None:
        raise ImportError('Failed to import Triton LayerNorm / RMSNorm kernels')

    self.layers = nn.ModuleList(
      [
        create_block(
          d_model,
          ssm_cfg=config.ssm_cfg,
          norm_epsilon=config.norm_epsilon,
          rms_norm=config.rms_norm,
          residual_in_fp32=config.residual_in_fp32,
          fused_add_norm=config.fused_add_norm,
          layer_idx=i,
          bidirectional=config.bidirectional,
          bidirectional_strategy=config.bidirectional_strategy,
          bidirectional_weight_tie=config.bidirectional_weight_tie,
          use_adaLN=config.use_adaLN,
          cond_dim=config.cond_dim,
          **factory_kwargs,
        )
        for i in range(config.n_layer)
      ]
    )

    if config.use_adaLN:
      self.adaLN_modulation_final = nn.Linear(
        config.cond_dim, 2 * d_model, bias=True)
      self.adaLN_modulation_final.weight.data.zero_()
      self.adaLN_modulation_final.bias.data.zero_()
    else:
      self.adaLN_modulation_final = None

    norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
      d_model, eps=config.norm_epsilon, **factory_kwargs)
    self.norm_f = norm_f

  def forward(
    self,
    input_ids: torch.LongTensor,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    output_hidden_states: Optional[bool] = None,
    cond_embeds: Optional[torch.Tensor] = None,
    inference_params: Optional[InferenceParams] = None
  ):

    """Mixer forward."""
    all_hidden_states = []
    if hidden_states is None:
      if inputs_embeds is not None:
        hidden_states = inputs_embeds
      else:
        if input_ids.ndim == 2:  # indices (B, L)
          hidden_states = self.embeddings(input_ids)
        else:  # one-hots (B, L, V)
          hidden_states = F.linear(
            input_ids.to(torch.float),
            self.embeddings.word_embeddings.weight.T)

      residual = None
      for ind, layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        # TODO: Add support for gradient checkpointing
        layer_out = layer(
          hidden_states, residual,
          inference_params=inference_params,
          cond_embeds=cond_embeds
        )

        hidden_states, residuals = layer_out

      if not self.fused_add_norm:
        if self.config.use_adaLN:
          raise NotImplementedError('adaln only implemented for fused_add_norm')
        residual = (
          (hidden_states + residual) if residual is not None else hidden_states
        )
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
      else:
        if cond_embeds is not None and self.config.use_adaLN:
          shift, scale = self.adaLN_modulation_final(
            cond_embeds)[:, None].chunk(2, dim=2)
        else:
          shift, scale = None, None

        fused_add_norm_fn = (
          rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        )

        # Set prenorm=False here since we don't need the residual
        hidden_states = fused_add_norm_fn(
          hidden_states,
          self.norm_f.weight,
          self.norm_f.bias,
          eps=self.norm_f.eps,
          residual=residual,
          prenorm=False,
          residual_in_fp32=self.residual_in_fp32,
        )
        if cond_embeds is not None and self.config.use_adaLN:
          hidden_states = modulate_fused(hidden_states, shift, scale)
    else:
      if cond_embeds is not None and self.config.use_adaLN:
        shift, scale = self.adaLN_modulation_final(
          cond_embeds)[:, None].chunk(2, dim=2)
        hidden_states = modulate_fused(hidden_states, shift, scale)

    if output_hidden_states:
      all_hidden_states.append(hidden_states)

    return hidden_states, all_hidden_states

  def allocate_inference_cache(
    self, batch_size, max_seqlen, dtype=None, **kwargs):
    return {
      i: layer.allocate_inference_cache(
        batch_size, max_seqlen, dtype=dtype, **kwargs)
      for i, layer in enumerate(self.layers)
    }


def cross_entropy(logits, y, ignore_index=-100):
  """Cross-entropy loss."""
  logits = logits.view(-1, logits.shape[-1])
  y = y.view(-1)
  return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
  """Weighted cross-entropy loss (discounts certain tokens)."""
  logits = logits.view(-1, logits.shape[-1])
  y = y.view(-1)
  ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction='none')
  loss_weights = loss_weights.view(-1)
  loss_weights[y == ignore_index] = 0.0
  return (ce * (loss_weights / loss_weights.sum())).sum()


class BiMambaPreTrainedModel(PreTrainedModel):
  """PreTrainedModel wrapper for BiMamba backbone."""

  config_class = BiMambaConfig
  base_model_prefix = 'bimamba'
  supports_gradient_checkpointing = False
  _no_split_modules = ['BiMambaWrapper']

  def _init_weights(
    self,
    module,
    initializer_range=0.02,  # Now only used for embedding layer.
    **kwargs,
  ):
    """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

    n_layer = self.config.n_layer
    initialized_cfg = self.config.initializer_cfg if self.config.initializer_cfg is not None else {}
    rescale_prenorm_residual = initialized_cfg.get('rescale_prenorm_residual', True)
    initializer_range = initialized_cfg.get('initializer_range', initializer_range)
    n_residuals_per_layer = initialized_cfg.get('n_residuals_per_layer', 1)

    if isinstance(module, nn.Linear):
      if module.bias is not None:
        if not getattr(module.bias, '_no_reinit', False):
          nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
      # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
      #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
      #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
      #   residual layers.
      #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
      #
      # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
      for name, p in module.named_parameters():
        if name in ['out_proj.weight', 'fc2.weight']:
          # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
          # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
          # We need to reinit p since this code could be called multiple times
          # Having just p *= scale would repeatedly scale it down
          nn.init.kaiming_uniform_(p, a=math.sqrt(5))
          with torch.no_grad():
            p /= math.sqrt(n_residuals_per_layer * n_layer)


class BiMamba(BiMambaPreTrainedModel):
  """BiMamba model that can be instantiated using HF patterns."""

  def __init__(self, config: BiMambaConfig, device=None, dtype=None, **kwargs):
    super().__init__(config)

    # Adjust vocab size if vocab padding is set.
    if config.vocab_size % config.pad_vocab_size_multiple != 0:
      config.vocab_size += config.pad_vocab_size_multiple - (
        config.vocab_size % config.pad_vocab_size_multiple
      )

    self.config = config
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.backbone = BiMambaMixerModel(config, **factory_kwargs, **kwargs)

  def forward(
    self,
    input_ids: torch.LongTensor = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cond_embeds: Optional[torch.Tensor] = None,
    inference_params: Optional[InferenceParams] = None,
  ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
    """HF-compatible forward method."""
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    backbone_out = self.backbone(
      input_ids,
      inputs_embeds=inputs_embeds,
      hidden_states=hidden_states,
      output_hidden_states=output_hidden_states,
      cond_embeds=cond_embeds,
      inference_params=inference_params,
    )

    hidden_states, all_hidden_states = backbone_out

    if return_dict:
      return BaseModelOutputWithNoAttention(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states if output_hidden_states else None,
      )
    elif output_hidden_states:
      return hidden_states, all_hidden_states
    else:
      return hidden_states


class BiMambaForMaskedLM(BiMambaPreTrainedModel):
  """HF-compatible BiMamba model for masked language modeling."""

  def __init__(self, config: BiMambaConfig, device=None, dtype=None, **kwargs):
    super().__init__(config, **kwargs)
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.bimamba = BiMamba(config, **factory_kwargs, **kwargs)
    self.config = config
    lm_head_in_dim = config.d_model
    # LM head may only take in concatenated timestep embeddings
    # if its weights are not tied to the vocab embedding
    self.lm_head = nn.Linear(
      lm_head_in_dim,
      self.config.vocab_size,  # Use BiMamba config as it might have been updated
      bias=False,
      **factory_kwargs,
    )
    # Initialize weights and apply final processing
    self.post_init()
    if self.config.tie_word_embeddings:
      self.tie_weights()

  def init_weights(self):
    """
    If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
    initialization logic in `_init_weights`.
    """

    # Initialize weights
    self.apply(self._initialize_weights)

    # Tie weights should be skipped when not initializing all weights
    # since from_pretrained(...) calls tie weights anyway.

  def post_init(self):
    """
    A method executed at the end of each Transformer model initialization, to execute code that needs the model's
    modules properly initialized (such as weight initialization).
    """
    self.init_weights()
    self._backward_compatibility_gradient_checkpointing()

  def get_input_embeddings(self):
    return self.bimamba.backbone.embeddings.word_embeddings

  def set_input_embeddings(self, value):
    self.bimamba.backbone.embeddings.word_embeddings = value

  def get_output_embeddings(self):
    return self.lm_head

  def set_output_embeddings(self, new_embeddings):
    """Overrides output embeddings."""
    self.lm_head = new_embeddings

  def tie_weights(self):
    """Tie weights."""
    super().tie_weights()

  def get_encoder(self):
    """Get encoder (backbone) for the model."""
    return self.bimamba

  def set_encoder(self, encoder):
    """Set encoder (backbone) for the model."""
    self.bimamba = encoder

  def forward(
    self,
    input_ids: torch.LongTensor = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    loss_weights: Optional[torch.FloatTensor] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cond_embeds: Optional[torch.FloatTensor] = None,
    inference_params: Optional[InferenceParams] = None,
    num_last_tokens: int = 0
  ) -> Union[Tuple, MaskedLMOutput]:
    """HF-compatible forward method."""

    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.bimamba(
      input_ids=input_ids,
      inputs_embeds=inputs_embeds,
      hidden_states=hidden_states,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
      cond_embeds=cond_embeds,
      inference_params=inference_params,
    )
    hidden_states = outputs[0]

    if num_last_tokens > 0:
      hidden_states = hidden_states[:, -num_last_tokens:]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
      if loss_weights is not None:
        loss = weighted_cross_entropy(
          logits, labels, loss_weights, ignore_index=self.config.pad_token_id
        )
      else:
        loss = cross_entropy(
          logits, labels, ignore_index=self.config.pad_token_id
        )

    if not return_dict:
      output = (logits,) + outputs[1:]
      return (loss,) + output if loss is not None else output

    return MaskedLMOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,)


class DiMamba(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int, pad_token_id: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    if config.parameterization == 'ar':
      self.sigma_map = None
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

    mamba_config = BiMambaConfig(
      d_model=config.model.hidden_size,
      n_layer=config.model.n_blocks,
      pad_token_id=pad_token_id,
      vocab_size=vocab_size,
      pad_vocab_size_multiple=1,
      tie_word_embeddings=config.model.tie_word_embeddings,
      bidirectional=getattr(config.model, 'bidirectional', True),
      bidirectional_strategy=getattr(config.model, 'bidirectional_strategy', 'add'),
      bidirectional_weight_tie=getattr(config.model, 'bidirectional_weight_tie', True),
      use_adaLN=self.sigma_map is not None or self.cond_map is not None,
      cond_dim=config.model.cond_dim,
    )

    self.model = BiMambaForMaskedLM(config=mamba_config)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(
    self,
    indices,
    sigma,
    cond=None,
    x_emb=None,
    return_hidden_states=False,
    inference_params=None
  ):
    c = None
    if self.sigma_map is not None:
      c = F.silu(self.sigma_map(sigma))
    if cond is not None:
      if self.cond_map is None:
        raise ValueError("Conditioning variable provided, "
                         "but Model was not initialized "
                         "with condition embedding layer.")
      else:
        c = c + F.silu(self.cond_map(cond)) if c is not None \
          else F.silu(self.cond_map(cond))

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      model_out = self.model(
        indices,
        hidden_states=x_emb,
        cond_embeds=c,
        output_hidden_states=return_hidden_states,
        inference_params=inference_params
      )

    if return_hidden_states:
      return model_out.logits, model_out.hidden_states
    return model_out.logits


class DiMambaClassifier(nn.Module):
  def __init__(self, config, vocab_size: int, pad_token_id: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    if config.parameterization == 'ar':
      self.sigma_map = None
    else:
      self.sigma_map = TimestepEmbedder(config.classifier_model.cond_dim)

    mamba_config = BiMambaConfig(
      d_model=config.classifier_model.hidden_size,
      n_layer=config.classifier_model.n_blocks,
      pad_token_id=pad_token_id,
      vocab_size=vocab_size,
      pad_vocab_size_multiple=1,
      tie_word_embeddings=config.classifier_model.tie_word_embeddings,
      bidirectional=getattr(config.classifier_model, 'bidirectional', True),
      bidirectional_strategy=getattr(config.classifier_model, 'bidirectional_strategy', 'add'),
      bidirectional_weight_tie=getattr(config.classifier_model, 'bidirectional_weight_tie', True),
      use_adaLN=self.sigma_map is not None,
      cond_dim=config.classifier_model.cond_dim,
    )

    self.model = BiMamba(config=mamba_config)
    self.pooling = getattr(config.classifier_model, 'pooling', 'mean')
    self.output_layer = nn.Linear(
      config.classifier_model.hidden_size,
      config.classifier_model.num_classes)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(
    self,
    indices_or_one_hots,
    sigma,
    x_emb=None,
    attention_mask=None
  ):
    c = None
    if self.sigma_map is not None:
      c = F.silu(self.sigma_map(sigma))

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      x = self.model(
        indices_or_one_hots,
        hidden_states=x_emb,
        cond_embeds=c,
        output_hidden_states=False,
        inference_params=None
      )[0]

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      if self.pooling == 'mean':
        x = x.mean(dim=1)
      elif self.pooling == 'max':
        x = x.max(dim=1)
      elif self.pooling == 'cls':
        x = x[..., 0]
      elif self.pooling == 'last':
        x = x[..., -1]
      elif self.pooling == 'no_pooling':  # used for ar_fudge
        pass
      elif self.pooling == 'attention_mean':  # used for ar_pplm
        masked_x = x * attention_mask.unsqueeze(2)
        x = torch.sum(masked_x, dim=1) / (
              torch.sum(attention_mask, dim=1,
                        keepdim=True) + 1e-15)
      else:
        raise NotImplementedError(
          f"`{self.pooling}` method not implemented.")
      x = self.output_layer(x)
    return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.sigma_map = encoder.sigma_map
    self.model = encoder.model.bimamba

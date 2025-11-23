import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import fsspec
import hydra
from hydra.utils import to_absolute_path
from flash_attn.utils import pretrained
from hydra import initialize, compose
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm

import classifier
import dataloader
import diffusion
import diffusion_mdlm
import eval_utils
import utils
import models
from collections import OrderedDict
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from pathlib import Path

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else',
  lambda condition, x, y: x if condition else y
)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')

  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False).to('cuda')


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def _train(config, logger, tokenizer, train_classifier=False):
  logger.info('Starting Training.')
  wandb_logger = None
  config['wandb'] = None  # TODO: for debugging, debug 的时候跳过 wandb 记录
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
  if not config.is_vision:
    _print_batch(train_ds, valid_ds, tokenizer)

  if train_classifier:
    # This param indicates classifier will be used for
    #   PPLM / NOS-style guidance
    #  (see: https://arxiv.org/abs/2305.20009).
    if getattr(config, 'is_pplm_classifier', False):
      pretrained_model = _load_from_checkpoint(config, tokenizer)
      if (getattr(config.classifier_model, 'use_encoder_ema', True) and pretrained_model.ema):
        pretrained_model.load_ema_params()
      pretrained_backbone = pretrained_model.backbone
      # Remove the last layer for the classifier
      if hasattr(pretrained_backbone, 'output_layer'):  #DiT
        delattr(pretrained_backbone, 'output_layer')   # 用于 embedding 提取，output_layer 直接扔了
      if hasattr(pretrained_backbone, 'model.lm_head'):  #DiMamba
        delattr(pretrained_backbone, 'model.lm_head')
      if getattr(config.classifier_model, 'freeze_encoder', True):
        for param in pretrained_backbone.parameters():
          param.requires_grad = False
    else:
      pretrained_backbone = None

    model = classifier.Classifier(
      config,
      tokenizer=valid_ds.tokenizer,
      pretrained_backbone=pretrained_backbone)
  else:
    model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

def guide_sample_AMP(config, tokenizer):
  # config_pretrained = OmegaConf.load(config.sampling.pretrain_backbone_config)
  pretrained = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')  # 加载 guidance 的 diffusion model

  pretrained_backbone_ckpt = torch.load(config.sampling.pretrained_ckpt_path, map_location='cuda')
  pretrained_state_dict = pretrained_backbone_ckpt['state_dict']
  new_sd = OrderedDict()
  for k, v in pretrained_state_dict.items():
    if k.startswith('backbone.'):
      new_key = k[len('backbone.'):]
    else:
      new_key = k
    new_sd[new_key] = v

  pretrained.backbone = models.dit.DIT(config, vocab_size=len(tokenizer.get_vocab()))
  pretrained.backbone.load_state_dict(new_sd, strict=False)
  pretrained = pretrained.to('cuda')

  # 这下面就可以大抄特抄 _gen_ppl_eval 了
  pretrained.eval()
  samples = []
  for _ in tqdm(range(config.sampling.num_sample_batches), desc='Gen. batches', leave=False):
    sample = pretrained.sample_AMP(config=config, tokenizer = tokenizer)
    sample = eval_utils.extract_valid_SELFIES(sample, tokenizer)
    # print(sample[0])
    if len(sample) > 0:
      decoded_SEFLIES = [SELFIES_str.replace('] [', '][') for SELFIES_str in pretrained.tokenizer.batch_decode(sample)]
      samples.extend(decoded_SEFLIES)

    # 可视化这些 decode 出来的分子
  if len(samples) > 0:
    if config.classifier_backbone != 'dit_synergy_cls_AMP':
      samples = eval_utils.draw_sampled_mol_fig(samples, Path(config.sampling.mol_img_save_dir), config)
    else:
      if config.guidance.method == 'cbg_antibiotic':
        samples = eval_utils.draw_sampled_mol_fig(samples, Path(config.sampling.syn_mol_img_save_dir_no_guide), config)
      else:
        samples = eval_utils.draw_sampled_mol_fig(samples, Path(config.sampling.syn_mol_img_save_dir), config)
    eval_utils.save_sampled_mols_SEFLIES(samples, config)
  # for sample in samples:
  #   print(sample)


def _gen_ppl_eval(config, tokenizer):
  pretrained = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')  # 加载 guidance 的 diffusion model

  pretrained_backbone_ckpt = torch.load(config.sampling.pretrained_ckpt_path, map_location='cuda')
  pretrained_state_dict = pretrained_backbone_ckpt['state_dict']
  new_sd = OrderedDict()
  for k, v in pretrained_state_dict.items():
    if k.startswith('backbone.'):
      new_key = k[len('backbone.'):]
    else:
      new_key = k
    new_sd[new_key] = v

  pretrained.backbone = models.dit.DIT(config, vocab_size=len(tokenizer.get_vocab()))
  pretrained.backbone.load_state_dict(new_sd, strict=False)
  pretrained = pretrained.to('cuda')

  # 这下面就可以大抄特抄 _gen_ppl_eval 了


  # pretrained = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  pretrained.eval()
  samples = []
  for _ in tqdm(range(config.sampling.num_sample_batches),
                desc='Gen. batches', leave=False):
    sample = pretrained.sample()
    samples.extend(pretrained.tokenizer.batch_decode(sample))

  # Replace CLS token with BOS token (if applicable) and
  # remove padding and mask tokens
  tok_bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.cls_token
  samples = [
    s.replace('[PAD]', '').replace('[MASK]', '').strip()
    for s in samples
  ]
  # Add BOS token to the beginning of each sample (if not already present)
  samples = [
    s if s.startswith(tok_bos_token) else f"{tok_bos_token} {s}"
    for s in samples
  ]
  del pretrained  # free up space for eval
  print(f"Generated {len(samples)} samples.")

  generative_ppl = eval_utils.compute_generative_ppl(
    samples,
    eval_model_name_or_path=config.eval.generative_ppl_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=config.model.length)
  tokens = tokenizer.batch_encode_plus(
    samples,
    return_tensors='pt',
    add_special_tokens=False,
    max_length=config.model.length,
    padding='max_length',
    truncation=True)['input_ids']
  _, counts = torch.unique(
    torch.tensor(tokens), return_counts=True, sorted=False)
  entropy = torch.special.entr(
    counts.float() / counts.sum()).sum().item()
  with open(config.eval.generated_samples_path, 'w') as f:
    json.dump({
      'generative_ppl': generative_ppl,
      'entropy': entropy,
      'generated_seqs': samples,
    },
      f, indent=4) # type: ignore
  print(f"Entropy: {entropy:0.3f}")
  print(f"Gen. PPL: {generative_ppl:0.3f}")


def _ppl_eval(config, tokenizer):
  print(f"Evaluating perplexity on {config.data.valid}.")
  pretrained = _load_from_checkpoint(
    config=config, tokenizer=tokenizer)
  pretrained.eval()
  if not config.eval.disable_ema:
    pretrained.load_ema_params()

  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  ppl = eval_utils.compute_ppl(pretrained, valid_ds)
  print(f"PPL: {ppl:0.3f}")


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  print(f'strain: {config.sampling.strain}')
  print(f'target length: {config.sampling.target_length}')

  logger = utils.get_logger(__name__)
  # tokenizer = dataloader.get_tokenizer(config)
  model_name = "ibm-research/materials.selfies-ted"
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  if config.mode == 'guide_sample':
    guide_sample_AMP(config, tokenizer)
  elif config.mode == 'gen_ppl_eval':
    _gen_ppl_eval(config, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, tokenizer)
  elif 'train' in config.mode:  # train classifier 也是从这进来的
    _train(config, logger, tokenizer, train_classifier='classifier' in config.mode)
  else:
    raise NotImplementedError(f"Mode {config.mode} not implemented.")


if __name__ == '__main__':
  main()

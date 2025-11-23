import os

import torch
import transformers
from tqdm import tqdm

import diffusion
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw

from pathlib import Path
import shutil


def compute_ppl(
    pretrained_model,
    val_ds
):
  ppl_metrics = diffusion.Perplexity().to('cuda')
  pbar = tqdm(val_ds, desc='PPL')
  for batch in pbar:
    input_ids = batch['input_ids'].to('cuda')
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask'].to('cuda')
    else:
      attention_mask = None
    losses = pretrained_model._loss(input_ids, attention_mask)
    ppl_metrics.update(losses.nlls, losses.token_mask)
    pbar.set_postfix({'ppl': ppl_metrics.compute().item()})
  return ppl_metrics.compute().item()


def compute_generative_ppl(
    sentences,
    eval_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=128):
  gen_ppl_metric = diffusion.Perplexity().to('cuda')
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
    eval_model_name_or_path)
  if eval_model_tokenizer.pad_token is None:
    eval_model_tokenizer.pad_token = \
      eval_model_tokenizer.eos_token
    eval_model_tokenizer.pad_token_id = \
      eval_model_tokenizer.eos_token_id
  eval_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name_or_path).eval()
  if max_length is None:
    max_length = max_length
  eval_model = eval_model.to('cuda')
  # Re-tokenize using eval model's tokenizer
  tokenizer_kwargs = {
    'return_tensors': 'pt',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'truncation': True,
    'padding': True,
    'max_length': max_length,
  }
  eval_context_size = 1024
  samples = eval_model_tokenizer(
    sentences, **tokenizer_kwargs)
  attn_mask = samples['attention_mask']
  samples = samples['input_ids']
  attn_mask = attn_mask.to('cuda')
  samples = samples.to('cuda')
  num_batches = samples.shape[0] // gen_ppl_eval_batch_size
  for i in tqdm(range(num_batches),
                desc='Gen. PPL', leave=False):
    _samples = torch.split(
      samples[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    _attn_mask = torch.split(
      attn_mask[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
      logits = eval_model(
        sample_chunk, attention_mask=attn_mask_chunk)[0]
      logits = logits.transpose(-1, -2)

      nlls = torch.nn.functional.cross_entropy(
        logits[..., :-1],
        sample_chunk[..., 1:],
        reduction='none')
      # first_eos = (sample_chunk == eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
      # token_mask = (sample_chunk != eval_model_tokenizer.eos_token_id)
      # gen_ppl_metric.update(
      #   nlls, first_eos[..., 1:] + token_mask[..., 1:])
      gen_ppl_metric.update(
        nlls, attn_mask_chunk[..., 1:])
  return gen_ppl_metric.compute().item()

def draw_sampled_mol_fig(samples, mol_img_save_dir: Path, config):
  valid_samples = []

  # 先清空之前存下来的
  if mol_img_save_dir.exists() and mol_img_save_dir.is_dir():
    shutil.rmtree(mol_img_save_dir)
  # 重新创建一个空文件夹
  os.makedirs(mol_img_save_dir, exist_ok=True)

  # 肽键 pattern
  amide_smarts = "[NX3][CX3](=O)[#6]"
  amide_pattern = Chem.MolFromSmarts(amide_smarts)

  fig_count = 0
  non_pep_count = 0

  for i, sample in enumerate(samples):
    if '[CLS]' in sample:
      continue

    if '[Nop]' in sample:
      continue

    if '[UNK]' in sample:
      continue

    try:
      SMILES_str = sf.decoder(sample)
    except Exception:
      continue
    mol = Chem.MolFromSmiles(SMILES_str)

    if mol is None:
      # record and skip
      continue

    # 检查有没有肽键, 没有的直接跳过：
    if config.sampling.peptide_only:
      if not mol.HasSubstructMatch(amide_pattern):
        non_pep_count += 1
        continue

    try:
      img = Draw.MolToImage(mol, size=(1500, 1000))
    except ValueError as e:
      print(f"Warning: failed drawing molecule #{i}: {e!r}")
      continue

    # img = Draw.MolToImage(mol, size=(1500, 1000))

    img.save(mol_img_save_dir/f"mol_{fig_count}.png")
    fig_count += 1
    valid_samples.append(sample)
  print(f' Generated mol figs saved to {str(mol_img_save_dir)}')
  print(f' non peptide count: {non_pep_count}')

  return valid_samples

def extract_valid_SELFIES(samples, tokenizer):
  valid_SELFIES_token_ids = []
  for sample in samples:
    idxs = torch.where(sample == tokenizer.sep_token_id)[0]  # 找到停止的地方
    first_idx = idxs[0].item() if idxs.numel() > 0 else None
    if first_idx is not None and sample[0] == tokenizer.cls_token_id:
      if tokenizer.mask_token_id in sample[1:first_idx] or tokenizer.pad_token_id in sample[1:first_idx]:
        continue
      valid_SELFIES_token_ids.append(sample[1:first_idx])
    else:
      print(f' invalid SELFIES: {sample}')

  print(f' len valid SELFIES: {len(valid_SELFIES_token_ids)}\n len all SELFIES: {len(samples)}')
  return valid_SELFIES_token_ids

def save_sampled_mols_SEFLIES(samples, config):
  guidance_method = "noise" if config.guidance.noise else "clean"
  if config.classifier_backbone != 'dit_synergy_cls_AMP':
    save_path = Path(config.sampling.mol_SELFIES_save_dir)/f"strain_{config.sampling.strain}_MIC_{config.sampling.target_MIC}_length_{config.sampling.target_length}_{guidance_method}.txt"
  else:
    if config.guidance.method == 'cbg_antibiotic':
      save_path = Path(config.sampling.mol_SELFIES_save_dir) / f"strain_{config.sampling.strain}_synoguide_{config.sampling.synergy_mol_name}_length_{config.sampling.target_length}_{guidance_method}.txt"
    else:
      save_path = Path(config.sampling.mol_SELFIES_save_dir) / f"strain_{config.sampling.strain}_syn_{config.sampling.synergy_mol_name}_length_{config.sampling.target_length}_{guidance_method}.txt"

  with open(save_path, 'w') as f:
    for line in samples:
      # 写入字符串，并在末尾加上换行符
      f.write(line + "\n")


  print(f' Generated mol selfies saved to {str(save_path)}')
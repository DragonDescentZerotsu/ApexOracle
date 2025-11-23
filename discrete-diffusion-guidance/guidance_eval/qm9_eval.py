import json
import os
import time
import typing

import datasets
import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rdkit
import rich.syntax
import rich.tree
import torch
from rdkit import Chem as rdChem
from rdkit.Chem import QED
from tqdm.auto import tqdm

import dataloader
import diffusion

rdkit.rdBase.DisableLog('rdApp.error')

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


def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.

  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style,
                        guide_style=style)

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


def get_mol_property_fn(
    prop: str
) -> typing.Callable[[rdChem.Mol], typing.Union[int, float]]:
  if prop == 'qed':
    return QED.qed
  if prop == 'ring_count':
    return lambda x_mol: len(rdChem.GetSymmSSSR(x_mol))
  raise NotImplementedError(
    f"Property function for {prop} not implemented")


@hydra.main(version_base=None, config_path='../configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  # Reproducibility
  L.seed_everything(config.seed)
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False

  _print_config(config, resolve=True)
  print(f"Checkpoint: {config.eval.checkpoint_path}")

  qm9_dataset = datasets.load_dataset(
    'yairschiff/qm9', trust_remote_code=True,
    split='train')
  tokenizer = dataloader.get_tokenizer(config)
  pretrained = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False)
  pretrained.eval()
  label_col = config.data.label_col
  pctile_threshold = config.data.label_col_pctile
  pctile_threshold_value = np.percentile(
    qm9_dataset[label_col], q=pctile_threshold)
  above_threshold = np.array(qm9_dataset[label_col])[
    qm9_dataset[label_col] >= pctile_threshold_value]
  below_threshold = np.array(qm9_dataset[label_col])[
    qm9_dataset[label_col] < pctile_threshold_value]
  result_dicts = []
  mol_property_fn = get_mol_property_fn(label_col)

  print(
    f"All          - {label_col.upper()} Mean: {np.mean(qm9_dataset[label_col]):0.3f}, {label_col.upper()} Median: {np.median(qm9_dataset[label_col]):0.3f}")
  print(
    f"Below {pctile_threshold}%ile - {label_col.upper()} Mean: {np.mean(below_threshold):0.3f}, {label_col.upper()} Median: {np.median(below_threshold):0.3f}")
  print(
    f"Above {pctile_threshold}%ile - {label_col.upper()} Mean: {np.mean(above_threshold):0.3f}, {label_col.upper()} Median: {np.median(above_threshold):0.3f}")
  result_dicts.append({
    'Seed': -1,
    'T': -1,
    'Num Samples': len(qm9_dataset),
    'Valid': 1.0,
    'Unique': 1.0,
    'Novel': 1.0,
    f'{label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
    f'{label_col.upper()} 25%ile': np.percentile(qm9_dataset[label_col], q=25),
    f'{label_col.upper()} Median': np.median(qm9_dataset[label_col]),
    f'{label_col.upper()} 75%ile': np.percentile(qm9_dataset[label_col], q=75),
    f'Novel {label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
    f'Novel {label_col.upper()} 25%ile': np.percentile(qm9_dataset[label_col], q=25),
    f'Novel {label_col.upper()} Median': np.median(qm9_dataset[label_col]),
    f'Novel {label_col.upper()} 75%ile': np.percentile(qm9_dataset[label_col], q=75),
  } | {k.capitalize(): -1 for k, v in config.guidance.items()})

  samples = []
  for _ in tqdm(
      range(config.sampling.num_sample_batches),
      desc='Gen. batches', leave=False):
    start = time.time()
    sample = pretrained.sample()
    # print(f"Batch took {time.time() - start:.2f} seconds.")
    samples.extend(
      pretrained.tokenizer.batch_decode(sample))
  invalids = []
  valids = []
  mol_property = []
  for t in samples:
    t = t.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')
    try:
      mol = rdChem.MolFromSmiles(t)
      if mol is None or len(t) == 0:
        invalids.append(t)
      else:
        valids.append(t)
        mol_property.append(mol_property_fn(mol))
    except rdkit.Chem.rdchem.KekulizeException as e:
      print(e)
      invalids.append(t)
  valid = len(valids)
  valid_pct = len(valids) / len(samples)
  unique = len(set(valids))
  novel = len(set(valids) - set(qm9_dataset['canonical_smiles']))
  try:
    unique_pct = unique / valid
    novel_pct = novel / valid
  except ZeroDivisionError:
    unique_pct, novel_pct = 0., 0.
  mol_property_novel = [
    mol_property_fn(rdChem.MolFromSmiles(s))
    for s in set(valids) - set(qm9_dataset['canonical_smiles'])
  ]
  result_dicts.append({
    'Seed': config.seed,
    'T': config.sampling.steps,
    'Num Samples': config.sampling.batch_size * config.sampling.num_sample_batches,
    'Valid': valid_pct,
    'Unique': unique_pct,
    'Novel': novel_pct,
    f'{label_col.upper()} Mean': np.mean(mol_property) if len(mol_property) > 0 else 0.,
    f'{label_col.upper()} 25%ile': np.percentile(mol_property, q=25) if len(mol_property) > 0 else 0.,
    f'{label_col.upper()} Median': np.median(mol_property) if len(mol_property) > 0 else 0.,
    f'{label_col.upper()} 75%ile': np.percentile(mol_property, q=75) if len(mol_property) > 0 else 0.,
    f'Novel {label_col.upper()} Mean': np.mean(mol_property_novel) if len(mol_property_novel) > 0 else 0.,
    f'Novel {label_col.upper()} 25%ile': np.percentile(mol_property_novel, q=25) if len(mol_property_novel) > 0 else 0.,
    f'Novel {label_col.upper()} Median': np.median(mol_property_novel) if len(mol_property_novel) > 0 else 0.,
    f'Novel {label_col.upper()} 75%ile': np.percentile(mol_property_novel, q=75) if len(mol_property_novel) > 0 else 0.,
  } | {k.capitalize(): v for k, v in config.guidance.items()})
  print("Guidance:", ", ".join([f"{k.capitalize()} - {v}" for k, v in config.guidance.items()]))
  print(f"\tValid: {valid:,d} / {len(samples):,d} ({100 * valid_pct:0.2f}%) ",
        f"Unique (of valid): {unique:,d} / {valid:,d} ({100 * unique_pct:0.2f}%) ",
        f"Novel (of valid): {novel:,d} / {valid:,d} ({100 * novel_pct:0.2f}%)\n",
        f"\t{label_col.upper()} Mean: {np.mean(mol_property) if len(mol_property) else 0.:0.3f}, {label_col.upper()} Median: {np.median(mol_property) if len(mol_property) else 0.:0.3f}\n",
        f"\tNovel {label_col.upper()} Mean: {np.mean(mol_property_novel) if len(mol_property_novel) else 0.:0.3f}, Novel {label_col.upper()} Median: {np.median(mol_property_novel) if len(mol_property_novel) else 0.:0.3f}"
        )
  print(f"Generated {len(samples)} sentences.")
  with open(config.eval.generated_samples_path, 'w') as f:
    json.dump(
      {
        'valid': valids,
        'novel': list(set(valids) - set(qm9_dataset['canonical_smiles'])),
        f"{label_col}_valid": mol_property,
        f"{label_col}_novel": mol_property_novel,
      },
      f, indent=4) # type: ignore
  results_df = pd.DataFrame.from_records(result_dicts)
  results_df.to_csv(config.eval.results_csv_path)


if __name__ == '__main__':
  main()

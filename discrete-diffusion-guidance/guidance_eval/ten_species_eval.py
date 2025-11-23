import itertools
import json
import os
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
import scipy
import torch
import transformers
from sklearn.metrics import (
  f1_score,
  matthews_corrcoef,
  precision_score,
  recall_score,
  roc_auc_score
)
from tqdm.auto import tqdm

import classifier
import custom_datasets
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


def generate_ordered_kmers(
    kmer_length: int
) -> typing.List[str]:
  """
  Function that generates all kmers of a given length and orders them by their index
  defined by the kmer_to_index function.

  Args:
      kmer_length (int): The length of the kmers to generate

  Returns:
      List[str]: A list of all kmers of the given length ordered by their index
  """
  characters = ["A", "C", "G", "T"]

  kmers = ["".join(kmer) for kmer in
           itertools.product(characters,
                             repeat=kmer_length)]
  ordered_kmers = sorted(kmers, key=kmer_to_index)

  return ordered_kmers


def kmer_to_index(kmer: str) -> int:
  """
  Function that converts a given kmer to a unique value
  system.

  Args:
      kmer (str): The given kmer

  Returns:
      int: The associated unique value

  Example:
      >>> kmer_to_index("AAC")
      1

  """
  mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
  index = 0
  for char in kmer:
    index = index * 4 + mapping[char]
  return index


def compute_kmer_frequencies(
    seqs: typing.List[str], kmer_length: int
) -> typing.Tuple[typing.List[float], typing.List[str]]:
  """
  Computes the kmer frequencies in a list of sequences.
  Each element of the output array is the frequency of a given kmer over the whole
  set of sequences.

  Args:
      seqs (List[str]): List of nucleotide sequences
      kmer_length (int): Length of the kmers

  Returns:
      List[float]: Kmer frequencies
      List[str]: The kmers

  Example:
      >>> sequences = ["AGCT", "AAAA"]
      >>> compute_kmer_frequencies(seqs, kmer_length=1)
      ([0.625, 0.125, 0.125, 0.125], ['A', 'C', 'G', 'T'])
  """

  kmer_counts: typing.Dict[str, int] = {}
  count_kmers_occurrences = 0
  for seq in seqs:
    for i in range(len(seq) - kmer_length + 1):
      kmer = seq[i: i + kmer_length]
      if kmer in kmer_counts:
        kmer_counts[kmer] += 1
      else:
        kmer_counts[kmer] = 1
      count_kmers_occurrences += 1

  kmer_list = generate_ordered_kmers(kmer_length)
  kmer_frequencies = []
  for kmer in kmer_list:
    try:
      kmer_frequencies.append(
        kmer_counts[kmer] / count_kmers_occurrences)
    except KeyError:
      kmer_frequencies.append(0)

  return kmer_frequencies, kmer_list


def run_eval_pipeline(
  seqs: typing.Dict[int, typing.List[str]],
  num_samples_per_class: int,
  train_weights_path: str,
  val_weights_path: str,
  eval_classifier_checkpoint_path: str,
  kmer_freqs_path: str
):
  # Eval pipeline
  L.seed_everything(42)

  # Load classifier
  with hydra.initialize(version_base=None,
                        config_path='../configs/'):
    classifier_config = hydra.compose(
      config_name='config',
      overrides=[
        'hydra.output_subdir=null',
        'hydra.job.chdir=False',
        'hydra/job_logging=disabled',
        'hydra/hydra_logging=disabled',
        '+is_eval_classifier=True',
        'mode=train_classifier',
        'loader.global_batch_size=32',
        'loader.eval_global_batch_size=64',
        'loader.batch_size=2',
        'loader.eval_batch_size=4',
        'data=ten_species',
        'classifier_model=hyenadna-classifier',
        'classifier_model.hyena_model_name_or_path=LongSafari/hyenadna-small-32k-seqlen-hf',
        'classifier_backbone=hyenadna',
        'classifier_model.n_layer=8',
        'model.length=32768',
        'diffusion=null',
        'T=null',
        f"eval.checkpoint_path={eval_classifier_checkpoint_path}"
      ]
    )
  classifier_config = omegaconf.OmegaConf.create(
    classifier_config)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    classifier_config.data.tokenizer_name_or_path,
    trust_remote_code=True)
  pretrained_classifier = classifier.Classifier.load_from_checkpoint(
    classifier_config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=classifier_config, logger=False)
  pretrained_classifier.eval()

  tokenizer = dataloader.get_tokenizer(classifier_config)
  _, val_dl = dataloader.get_dataloaders(
    classifier_config, tokenizer, skip_train=True,
    valid_seed=classifier_config.seed)

  dataset = datasets.load_dataset(
    'yairschiff/ten_species',
    split='train',
    # original dataset only has `train` split
    chunk_length=classifier_config.model.length,
    overlap=0,
    trust_remote_code=True)
  dataset = dataset.train_test_split(
    test_size=0.05, seed=42)
  train_dataset = dataset['train']
  val_dataset = dataset['test']


  print(f"Len of train set {len(train_dataset) * (2 ** 15):,d}")
  print(f"Len of val set {len(val_dataset) * (2 ** 15):,d}")

  int_to_species = ['Homo_sapiens', 'Mus_musculus',
                    'Drosophila_melanogaster',
                    'Danio_rerio',
                    'Caenorhabditis_elegans',
                    'Gallus_gallus', 'Gorilla_gorilla',
                    'Felis_catus',
                    'Salmo_trutta', 'Arabidopsis_thaliana']

  if os.path.exists(train_weights_path):
    train_weights = torch.load(train_weights_path)
  else:
    train_weights = {k: 0 for k in range(10)}
    for i in tqdm(train_dataset, leave=False):
      train_weights[i['species_label']] += 1
    train_weights = {
      k: v / np.sum(list(train_weights.values())) for k, v
      in train_weights.items()}
    torch.save(train_weights, train_weights_path)
  print('Train weights:')
  for k, v in train_weights.items():
    print("\t", int_to_species[k], f"{100 * v:0.2f}")

  if os.path.exists(val_weights_path):
    val_weights = torch.load(val_weights_path)
  else:
    val_weights = {k: 0 for k in range(10)}
    for i in tqdm(val_dataset, leave=False):
      val_weights[i['species_label']] += 1
    val_weights = {k: v / np.sum(list(val_weights.values()))
                   for k, v in val_weights.items()}
    torch.save(val_weights, val_weights_path)
  print('\nVal weights:')
  for k, v in val_weights.items():
    print("\t", int_to_species[k], f"{100 * v:0.2f}")


  result_dict = {}
  test_data = []

  for k, v in seqs.items():
    test_data.extend(
      [
        {
          'sequence': s.replace('[CLS]', '').replace(
            '[BOS]', '').replace('[MASK]', '').replace(
            '[SEP]', '').replace('[PAD]', '').replace(
            '[UNK]', ''),
          'species_label': k
        }
        for s in v
      ]
    )
  test_dataset = custom_datasets.ten_species_dataset.TenSpeciesDataset(
    split='test',
    tokenizer=tokenizer,
    max_length=classifier_config.model.length,
    rc_aug=False,
    add_special_tokens=classifier_config.data.add_special_tokens,
    dataset=test_data
  )

  ## CLASSIFIER ACCURACY
  test_preds = [
    pretrained_classifier.forward(
      test_dataset[i]['input_ids'][None, ...].to(
        'cuda')).argmax(dim=-1).detach().item()
    for i in
    tqdm(range(len(test_dataset)), desc='Testing')
  ]
  test_preds = np.array(test_preds)

  test_labels = []
  for k, v in seqs.items():
    test_labels.extend([int(k)] * len(v))
  test_labels = np.array(test_labels)

  overall_accuracy_score = (test_preds == test_labels).sum() / test_preds.size
  overall_f1_score = f1_score(y_pred=test_preds,
                              y_true=test_labels,
                              average="macro",
                              labels=list(range(classifier_config.data.num_classes)))
  overall_mcc_score = matthews_corrcoef(y_pred=test_preds, y_true=test_labels)

  print(f"Overall Acc: {overall_accuracy_score:0.2f}")
  print(f"Overall F1:  {overall_f1_score:0.2f}")
  print(f"Overall MCC: {overall_mcc_score:0.2f}")
  result_dict['F1'] = overall_f1_score

  f1_scores = f1_score(
    y_pred=test_preds,
    y_true=test_labels,
    average=None,
    labels=list(range(classifier_config.data.num_classes)))
  precision_scores = precision_score(
    y_pred=test_preds,
    y_true=test_labels,
    average=None,
    labels=list(range(classifier_config.data.num_classes)))
  recall_scores = recall_score(
    y_pred=test_preds,
    y_true=test_labels,
    average=None,
    labels=list(range(classifier_config.data.num_classes)))

  species_list = ['Homo_sapiens', 'Mus_musculus',
                  'Drosophila_melanogaster',
                  'Danio_rerio',
                  'Caenorhabditis_elegans',
                  'Gallus_gallus', 'Gorilla_gorilla',
                  'Felis_catus',
                  'Salmo_trutta',
                  'Arabidopsis_thaliana']
  for s in range(classifier_config.data.num_classes):
    print(f"Class {s} - {species_list[s]}:")
    print(f"   F1:        {f1_scores[s]:0.3f}")
    print(f"   Precision: {precision_scores[s]:0.3f}")
    print(f"   Recall:    {recall_scores[s]:0.3f}")

  ## KMER SPECTRUM
  kmer_lengths = [3, 6]
  kmer_results = {k: [] for k in kmer_lengths}
  if os.path.exists(kmer_freqs_path):
    kmer_freqs = torch.load(kmer_freqs_path)
  else:
    kmer_freqs = {s: {
      kmer_length: {'frequencies': None,
                    'kmers': None} for kmer_length in
      kmer_lengths} for s in range(10)}
    for s in range(10):
      filter_ds = val_dataset.filter(
        lambda x: x['species_label'] == s,
        num_proc=len(os.sched_getaffinity(0)))
      print(f"Computing kmer frequencies for species class {s}")
      for kmer_length in kmer_lengths:
        kmer_frequencies_gt, kmer_list = compute_kmer_frequencies(
          seqs=filter_ds['sequence'],
          kmer_length=kmer_length
        )
        kmer_freqs[s][kmer_length]['frequencies'] = kmer_frequencies_gt
        kmer_freqs[s][kmer_length]['kmers'] = kmer_list
    torch.save(kmer_freqs, kmer_freqs_path)
  for s in range(10):
    print(f"Species class {s}")
    mean_js_divergence = 0
    for kmer_length in kmer_lengths:
      kmer_frequencies_gt = kmer_freqs[s][kmer_length]['frequencies']
      kmer_frequencies_generated, kmer_list = compute_kmer_frequencies(
        seqs=[i['sequence'] for i in test_data if
              i['species_label'] == s],
        kmer_length=kmer_length
      )

      js_divergence = np.sum(
        scipy.spatial.distance.jensenshannon(
          kmer_frequencies_gt,
          kmer_frequencies_generated)
      )
      kmer_results[kmer_length].append(js_divergence)
      mean_js_divergence += js_divergence
      print(
        f"\tJS divergence with k={kmer_length} : {js_divergence}")
    print(
      f"\tMean JS divergence : {mean_js_divergence / len(kmer_lengths):0.2f}")

  for k, v in kmer_results.items():
    weighted_kmer_js = (np.array(v) * np.array(
      list(val_weights.values()))).sum()
    print(
      f"Weighted mean JS divergence across classes with k={k}: {weighted_kmer_js:0.2f}")
    result_dict[f"{k}mer JS"] = weighted_kmer_js

  ## DISCRIMINATOR AUROC
  # Hyperparams
  d_model = 128
  n_layer = 2

  batch_size = 8
  lr = 1e-4
  epochs = 5

  disc_data = [
    {'sequence': i['sequence'], 'species_label': 0}
    for i in test_data]
  for s in range(10):
    filter_val_ds = val_dataset.filter(
      lambda x: x['species_label'] == s,
      num_proc=len(os.sched_getaffinity(0)))
    indices = np.random.permutation(
      np.arange(len(filter_val_ds)))[:num_samples_per_class]
    disc_data.extend(
      [{'sequence': i['sequence'], 'species_label': 1}
       for i in filter_val_ds.select(indices)]
    )
  print(f"Size of discriminator dataset: {len(disc_data)}")
  disc_dataset_hf = datasets.Dataset.from_list(
    disc_data)
  disc_dataset_hf = disc_dataset_hf.train_test_split(
    test_size=0.1, seed=42)

  disc_dataset_train = custom_datasets.ten_species_dataset.TenSpeciesDataset(
    split='train',
    tokenizer=tokenizer,
    max_length=classifier_config.model.length,
    rc_aug=False,
    add_special_tokens=classifier_config.data.add_special_tokens,
    dataset=disc_dataset_hf['train']
  )

  disc_dataset_val = custom_datasets.ten_species_dataset.TenSpeciesDataset(
    split='test',
    tokenizer=tokenizer,
    max_length=classifier_config.model.length,
    rc_aug=False,
    add_special_tokens=classifier_config.data.add_special_tokens,
    dataset=disc_dataset_hf['test']
  )

  disc_train_dl = torch.utils.data.DataLoader(
    disc_dataset_train,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True)

  disc_val_dl = torch.utils.data.DataLoader(
    disc_dataset_val,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=False)

  hyena_config = transformers.AutoConfig.from_pretrained(
    'LongSafari/hyenadna-small-32k-seqlen-hf',
    d_model=d_model,
    n_layer=n_layer,
    trust_remote_code=True)
  disc_model = transformers.AutoModelForSequenceClassification.from_config(
    hyena_config,
    pretrained=False,
    num_labels=2,
    problem_type='single_label_classification',
    trust_remote_code=True)

  optimizer = torch.optim.AdamW(
    disc_model.parameters(), lr=lr, weight_decay=0,
    betas=(0.9, 0.999), eps=1e-8)

  disc_model.to('cuda')
  losses = []
  auroc_list = []
  for ep in tqdm(range(epochs), desc='Epochs'):
    # Train loop:
    disc_model.train()
    train_pbar = tqdm(disc_train_dl, desc='Train',
                      leave=False)
    for batch in train_pbar:
      labels = batch['species_label'].to('cuda')
      logits = disc_model(
        batch['input_ids'].to('cuda')).logits
      loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels,
        ignore_index=-100,
        reduction='mean')
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_pbar.set_postfix({'loss': loss.item()})
      losses.append(loss.item())
    # Val loop:
    disc_model.eval()
    disc_labels = []
    disc_preds = []
    for batch in disc_val_dl:
      disc_labels.append(
        batch['species_label'].numpy())
      disc_preds.append(
        disc_model(
          batch['input_ids'].to('cuda')
        ).logits[..., 1].detach().to('cpu').numpy()
      )
    disc_labels = np.concatenate(disc_labels)
    disc_preds = np.concatenate(disc_preds)
    auroc = roc_auc_score(y_true=disc_labels, y_score=disc_preds)
    auroc_list.append(auroc)
    print(f"Ep {ep} - AUROC score {auroc}")
  result_dict["Disc AUROC"] = auroc_list[-1]
  del disc_model
  print('*****************************')
  return result_dict


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

  tokenizer = dataloader.get_tokenizer(config)
  pretrained = diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, logger=False)
  pretrained.eval()

  # Generate samples
  if not os.path.exists(config.eval.generated_samples_path):
    samples_per_class = {}
    classes = range(config.data.num_classes)
    for species in classes:
      config.guidance.condition = species
      print("Guidance:", ", ".join([f"{k.capitalize()} - {v}" for k, v in config.guidance.items()]))
      samples = []
      for _ in tqdm(
        range(config.sampling.num_sample_batches), desc='Gen. batches', leave=False):
        sample = pretrained.sample()
        samples.extend(pretrained.tokenizer.batch_decode(sample))
      samples_per_class[species] = samples
    with open(config.eval.generated_samples_path, 'w') as f:
      json.dump(samples_per_class, f, indent=4) # type: ignore
  else:
    with open(config.eval.generated_samples_path, 'r') as f:
      samples_per_class = json.load(f)
    samples_per_class = {int(k): v for k, v in samples_per_class.items()}

  # Run eval pipeline
  hydra.core.global_hydra.GlobalHydra.instance().clear()
  result_dict = run_eval_pipeline(
    samples_per_class,
    num_samples_per_class=config.sampling.num_sample_batches*config.sampling.batch_size,
    train_weights_path=config.eval.train_weights_path,
    val_weights_path=config.eval.val_weights_path,
    eval_classifier_checkpoint_path=config.eval.eval_classifier_checkpoint_path,
    kmer_freqs_path=config.eval.kmer_freqs_path)
  result_dict['Seed'] = config.seed
  result_dict['T'] = config.sampling.steps
  result_dict = result_dict | {k.capitalize(): v for k, v in config.guidance.items()}
  result_dict['Num Samples'] = sum([len(v) for v in samples_per_class.values()])
  results_df = pd.DataFrame.from_records([result_dict])
  results_df.to_csv(config.eval.results_csv_path)

if __name__ == '__main__':
  main()

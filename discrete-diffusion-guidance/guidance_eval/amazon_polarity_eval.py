import collections
import json
import os

import hydra
import lightning as L
import omegaconf
import pandas as pd
import rdkit
import rich.syntax
import rich.tree
import spacy
import torch
import transformers
# from evaluate import load
from nltk.util import ngrams
from tqdm.auto import tqdm

import dataloader
import diffusion
import eval_utils

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

def compute_diversity(sentences):
  # compute diversity
  ngram_range = [2, 3, 4]

  tokenizer = spacy.load("en_core_web_sm").tokenizer
  token_list = []
  for sentence in sentences:
    token_list.append(
      [str(token) for token in tokenizer(sentence)])
  ngram_sets = {}
  ngram_counts = collections.defaultdict(int)
  n_gram_repetition = {}

  for n in ngram_range:
    ngram_sets[n] = set()
    for tokens in token_list:
      ngram_sets[n].update(ngrams(tokens, n))
      ngram_counts[n] += len(list(ngrams(tokens, n)))
    n_gram_repetition[f"{n}gram_repetition"] = (
          1 - len(ngram_sets[n]) / ngram_counts[n])
  diversity = 1
  for val in n_gram_repetition.values():
    diversity *= (1 - val)
  return diversity


def compute_sentiment_classifier_score(sentences, eval_model_name_or_path):
  tokenizer = transformers.AutoTokenizer.from_pretrained(eval_model_name_or_path)
  eval_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    eval_model_name_or_path).to('cuda')
  eval_model.eval()

  total_pos = 0
  total_neg = 0
  pbar = tqdm(sentences, desc='Classifier eval')
  for sen in pbar:
    # Tokenize the input text
    inputs = tokenizer(
      sen,
      return_tensors="pt",
      truncation=True,
      padding=True).to('cuda')

    # Get the model predictions
    with torch.no_grad():
      outputs = eval_model(**inputs)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(
      outputs.logits, dim=-1)

    # Get the predicted class
    predicted_class = torch.argmax(probs, dim=1).item()
    if predicted_class == 1:
      total_pos += 1
    else:
      total_neg += 1
    pbar.set_postfix(accuracy=total_pos / (total_pos + total_neg))
  return total_pos / (total_pos + total_neg)


# def compute_mauve(config, tokenizer, sentences):
#   os.environ["TOKENIZERS_PARALLELISM"] = "false"
#   # compute mauve
#   torch.cuda.empty_cache()
#   mauve = load("mauve")
#   human_references = []
#
#   valid_loader = dataloader.get_dataloaders(
#     config, tokenizer, valid_seed=config.seed)
#
#   # construct reference
#   for batch_id in range(config.sampling.num_sample_batches):
#     batch = next(iter(valid_loader))
#     input_ids = batch['input_ids']
#     for i in range(config.sampling.batch_size):
#       idx = (
#             input_ids[i] == tokenizer.eos_token_id).nonzero(
#         as_tuple=True)
#       if idx[0].numel() > 0:
#         idx = idx[0][0].item()
#         input_ids[i, (idx + 1):] = 0
#     human_references.extend(
#       tokenizer.batch_decode(
#         input_ids, skip_special_tokens=True))
#
#   assert len(sentences) == len(human_references)
#
#   results = mauve.compute(predictions=sentences,
#                           references=human_references,
#                           featurize_model_name=config.data.mauve_model,
#                           max_text_length=256, device_id=0)
#   return results.mauve



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
  result_dicts = []
  samples = []
  for _ in tqdm(
      range(config.sampling.num_sample_batches),
      desc='Gen. batches', leave=False):
    sample = pretrained.sample()
    samples.extend(
      pretrained.tokenizer.batch_decode(sample))
  samples = [
    s.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').replace('[MASK]', '').strip()
    for s in samples
  ]
  del pretrained  # free up space for eval

  diversity_score = compute_diversity(samples)
  classifier_accuracy = compute_sentiment_classifier_score(
    samples, eval_model_name_or_path=config.eval.classifier_model_name_or_path)

  generative_ppl = eval_utils.compute_generative_ppl(
    samples,
    eval_model_name_or_path=config.eval.generative_ppl_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=config.model.length)

  result_dicts.append({
    'Seed': config.seed,
    'T': config.sampling.steps,
    'Num Samples': config.sampling.batch_size * config.sampling.num_sample_batches,
    'Diversity': diversity_score,
    'Accuracy': classifier_accuracy,
    'Gen. PPL': generative_ppl,
  } | {k.capitalize(): v for k, v in config.guidance.items()})
  print("Guidance:", ", ".join([f"{k.capitalize()} - {v}" for k, v in config.guidance.items()]))
  print(f"\tDiversity: {diversity_score:0.3f} ",
        f"Accuracy: {classifier_accuracy:0.3f} ",
        f"Gen. PPL: {generative_ppl:0.3f}")
  print(f"Generated {len(samples)} sentences.")
  with open(config.eval.generated_samples_path, 'w') as f:
    json.dump(
      {
        'generated_seqs': samples,
      },
      f, indent=4) # type: ignore
  results_df = pd.DataFrame.from_records(result_dicts)
  results_df.to_csv(config.eval.results_csv_path)


if __name__ == '__main__':
  main()

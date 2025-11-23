"""Ten Species Dataset.

Load dataset from HF; tokenize 'on-the-fly'
"""

import random

import datasets
import torch
import transformers

STRING_COMPLEMENT_MAP = {
  "A": "T", "C": "G", "G": "C", "T": "A",
  "a": "t", "c": "g", "g": "c", "t": "a",
  "N": "N", "n": "n",
}


def coin_flip(p=0.5):
    """Flip a (potentially weighted) coin."""
    return random.random() > p


def string_reverse_complement(seq):
    """Reverse complement a DNA sequence."""
    rev_comp = ""
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

class TenSpeciesDataset(torch.utils.data.Dataset):
  """Ten Species Dataset.

  Tokenization happens on the fly.
  """
  def __init__(
      self,
      split: str,
      tokenizer: transformers.PreTrainedTokenizer,
      max_length: int = 1024,
      rc_aug: bool = False,
      add_special_tokens: bool = False,
      dataset=None):
    if dataset is None:
      dataset = datasets.load_dataset(
        'yairschiff/ten_species',
        split='train',  # original dataset only has `train` split
        chunk_length=max_length,
        overlap=0,
        trust_remote_code=True)
      self.dataset = dataset.train_test_split(
        test_size=0.05, seed=42)[split]  # hard-coded seed & size
    else:
      self.dataset = dataset
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.rc_aug = rc_aug
    self.add_special_tokens = add_special_tokens

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    """Returns a sequence and species label."""
    seq = self.dataset[idx]['sequence']
    if self.rc_aug and coin_flip():
      seq = string_reverse_complement(seq)
    seq = self.tokenizer(
      seq,
      max_length=self.max_length,
      padding="max_length",
      truncation=True,
      add_special_tokens=self.add_special_tokens,
      return_attention_mask=True)

    input_ids = seq['input_ids']
    attention_mask = seq['attention_mask']
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)

    return {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'species_label': torch.LongTensor([
        self.dataset[idx]['species_label']]).squeeze(),
    }

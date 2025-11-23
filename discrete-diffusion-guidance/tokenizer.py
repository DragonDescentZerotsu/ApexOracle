"""Custom Tokenization classes."""

import collections
import json
import os
import re
from typing import List, Optional, Tuple, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json'}
PRETRAINED_VOCAB_FILES_MAP = {
    'qm9': {
        'vocab_file': {
            'yairschiff/qm9-tokenizer': 'https://huggingface.co/yairschiff/qm9-tokenizer/resolve/main/vocab.json'
        }
    },
    'zinc250k': {
        'vocab_file': {
            'yairschiff/zinc250k-tokenizer': 'https://huggingface.co/yairschiff/zinc250k-tokenizer/resolve/main/vocab.json'
        }
    }
}


class SMILESTokenizer(PreTrainedTokenizer):
    r"""
    Construct a tokenizer for SMILES datasets.
    Based on regex.

    This tokenizer inherits from [`PreTrainedTokenizer`]
    which contains most of the main methods. Users should
    refer to this superclass for more information regarding
    those methods.

    Adapted from:
        https://huggingface.co/ibm/MoLFormer-XL-both-10pct

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token not in the vocabulary
            cannot be converted to an ID and is set to be
            this token instead.
        sep_token (`str`, *optional*, defaults to `"<eos>"`):
            The separator token, which is used when building
            a sequence from multiple sequences, e.g., two
            sequences for sequence classification or for a
            text and a question for question answering.
            It is also used as the last token of a sequence
            built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example, when
            batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<bos>"`):
            The classifier token which is used when doing
            sequence classification (classification of the
            whole sequence
            instead of per-token classification). It is the
            first token of the sequence when built with
            special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the
            token used when training this model with masked
            language modeling. This is the token, which the
            model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token='<unk>',
        sep_token='<eos>',
        pad_token='<pad>',
        cls_token='<bos>',
        mask_token='<mask>',
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path"
                f"'{vocab_file}'."
            )
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            vocab_from_file = json.load(vocab_handle)
        # Re-index to account for special tokens
        self.vocab = {
            cls_token: 0,
            sep_token: 1,
            mask_token: 2,
            pad_token: 3,
            unk_token: 4,
            **{k: v + 5 for k, v in vocab_from_file.items()}
        }

        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        # Regex pattern taken from:
        #  https://github.com/pschwllr/MolecularTransformer
        self.pattern = (
            r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        )
        self.regex_tokenizer = re.compile(self.pattern)

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).strip()
        return out_string

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of
        sequences for sequence classification tasks by
        concatenating and adding special tokens.
        A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will
                be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence
                pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids)
            with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no
        special tokens added. This method is called when
        adding special tokens using the tokenizer
        `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether the token list is already formatted
                with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range
            [0, 1]: 1 for a special token, 0 for a sequence
            token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be
        used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns
        the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence
                pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(
        self, save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Union[Tuple[str],  None]:
        if not os.path.isdir(save_directory):
            logger.error(
                f"Vocabulary path ({save_directory}) should"
                "be a directory.")
            return None
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    self.vocab,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False
                ) + "\n")
        return (vocab_file,)


class QM9Tokenizer(SMILESTokenizer):
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP['qm9']


class Zinc250kTokenizer(SMILESTokenizer):
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP['zinc250k']

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from rdkit import Chem
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import AgglomerativeClustering
from Bio import Phylo
from triton.language import bfloat16
from scipy.stats import pearsonr, spearmanr
import json
import itertools
import logging

import hydra
from hydra import compose, initialize

import models
from collections import OrderedDict
import noise_schedule

import torch.nn.functional as F
import ast

# current_directory = Path(__file__).parent
current_directory = Path('/data2/tianang/projects/Synergy')

with initialize(config_path="configs"):
    config = compose(config_name="config")

class mol_emb_mdlm(nn.Module):
    def __init__(self, config, vocab_size, ckpt_path, mask_index):
        super(mol_emb_mdlm, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.ckpt_path = ckpt_path
        self.parameterization = self.config.parameterization
        self.time_conditioning = self.config.time_conditioning
        self.backbone = self.load_DIT()  # hidden_size = 768
        # print(self.bert.config.max_position_embeddings)
        self.noise = noise_schedule.get_noise(self.config)

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

    def _sample_t(self, n, device):
        sampling_eps = 1e-3
        _eps_t = torch.rand(n, device=device)  
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        return t * 0

    def _forward(self, x, sigma):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = self.backbone.vocab_embed(x)
            c = F.silu(self.backbone.sigma_map(sigma))
            rotary_cos_sin = self.backbone.rotary_emb(x)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(len(self.backbone.blocks)):
                    x = self.backbone.blocks[i](x, rotary_cos_sin, c, seqlens=None)

        return x

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def forward(self, input_ids, attention_mask=None):
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, dsigma = self.noise(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = self.q_xt(input_ids, move_chance)
        outputs = self._forward(xt, unet_conditioning)
        return outputs

    def load_DIT(self):
        backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        lightning_ckpt = torch.load(self.ckpt_path, map_location='cpu')
        state_dict = lightning_ckpt['state_dict']

        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k[len('backbone.'):]
            else:
                new_key = k
            new_sd[new_key] = v

        backbone.load_state_dict(new_sd, strict=False)

        return backbone


if __name__ == '__main__':

    pooling_method = 'cls_wo_pad_eval' # cls_wo_pad / mean_w_pad / cls_w_pad / mean_wo_pad / cls_wo_pad_eval / mean_wo_pad_eval

    # Load polymer data
    polymer_data_path = Path('temp_data/pep_MICs_sorted_withsource.csv')
    polymer_data = pd.read_csv(polymer_data_path).values[:,0]
    
    print(f"Loaded polymer data with {len(polymer_data)} rows")
    
    # Extract all unique monomer SMILES
    # monomer_columns = ["monomer A", "monomer B", "monomer C", "monomer D", "monomer E", "monomer F"]
    # all_monomers = set()
    #
    # for col in monomer_columns:
    #     if col in polymer_data.columns:
    #         monomers = polymer_data[col].dropna().unique()
    #         all_monomers.update(monomers)
    #
    # print(f"Total unique monomers found: {len(all_monomers)}")
    
    # Convert SMILES strings to token lists (assuming they are already in SELFIES format or tokenized)
    # Based on the CSV, these appear to be SMILES strings that need to be converted to SELFIES and tokenized
    import selfies as sf
    
    monomer_dict = {}
    valid_count = 0
    
    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for pep_seq in tqdm(polymer_data, desc='Processing monomers'):
        try:
            mol = Chem.MolFromSequence(pep_seq)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            # Convert SMILES to SELFIES
            selfies_str = sf.encoder(smiles)
            # Tokenize SELFIES
            tokens = tokenizer(
                selfies_str.replace('][', '] ['),
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Store the input_ids as numpy array (similar to reference code)
            monomer_dict[pep_seq] = tokens['input_ids'].squeeze(0)
            valid_count += 1
            
        except Exception as e:
            print(f"Error processing {pep_seq}: {e}")
            continue
    
    print(f"Successfully processed {valid_count} monomers")

    device = 'cuda:2'

    # Load MDLM model
    DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/1-255000-fine-tune.ckpt'
    mdlm_model = mol_emb_mdlm(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id)
    mdlm_model.to(device)

    # Generate embeddings for all monomers
    monomer_emb = {}
    for smiles, input_ids in tqdm(monomer_dict.items(), desc='Computing monomer embeddings:'):
        input_ids = input_ids.to(device)
        with torch.amp.autocast('cuda', enabled=True):
            if pooling_method == 'cls_wo_pad':
                monomer_emb[smiles] = mdlm_model(input_ids.unsqueeze(0))[:, 0, :].detach().cpu()
            elif pooling_method == 'cls_wo_pad_eval':
                mdlm_model.eval()
                monomer_emb[smiles] = mdlm_model(input_ids.unsqueeze(0))[:, 0, :].detach().cpu()
            elif pooling_method == 'mean_w_pad':
                input_ids_batch = input_ids.unsqueeze(0)
                seq_len = input_ids_batch.shape[-1]
                paddings = torch.ones([1, 1024]) * tokenizer.pad_token_id
                paddings[0, :seq_len] = input_ids_batch[0, :seq_len]
                output = mdlm_model(paddings.long().to(device))[0, :seq_len, :].detach().cpu()
                monomer_emb[smiles] = torch.mean(output, dim=0)
            elif pooling_method == 'cls_w_pad':
                input_ids_batch = input_ids.unsqueeze(0)
                seq_len = input_ids_batch.shape[-1]
                paddings = torch.ones([1, 1024]) * tokenizer.pad_token_id
                paddings[0, :seq_len] = input_ids_batch[0, :seq_len]
                monomer_emb[smiles] = mdlm_model(paddings.long().to(device))[:, 0, :].detach().cpu()
            elif pooling_method == 'mean_wo_pad':
                input_ids_batch = input_ids.unsqueeze(0)
                output = mdlm_model(input_ids_batch.long().to(device))[0].detach().cpu()
                monomer_emb[smiles] = torch.mean(output, dim=0)
            elif pooling_method == 'mean_wo_pad_eval':
                mdlm_model.eval()
                input_ids_batch = input_ids.unsqueeze(0)
                output = mdlm_model(input_ids_batch.long().to(device))[0].detach().cpu()
                monomer_emb[smiles] = torch.mean(output, dim=0)

    print(f"Generated embeddings for {len(monomer_emb)} monomers")
    if len(monomer_emb) > 0:
        sample_emb = next(iter(monomer_emb.values()))
        print(f"Embedding shape: {sample_emb.shape}")
    
    # Check for identical embeddings
    print("\nChecking for identical embeddings...")
    embedding_list = list(monomer_emb.items())
    identical_pairs = []

    for i in range(len(embedding_list)):
        if i>10:
            break
        for j in range(i + 1, len(embedding_list)):
            if j > 10:
                break
            smiles1, emb1 = embedding_list[i]
            smiles2, emb2 = embedding_list[j]
            
            # Convert to numpy for comparison
            emb1_np = emb1.numpy() if isinstance(emb1, torch.Tensor) else emb1
            emb2_np = emb2.numpy() if isinstance(emb2, torch.Tensor) else emb2
            
            # Check if embeddings are identical (with small tolerance for floating point precision)
            if np.allclose(emb1_np, emb2_np, rtol=1e-09, atol=1e-09):
                identical_pairs.append((smiles1, smiles2))


    if identical_pairs:
        print(f"Found {len(identical_pairs)} pairs of identical embeddings:")
        for smiles1, smiles2 in identical_pairs[:10]:  # Show first 10 pairs
            print(f"  {smiles1} <-> {smiles2}")
        
        if len(identical_pairs) > 10:
            print(f"  ... and {len(identical_pairs) - 10} more pairs")
        
        # Analyze the identical groups
        print("\nAnalyzing identical embedding groups...")
        
        # Create groups of molecules with identical embeddings
        identical_groups = {}
        processed = set()
        
        for smiles1, smiles2 in identical_pairs:
            if smiles1 not in processed and smiles2 not in processed:
                # Find all molecules identical to smiles1
                group = {smiles1, smiles2}
                for other_smiles1, other_smiles2 in identical_pairs:
                    if other_smiles1 in group:
                        group.add(other_smiles2)
                    elif other_smiles2 in group:
                        group.add(other_smiles1)
                
                group_key = frozenset(group)
                if group_key not in identical_groups:
                    identical_groups[group_key] = group
                    processed.update(group)
        
        print(f"Found {len(identical_groups)} groups of molecules with identical embeddings:")
        for i, group in enumerate(list(identical_groups.values())[:5], 1):  # Show first 5 groups
            print(f"\nGroup {i} ({len(group)} molecules):")
            for smiles in sorted(list(group)[:3]):  # Show first 3 in each group
                print(f"  SMILES: {smiles}")
            if len(group) > 3:
                print(f"  ... and {len(group) - 3} more molecules")
    else:
        print("No identical embeddings found.")

    # Save embeddings - convert torch tensors to numpy arrays first
    # monomer_embeddings_np = {}
    # for smiles, emb in monomer_emb.items():
    #     if isinstance(emb, torch.Tensor):
    #         monomer_embeddings_np[smiles] = emb.numpy()[0]
    #     else:
    #         monomer_embeddings_np[smiles] = emb[0]
    
    # Save results
    # np.save("temp_data/stf_polymer_monomer_embeddings.npy", monomer_embeddings_np)
    torch.save(monomer_emb,'temp_data/milk_embeddings.pt')
    # print("Embeddings saved to temp_data/stf_polymer_monomer_embeddings.npy")
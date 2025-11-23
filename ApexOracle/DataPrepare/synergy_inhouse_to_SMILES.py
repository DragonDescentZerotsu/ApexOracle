from aa_seq_to_smiles import *
from rdkit import Chem
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import selfies as sf
from transformers import AutoModel, AutoTokenizer


aa_smiles_dict = get_aa_smiles_dict('./Data/all_aa_smiles_new_handcrafted.csv')

# df = pd.read_csv('./Data/inhouse_synergy/processed/inhouse_synergy_Evo_pep_seq.csv')
df = pd.read_csv('./Data/inhouse_synergy/processed/combine_create_inhouse_synergy_Evo_pep_seq.csv')
column_names = df.columns
data = df.values

# model_name = "ibm-research/materials.selfies-ted"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

for line in tqdm(data):
    error_flag = False
    seq_1 = line[3]
    seq_2 = line[4]
    special_signs = ['(', ')']
    for special_sign in special_signs:
        if special_sign in seq_1 or special_sign in seq_2:
            error_flag = True
    if error_flag:
        continue
    try:
        pep_obj_1 = Peptide(seq_1, aa_smiles_dict=aa_smiles_dict)
    except:
        print("error: ", seq_1)
        exit(1)
    try:
        pep_obj_2 = Peptide(seq_2, aa_smiles_dict=aa_smiles_dict)
    except:
        print("error: ", seq_2)
        exit(1)
    pep_smiles_1 = Chem.MolToSmiles(pep_obj_1.ncTerminus_modified_mols[0])
    pep_smiles_2 = Chem.MolToSmiles(pep_obj_2.ncTerminus_modified_mols[0])

    # SELFIES_1 = sf.encoder(pep_smiles_1)
    # SELFIES_2 = sf.encoder(pep_smiles_2)
    #
    # input_ids_1 = tokenizer(SELFIES_1.replace('][', '] ['))['input_ids']
    # input_ids_2 = tokenizer(SELFIES_2.replace('][', '] ['))['input_ids']

    line[3], line[4] = pep_smiles_1, pep_smiles_2

df_new = pd.DataFrame(data, columns=column_names)

# df_new.to_csv('./Data/inhouse_synergy/processed/inhouse_synergy_Evo_smiles_seq.csv', index=False)
df_new.to_csv('./Data/inhouse_synergy/processed/combine_create_inhouse_synergy_Evo_smiles_seq.csv', index=False)
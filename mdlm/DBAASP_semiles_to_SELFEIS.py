import torch
from transformers import AutoTokenizer
import hydra
import models
from collections import OrderedDict
import pandas as pd
import selfies
from tqdm import tqdm

@hydra.main(version_base=None, config_path='configs',config_name='config')
def main(config):
    MODEL_NAME = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = len(tokenizer.get_vocab())

    backbone = models.dit.DIT(config, vocab_size=vocab_size)

    ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/best.ckpt'

    lightning_ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = lightning_ckpt['state_dict']

    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_key = k[len('backbone.'):]
        else:
            new_key = k
        new_sd[new_key] = v

    backbone.load_state_dict(new_sd, strict=True)

    print(1)

def smiles_to_selfeise(smiles_path, save_selfies_path):
    data = pd.read_csv(smiles_path)
    column_names = data.columns
    # column_names[1] = 'SELFIES'
    data = data.values
    for line in tqdm(data, desc='converting to SELFEIS', unit=' lines'):
        line[1] = selfies.encoder(line[1])
    data_df = pd.DataFrame(data, columns=column_names)
    data_df.to_csv(save_selfies_path, index=False)


if __name__=='__main__':
    smiles_path = '/data2/tianang/projects/Synergy/DataPrepare/Data/DBAASP_id_SMILES_bact_MICs.csv'
    save_selfies_path = '/data2/tianang/projects/Synergy/DataPrepare/Data/DBAASP_id_SELFIES_bact_MICs.csv'
    smiles_to_selfeise(save_selfies_path, save_selfies_path)

    # main()
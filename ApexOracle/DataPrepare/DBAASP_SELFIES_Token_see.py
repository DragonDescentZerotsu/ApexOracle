import pandas as pd
import ast
from transformers import AutoModel, AutoTokenizer

# df = pd.read_csv('/data2/tianang/projects/Synergy/DataPrepare/Data/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv')
#
# token_ids = set()
#
# for token_id_seq in df['SMILES']:
#     token_ids.update(set(ast.literal_eval(token_id_seq)))

model_name = "ibm-research/materials.selfies-ted"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# token_names = tokenizer.convert_ids_to_tokens(list(token_ids))

vocab = tokenizer.get_vocab()
# token_id_forbid = [vocab[word] for word in vocab.keys() if '.' in word]
# token_id_forbid += [vocab[word] for word in vocab.keys() if 'I' in word]
# token_id_forbid += [vocab[word] for word in vocab.keys() if 'Sn' in word]
# token_id_forbid += [vocab[word] for word in vocab.keys() if 'Br' in word]

# print(tokenizer.convert_ids_to_tokens(list(set(token_id_forbid))))
print(vocab.keys())
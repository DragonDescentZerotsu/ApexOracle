from transformers import AutoModel, AutoTokenizer
import json

model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(tokenizer.get_vocab().keys())
print(json.dumps(list(tokenizer.get_vocab().keys()), indent=2))
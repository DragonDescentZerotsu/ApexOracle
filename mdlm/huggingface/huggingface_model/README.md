---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---
![ApexOracle](./hf.png)
# Molecule Embedding Diffusion Language Model (DLM)
This HuggingFace 🤗 implementation code only support molecule embedding extraction with DLM, for generation code please refer to our [main ApexOracle GitHub repo](https://github.com/DragonDescentZerotsu/ApexOracle).

### Example Usage  
1. Clone repo
```shell
git clone https://huggingface.co/Kiria-Nozan/ApexOracle
cd ApexOracle
```
2. Extract embedding
```python
from DLM_emb_model import MolEmbDLM
from transformers import AutoTokenizer
import torch

MODEL_DIR = "Kiria-Nozan/ApexOracle"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = MolEmbDLM.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

seq = "[C][C][O]"          # ← replace with the SELFIES string of your molecule
batch = tokenizer(
    seq.replace('][', '] ['),
    padding=False,
    truncation=False,
    return_tensors="pt",
)
print(batch)

batch.to(device)

with torch.no_grad():
    embeddings = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )                       # (1, seq_len + 2, hidden_size), including <cls> and <eos> special tokens


print(f"Embedding shape: {embeddings.shape}")
```

### Paper can be found at [Predicting and generating antibiotics against future pathogens with ApexOracle](https://arxiv.org/pdf/2507.07862) 🚀

### Citation
```
@article{leng2025predicting,
  title={Predicting and generating antibiotics against future pathogens with ApexOracle},
  author={Leng, Tianang and Wan, Fangping and Torres, Marcelo Der Torossian and de la Fuente-Nunez, Cesar},
  journal={arXiv preprint arXiv:2507.07862},
  year={2025}
}
```
  

<p align="center">
  <img src="./UPenn_logo.jpg" alt="UPenn" width="300">
</p>

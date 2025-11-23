from DLM_emb_model import MolEmbDLM
from transformers import AutoTokenizer
import torch

MODEL_DIR = "Kiria-Nozan/ApexOracle"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = MolEmbDLM.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

seq = "[C][C][O]"          # ← 替换成你的输入串
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
    )                       # (1, seq_len, hidden_size)


print(embeddings.shape)
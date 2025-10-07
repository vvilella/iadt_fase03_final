# Tech Challenge — Fase 03: Fine-tuning FLAN-T5 (LoRA)

**Tarefa:** gerar descrição de produto a partir do título.  
**Modelo base:** `google/flan-t5-base` (seq2seq).  
**Técnica:** LoRA (leve, MPS-friendly p/ rodar no meu macbook).

## Estrutura (diretório notebooks)
- `01_data_prep.ipynb`: cria `data/train|val|test.jsonl`
- `02_baseline_flan_t5.ipynb`: baseline + `outputs/baseline_val200.jsonl`
- `03_finetune_flan_t5.ipynb`: LoRA, treino, avaliação, comparação e artefatos

## Resultados
**Val(1k), ckpt-9000:**  
- ROUGE-L: **0.1255**  
- BLEU: **1.19**

**200 amostras (antes×depois):**
| Métrica | Baseline | Fine-tuned |
|---|---:|---:|
| ROUGE-L | 0.1223 | **0.1266** |
| BLEU | 0.0003 | **1.2264** |

> O fine-tuned melhora coerência e vocabulário (BLEU sai de ~0 para ~1.2) mantendo ganho leve no ROUGE-L.

## Como reproduzir
1. `01_data_prep.ipynb`: gera `data/*.jsonl`  
2. `02_baseline_flan_t5.ipynb`: salva `outputs/baseline_val200.jsonl`  
3. `03_finetune_flan_t5.ipynb`: carrega `checkpoint-9000`, avalia e salva `artifacts/t5_lora_best/`

## Inferência rápida
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model = PeftModel.from_pretrained(base, "artifacts/t5_lora_best").to(device)

title = "GoFit Weightlifting Glove"
prompt = f"Given a product title, generate its product description.\nTitle: {title}\nDescription:"
enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=224, num_beams=1)
print(tok.decode(out[0], skip_special_tokens=True))

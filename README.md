# Tech Challenge – Fase 03 
Fine-tuning do modelo FLAN-T5 com LoRA

## 1. Descrição geral

Este projeto implementa o fine-tuning do modelo **FLAN-T5-base** para a tarefa de geração de descrições de produtos a partir de seus títulos.

A abordagem segue um fluxo completo: 
 - Preparação de dados;
 - Baseline; 
 - Fine-tuning com LoRA;
 - Playground de inferência.

O objetivo é comparar o desempenho do modelo base com o modelo ajustado, avaliando métricas de similaridade textual (ROUGE-L e BLEU) e analisando qualitativamente as melhorias na geração de texto.

## 2. Estrutura de diretórios

```
IADT_FASE03_FINAL/
│
├── artifacts/
│   └── t5_lora_best/               # Modelo final fine-tunado salvo (LoRA)
│
├── data/                           # Arquivos baixados na indicação
│   ├── trn.json, tst.json, lbl.json
│   ├── train.jsonl, val.jsonl, test.jsonl
│   └── (outros arquivos auxiliares)
│
├── notebooks/
│   ├── 01_data_prep.ipynb          # Criação dos datasets .jsonl
│   ├── 02_baseline_flan_t5.ipynb   # Avaliação do modelo base (sem treino)
│   ├── 03_finetune_flan_t5.ipynb   # Fine-tuning LoRA + métricas
│   └── 04_playground.ipynb         # Inferência interativa (título → descrição)
│
├── outputs/                        # Outputs gerados a partir das execuções
│   ├── baseline_val200.jsonl
│   ├── compare_baseline_vs_finetuned.jsonl
│   ├── eval_snapshot.json
│   └── t5_lora_mps/checkpoint-9000/
│
├── requirements.txt
└── README.md
```

## 3. Execução passo a passo

### 3.1 Preparação de ambiente

O projeto foi desenvolvido em **Python 3.13.3** com **torch MPS (Apple Silicon)**.  
Instale as dependências no ambiente virtual:

```bash
pip install -r requirements.txt
```

Se preferir executar diretamente nos notebooks, inclua a célula:

```python
%pip install -q transformers datasets accelerate peft evaluate rouge-score sacrebleu ipywidgets torch
```

Verifique o dispositivo:

```python
import torch
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)
```

### 3.2 Estrutura de dados

Os arquivos originais `trn.json` e `tst.json` devem estar na pasta `data/`.  
Caso o repositório não contenha esses arquivos, faça o download e coloque-os diretamente em `data/` antes de executar o notebook `01_data_prep.ipynb`.

Cada registro de entrada possui o formato:
```json
{
  "input_text": "Given a product title, generate its product description.\nTitle: GoFit Weightlifting Glove\nDescription:",
  "target_text": "GoFit Diamond-Tac Weightlifting Gloves provide firm grip and hand protection for training."
}
```

### 3.3 Execução dos notebooks

#### 1. `01_data_prep.ipynb`
- Converte os arquivos originais em `train.jsonl`, `val.jsonl`, `test.jsonl`.
- Realiza split 80/10/10.
- Verifica amostras e contagem de registros.

Saídas esperadas:
```
data/train.jsonl
data/val.jsonl
data/test.jsonl
```

#### 2. `02_baseline_flan_t5.ipynb`
- Carrega `google/flan-t5-base`.
- Avalia 200 amostras da validação.
- Gera as métricas baseline:

```
ROUGE-L: 0.1229
BLEU: 0.0
```

Saída esperada:  
`outputs/baseline_val200.jsonl`

#### 3. `03_finetune_flan_t5.ipynb`
- Aplica fine-tuning com **LoRA**.
- Configuração otimizada para Apple MPS:
  - `evaluation_strategy="no"`
  - `BATCH=1`, `GRAD_ACC=16`
  - `SAVE_STEPS=200`
- Realiza avaliação leve posterior (val 1k) e comparação com baseline.

Resultados obtidos:
- Val (1k): ROUGE-L 0.1255 | BLEU 1.19
- 200 amostras:
  - Baseline: ROUGE-L 0.1223 | BLEU 0.0003  
  - Fine-tuned: ROUGE-L 0.1266 | BLEU 1.2264

Saídas esperadas:
```
outputs/eval_snapshot.json
outputs/compare_baseline_vs_finetuned.jsonl
artifacts/t5_lora_best/
```

#### 4. `04_playground.ipynb`
- Notebook interativo para testar títulos livremente.
- Permite ajustar parâmetros de geração:
  - Feixes (num_beams)
  - Temperatura
  - Top P
  - Penalidade de repetição
  - Sem repetição de n-gramas

#### 5. `05_quick_test.ipynb`
- Notebook de uso rápido, com o teste de inferência citado nesse readme

#### 6. `06_compare_checkpoints.ipynb`
- Notebook de comparação de treinamentos, um realizado com epoch=2 e outro com epoch=3, consideração no final do readme

Exemplo de uso:
```python
Title: GoFit Weightlifting Glove
Description: GoFit weightlifting gloves provide grip and protection for heavy lifting, designed for comfort and durability.

Title: Samsung Galaxy SmartTag2
Description: Samsung Galaxy SmartTag2 is a wireless charging adapter that can be used to charge your smartphone or tablet. It is made of high-quality materials and has a slim design. This smart tag features a magnetic clip that clips into the back of the phone. The case is easy to use and secure.
```

## 4. Observações de performance (Apple M-series)

- Execute o treino sem avaliação no loop (`evaluation_strategy="no"`).
- Salve checkpoints a cada 200 steps.
- Use LoRA para reduzir consumo de memória.
- Desative o modo sleep do macOS durante o treino (quebrei a cabeça até chegar aqui rs!).
- Feche aplicativos pesados (tipo o seu Chrome com 50 abas abertas) para liberar RAM.

Tempo médio:
- Data prep: 3–5 min  
- Baseline: 5–8 min  
- Fine-tuning (até ckpt-9000): levou umas 5 horas no mac (macbook m3 pro)
- Avaliação val(1k): ~30 min  

## 5. Resultados consolidados

| Conjunto | ROUGE-L | BLEU |
|-----------|----------|------|
| Baseline (200) | 0.1223 | 0.0003 |
| Fine-tuned (200) | **0.1266** | **1.2264** |
| Val (1k, ckpt-9000) | **0.1255** | **1.19** |

O fine-tuning com LoRA trouxe uma melhora real na fluência e na estrutura das descrições geradas.  
Os ganhos em **ROUGE-L** mostram que o modelo passou a reproduzir mais fielmente a forma das descrições reais — menos ruído e mais coerência.  
Mas o salto mais expressivo veio no **BLEU**, que saiu de praticamente zero para acima de **1.2**, indicando que as frases geradas ficaram muito mais próximas das referências humanas em termos de vocabulário e sintaxe.

Em outras palavras: o modelo não apenas "aprendeu a falar sobre produtos",  
mas começou a **entender o estilo** e **o padrão textual** das descrições do dataset.  
Os resultados são consistentes tanto no conjunto reduzido (200 amostras) quanto na validação completa (1.000 amostras),  
mostrando que o modelo generaliza bem mesmo treinado de forma leve em hardware local.


## 6. Inferência rápida (fora do notebook)
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model = PeftModel.from_pretrained(base, "artifacts/t5_lora_best").to(device)

title = "Nespresso Vertuo Next Coffee Machine"
prompt = f"Given a product title, generate its product description.\nTitle: {title}\nDescription:"
enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
out = model.generate(**enc, max_new_tokens=224, num_beams=2, repetition_penalty=1.8, no_repeat_ngram_size=3, temperature=0.9, top_p=0.9, do_sample=True)
print(tok.decode(out[0], skip_special_tokens=True))

```
## 7. Comparativo de Checkpoints — Explorando diferentes estágios do treino

Durante o fine-tuning, cada época representa uma passagem completa pelos exemplos de treino.  
A quantidade de passos (“steps”) que o modelo executa depende diretamente disso:  
com duas épocas, ele vê cada exemplo duas vezes; com três, ele tem uma exposição 50% maior aos mesmos dados.  

Nos meus experimentos, o checkpoint **6064** correspondeu a **2 épocas completas**, e o checkpoint **9000** a **3 épocas**.  
Essa diferença foi o suficiente para gerar **um salto real de qualidade**:

| Épocas | Checkpoint | ROUGE-L | BLEU |
|--------|-------------|---------|------|
| 2 | 6064 | 0.1117 | 0.9039 |
| 3 | 9000 | **0.1266** | **1.2264** |

Mais uma época de treino fez o modelo continuar aprendendo padrões de linguagem sem cair em overfitting.  
As descrições ficaram mais completas, o vocabulário mais natural e o texto mais coeso.  

Em setups leves como este (LoRA + MPS + batch efetivo pequeno), uma época a mais **muda muito** o estágio de aprendizado,  
porque cada passada adicional pelo dataset refina as representações de texto.  

No fim, **três épocas** se mostraram o ponto de melhor equilíbrio entre aprendizado e generalização —  
o modelo amadureceu, mas continuou leve e estável. 

O **modelo final** usado no playground foi o checkpoint 9000, salvo em `artifacts/t5_lora_best/`.

## 8. Créditos

Desenvolvido por **Victor Nardi Vilella**  
MBA – IA para Devs – Fase 03 (Tech Challenge)

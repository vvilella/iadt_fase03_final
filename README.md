# Tech Challenge — Fase 03

Treinar (fine-tuning) um modelo de linguagem para **gerar a descrição** de um produto a partir do **título**, usando o dataset **AmazonTitles-1.3MM** (`trn.json` com colunas `title` e `content`).

## Entregáveis
- Vídeo (≤ 10 min) demonstrando o modelo antes/depois do fine-tuning.
- Repositório com código do fine-tuning e inferência.
- PDF com links do vídeo e do repositório.

## Estrutura
- data/ # (não vou versionar, pois o arquivo é muito grande)
- notebooks/ # EDA, preparação, treino, avaliação
- src/ # scripts (data_prep.py, train_t5.py, infer.py, app_gradio.py)
- docs/ # PDF final com links

markdown
Copy code

## Modelo alvo (inicial)
- `google/flan-t5-base` (ou `flan-t5-small` se GPU/tempo apertar)

## Como replicar (rascunho)
1. Criar ambiente Python 3.10+ (ou usar Google Colab).
2. `pip install -r requirements.txt`
3. (em breve) notebooks e scripts serão adicionados step-by-step.
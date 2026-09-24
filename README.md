# Classificação de espécies de morcegos com MLP

Trabalho final de PSI0325 (Poli-USP). Classifica 8 espécies de morcegos com uma rede MLP
em PyTorch, a partir de descritores acústicos de ecolocalização (Fmin, Fmax, Bandwidth, FME, duração).

## Estrutura
- `notebooks/`: exploração, preparação, modelos binários (Etapas A e B) e, depois, o multiclasse.
- `artigo/`: artigo no template SBrT 2026.
- `banco/` e `dados_processados/`: dados de pesquisa não publicados, fora do repositório.

## Como reproduzir
Python 3.13 com `torch`, `numpy`, `pandas` (só para ler os CSVs), `matplotlib` e `jupyter`.
Sem scikit-learn: divisão estratificada, normalização, PCA, LDA e métricas são feitos em NumPy.
Semente fixa 42; divisão 70/15/15; normalização ajustada só no treino.
Rode os notebooks em ordem numérica.

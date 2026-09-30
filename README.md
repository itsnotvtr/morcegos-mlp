# Classificação de espécies de morcegos com MLP

Trabalho final de PSI0325 (Poli-USP). Classifica 8 espécies de morcegos com uma rede MLP
em PyTorch, a partir de descritores acústicos de ecolocalização (Fmin, Fmax, Bandwidth, FME, duração).

## Notebooks (rodar em ordem)
| Notebook | Conteúdo |
|---|---|
| `01_exploracao` | Leitura e junção das tabelas, problemas da base, histogramas, PCA e LDA |
| `02_preparacao` | Representações simples (5 atributos) e completa (25), divisão 70/15/15 estratificada, normalização |
| `03_binario_lavali` | Etapa A: *Myotis lavali* × demais (regressão logística e MLP) |
| `04_par_dificil` | Etapa B: *M. lavali* × *Molossops temminckii* (desbalanceamento, atalho de local) |
| `05_checagens` | Rótulos embaralhados, régua da chave de identificação, validação cruzada por local (binário) |
| `06_multiclasse` | Etapa C: multiclasse crescente, de 3 a 8 espécies |
| `07_hiperparametros` | Etapa D: hiperparâmetros, um fator por vez |
| `08_local_multiclasse` | Validação cruzada por local com as 8 espécies |
| `09_teste` | Etapa E: avaliação única no conjunto de teste |
| **`10_entrega`** | **Notebook de entrega: (i) preparação, (ii) treinamento e (iii) avaliação do modelo final, autocontido** |

`artigo/`: artigo no template SBrT 2026.
`banco/` e `dados_processados/`: dados de pesquisa não publicados, fora do repositório.

## Como reproduzir
Versões em `requirements.txt` (Python 3.13.13, CPU).
Sem scikit-learn: divisão estratificada, normalização, PCA, LDA e métricas são feitos em NumPy.
Semente da divisão 42; divisão 70/15/15; normalização ajustada só no treino; o conjunto de
teste só é usado na avaliação final.

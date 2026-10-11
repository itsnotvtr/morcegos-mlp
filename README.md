# Identificação de espécies de morcegos com uma rede MLP

Trabalho final de PSI0325 – Fundamentos de Adaptação e Aprendizado de Máquina (Poli-USP, 2026).
Uma rede MLP em PyTorch identifica 8 espécies de morcegos a partir de medidas do pulso de
ecolocalização (frequência mínima, frequência máxima, largura de banda, frequência de máxima
energia e duração).

Resultado no conjunto de teste: acurácia balanceada de 0,959, contra 0,942 de um modelo linear
treinado nas mesmas condições e 0,697 de uma regra baseada na chave de identificação dos
especialistas.

## Dados

Os dados pertencem ao **Centro de Bioacústica do Norte de Minas Gerais** (projetos Cidade Viva,
em Montes Claros, e Parque Estadual da Lapa Grande, MG) e **não são públicos**. Por isso eles
não estão neste repositório, e os notebooks não podem ser executados a partir dele. Todos os
notebooks estão aqui **já executados**, com as saídas.

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_exploracao` | Leitura e junção das tabelas, problemas da base, histogramas, PCA e LDA |
| `02_preparacao` | Limpeza, divisão estratificada 70/15/15 e normalização |
| `03_binario_lavali` | *Myotis lavali* × demais espécies: regressão logística e MLP |
| `04_par_dificil` | *M. lavali* × *Molossops temminckii*: classes desbalanceadas |
| `05_checagens` | Rótulos embaralhados, regra da chave de identificação, validação cruzada por local |
| `06_multiclasse` | De 3 a 8 espécies, uma por vez |
| `07_hiperparametros` | Hiperparâmetros, um fator por vez (22 configurações) |
| `08_local_multiclasse` | Validação cruzada por local com as 8 espécies |
| `09_teste` | Avaliação no conjunto de teste |
| **`10_entrega`** | **Notebook de entrega: (i) preparação, (ii) treinamento e (iii) avaliação do modelo final** |
| `11_comparacao_linear` | Modelo linear treinado nas mesmas condições da MLP |
| `12_analise_entradas` | Menos medidas, PCA e LDA como entrada; variante de 3 medidas |
| `13_validacao_externa` | Checagens da base (projeto, gravações vizinhas, divisão por noite) e validação externa |

Os notebooks `01` a `09` e `11` registram o caminho até o modelo final. O `10` é a versão
entregue na disciplina: ele lê um arquivo `dados/gravacoes.csv` (uma linha por gravação, com a
espécie e as quatro medidas), que não acompanha o repositório. O `12` foi feito depois da
apresentação, a partir das sugestões dos professores. O `13` reúne as checagens e a validação
externa do artigo, que usam local, data e hora das gravações (só contagens e métricas aparecem
nas saídas).

`artigo/`: o artigo, no template SBrT 2026.

## Como foi feito

- Versões das bibliotecas em `requirements.txt` (Python 3.13, PyTorch em CPU).
- Sem scikit-learn: divisão estratificada, normalização, PCA, LDA e métricas são implementadas
  em NumPy, a partir das fórmulas da disciplina.
- Divisão com semente 42; treinos com sementes 0, 1 e 2; normalização ajustada só no treino.
- Hiperparâmetros escolhidos pela validação; o conjunto de teste é usado para avaliar o modelo
  final e, uma vez, a variante de 3 medidas escolhida pela validação (notebook `12`).

"""Figuras do artigo em largura de coluna (SBrT: coluna dupla, ~8,6 cm).

Rodar da raiz do repositório: .venv/Scripts/python artigo/gera_figuras_artigo.py
A matriz de confusão retreina a MLP final com o mesmo código e as mesmas sementes do
notebook 10 e confere a acurácia balanceada contra dados_processados/etapa_e_resultados.json.
"""
import copy
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAIDA = "artigo/figuras/"
os.makedirs(SAIDA, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.bbox": "tight", "savefig.dpi": 300})
VIRGULA = lambda casas: FuncFormatter(lambda v, _: f"{v:.{casas}f}".replace(".", ","))
LARGURA = 3.4  # polegadas

# 1) curva crescente (Etapa C)
res = pd.read_csv("dados_processados/etapa_c_resultados.csv")
fig, ax = plt.subplots(figsize=(LARGURA, 2.3))
for modelo, rep, peso, cor, traco, rot in [
    ("MLP", "simples", "com peso", "#2a78d6", "-", "MLP (com peso)"),
    ("MLP", "simples", "sem peso", "#2a78d6", ":", "MLP (sem peso)"),
    ("linear", "simples", "sem peso", "#1baf7a", "--", "modelo linear"),
    ("régua da Chave", "FME", "-", "#898781", "-.", "regra de FME da chave"),
]:
    g = res[(res["modelo"] == modelo) & (res["representação"] == rep) & (res["peso"] == peso)]
    g = g.groupby("n espécies")["acurácia balanceada"].mean()
    ax.plot(g.index, g.values, color=cor, linestyle=traco, marker="o", markersize=3, linewidth=1.2, label=rot)
ax.set_xlabel("número de espécies")
ax.set_ylabel("acurácia balanceada")
ax.set_ylim(0.6, 1.0)
ax.yaxis.set_major_formatter(VIRGULA(2))
ax.set_xticks(range(3, 9))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(SAIDA + "curva_crescente.pdf")
plt.close(fig)

if "--so-curva" in sys.argv:
    sys.exit()

# 3) curvas de perda 8 × 64 neurônios (mesmo código e semente do notebook 07: semente só antes da rede)
d = np.load("dados_processados/etapa2.npz", allow_pickle=True)
_ordem = pd.Series(d["y_simples"]).value_counts().index.tolist()
_X = d["X_simples"].astype(np.float32)
_y = np.array([_ordem.index(e) for e in d["y_simples"]])
_tr, _va = d["pos_treino_simples"], d["pos_val_simples"]
_pesos = torch.tensor(len(_tr) / (len(_ordem) * np.bincount(_y[_tr], minlength=len(_ordem))), dtype=torch.float32)
fig, axes = plt.subplots(1, 2, figsize=(LARGURA, 1.9), sharey=True)
for ax, n_oc in zip(axes, [8, 64]):
    torch.manual_seed(0)
    rede = nn.Sequential(nn.Linear(5, n_oc), nn.Tanh(), nn.Linear(n_oc, len(_ordem)))
    otim = torch.optim.Adam(rede.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss(weight=_pesos)
    loader = DataLoader(TensorDataset(torch.from_numpy(_X[_tr]), torch.from_numpy(_y[_tr])), batch_size=64, shuffle=True)
    ptr, pva = [], []
    for _ in range(1000):
        rede.train()
        acc = 0.0
        for xb, yb in loader:
            otim.zero_grad()
            perda = crit(rede(xb), yb)
            perda.backward()
            otim.step()
            acc += perda.item() * len(xb)
        ptr.append(acc / len(_tr))
        rede.eval()
        with torch.no_grad():
            pva.append(crit(rede(torch.from_numpy(_X[_va])), torch.from_numpy(_y[_va])).item())
    ep = int(np.argmin(pva)) + 1
    print(f"{n_oc} neurônios: val. mínima {min(pva):.3f} (época {ep}), final {pva[-1]:.3f}, treino final {ptr[-1]:.3f}")
    ax.plot(ptr, color="#2a78d6", linewidth=0.9, label="treino")
    ax.plot(pva, color="#eb6834", linewidth=0.6, label="validação")
    ax.axvline(ep - 1, color="#898781", linestyle="--", linewidth=0.8)
    ax.set_title(f"{n_oc} neurônios", fontsize=8)
    ax.set_xlabel("época")
    ax.set_ylim(0.05, 0.35)
    ax.yaxis.set_major_formatter(VIRGULA(2))
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("perda")
axes[0].legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(SAIDA + "curvas_8_64.pdf")
plt.close(fig)
if "--so-curvas" in sys.argv:
    sys.exit()

# 2) matriz de confusão no teste (retreino idêntico ao notebook 10)
d = np.load("dados_processados/etapa2.npz", allow_pickle=True)
ordem = pd.Series(d["y_simples"]).value_counts().index.tolist()
N = len(ordem)
X = d["X_simples"].astype(np.float32)
y = np.array([ordem.index(e) for e in d["y_simples"]])
pt, pv, pte = d["pos_treino_simples"], d["pos_val_simples"], d["pos_teste_simples"]
Xtr, ytr, Xva, yva, Xte, yte = X[pt], y[pt], X[pv], y[pv], X[pte], y[pte]
pesos = torch.tensor(len(ytr) / (N * np.bincount(ytr, minlength=N)), dtype=torch.float32)
registro = json.load(open("dados_processados/etapa_e_resultados.json", encoding="utf-8"))["MLP"]

matrizes = []
for s, reg in zip([0, 1, 2], registro):
    torch.manual_seed(s)
    modelo = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, N))
    otim = torch.optim.Adam(modelo.parameters(), lr=0.01)
    torch.manual_seed(s)
    crit = nn.CrossEntropyLoss(weight=pesos)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=True)
    melhor, melhores = float("inf"), None
    for _ in range(1000):
        modelo.train()
        for xb, yb in loader:
            otim.zero_grad()
            crit(modelo(xb), yb).backward()
            otim.step()
        modelo.eval()
        with torch.no_grad():
            pvv = crit(modelo(torch.from_numpy(Xva)), torch.from_numpy(yva)).item()
        if pvv < melhor:
            melhor, melhores = pvv, copy.deepcopy(modelo.state_dict())
    modelo.load_state_dict(melhores)
    modelo.eval()
    with torch.no_grad():
        prev = modelo(torch.from_numpy(Xte)).argmax(dim=1).numpy()
    m = np.zeros((N, N))
    for p, r in zip(prev, yte):
        m[p, r] += 1
    assert abs((np.diag(m) / m.sum(axis=0)).mean() - reg["teste acurácia balanceada"]) < 1e-9
    matrizes.append(m)

m = np.mean(matrizes, axis=0)
frac = m / m.sum(axis=0, keepdims=True)
nomes = [f"{e.split('_')[0]} sp." if e.endswith("_sp") else e.replace("_", " ") for e in ordem]
fig, ax = plt.subplots(figsize=(LARGURA, 3.2))
ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
for i in range(N):
    for j in range(N):
        if m[i, j] >= 0.5:
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if frac[i, j] > 0.5 else "black")
ax.set_xticks(range(N), nomes, rotation=45, ha="right", style="italic", fontsize=6.5)
ax.set_yticks(range(N), nomes, style="italic", fontsize=6.5)
ax.set_xlabel("espécie verdadeira")
ax.set_ylabel("espécie prevista")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.savefig(SAIDA + "matriz_teste.pdf")
plt.close(fig)
print("figuras do artigo em", SAIDA, "(matriz conferida contra a Etapa E)")

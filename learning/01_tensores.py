# -*- coding: utf-8 -*-
"""
LIÇÃO 1 — Tensores
==================

Um tensor é o array n-dimensional do PyTorch: como um np.ndarray, mas com
dois superpoderes que vamos usar nas próximas lições:
  1. pode viver na GPU (`device='cuda'`)
  2. pode lembrar as operações feitas sobre ele para calcular gradientes
     (autograd — lição 2)

Nesta lição só existem tensores "normais" (sem gradiente). Os exercícios
usam os dados reais do projeto e, no fim, você terá calculado o custo anual
do baseline M0 (casa sem bateria) — um número que o paper precisa.

Rode:  .venv\\Scripts\\python.exe learning\\01_tensores.py
O script para no primeiro TODO incompleto.
"""

from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = (PROJECT_ROOT / "data" / "csvs" / "reference"
       / "Simulation_REF_RAD_young-couple_south_poor_south-tilt(45)_PV1000.csv")
df = pd.read_csv(CSV, sep=";")


def check(nome, cond):
    assert cond, f"[{nome}] ainda não está certo — revise e rode de novo"
    print(f"ok  {nome}")


# ---------------------------------------------------------------------------
# 1. Criando tensores a partir de dados
# ---------------------------------------------------------------------------
# `torch.tensor(dados)` copia; `torch.from_numpy(arr)` compartilha memória
# com o numpy (mais rápido, mas cuidado: alterar um altera o outro).
# Obs.: se usar from_numpy aqui, o PyTorch avisa que o array do pandas é
# somente-leitura — inofensivo neste caso (nunca escrevemos nele), e é
# exatamente o risco do compartilhamento de memória se manifestando.
#
# dtype importa: numpy entrega float64; redes neurais usam float32 (metade
# da memória, e as GPUs são muito mais rápidas em float32). O método
# `.float()` converte para float32 (equivale a `.to(torch.float32)`).

demanda_np = df["electricity_demand_rate_W"].to_numpy()  # W, 35040 passos

# TODO 1: crie o tensor `demanda` (em kW, não W!) a partir de demanda_np,
#         em float32. Dica: operações aritméticas funcionam direto no tensor.
demanda = ...

check("1a: tipo", isinstance(demanda, torch.Tensor))
check("1b: dtype float32", demanda.dtype == torch.float32)
check("1c: em kW", abs(demanda.mean().item() - 1.4354) < 1e-3)
# `.item()` extrai o número Python de um tensor escalar — você vai usar
# muito isso para logar losses.


# ---------------------------------------------------------------------------
# 2. Shape e view: um ano como matriz (dias x passos)
# ---------------------------------------------------------------------------
# `.reshape(a, b)` reorganiza sem copiar (quando possível). Nossa série de
# 35040 passos de 15 min é um ano de 365 dias x 96 passos/dia.

# TODO 2: crie `diaria` com shape (365, 96) a partir de `demanda`.
diaria = ...

check("2a: shape", tuple(diaria.shape) == (365, 96))

# Reduções ao longo de um eixo: `t.sum(dim=1)` soma cada linha (dim 0 são
# as linhas/dias, dim 1 as colunas/passos). Energia = potência x tempo:
# cada passo tem 0.25 h, então kWh do dia = soma_do_dia_em_kW * 0.25.

# TODO 3: `energia_dia` (kWh por dia, shape (365,)) e `pior_dia` (índice
#         do dia de MAIOR energia — procure torch.argmax).
energia_dia = ...
pior_dia = ...

check("3a: shape", tuple(energia_dia.shape) == (365,))
check("3b: energia anual", abs(energia_dia.sum().item() - 12574.3) < 1.0)
check("3c: pior dia", int(pior_dia) == 4)


# ---------------------------------------------------------------------------
# 3. Fatiamento (slicing) — igual numpy
# ---------------------------------------------------------------------------
# `diaria[:, a:b]` pega todas as linhas, colunas a..b-1. O pico noturno
# belga é das 18h às 22h: passos 72 a 87 (72 = 18*4).

# TODO 4: `media_noturna` = média (em kW) de todos os dias entre 18h e 22h.
media_noturna = ...

check("4: media noturna", abs(float(media_noturna) - 2.9139) < 1e-3)
# Compare com a média geral (1.44 kW): a casa consome 2x mais à noite —
# é exatamente a janela que a bateria vai atacar.


# ---------------------------------------------------------------------------
# 4. Broadcasting — a regra de ouro
# ---------------------------------------------------------------------------
# Operações entre shapes diferentes se expandem automaticamente quando os
# eixos são compatíveis (iguais, ou um deles é 1):
#     (365, 96) / (365, 1)  → divide cada linha pelo seu escalar
#     (365, 96) / (365,)    → ERRO! (96 vs 365 não casam pela direita)
# O `.unsqueeze(1)` (ou indexar com [:, None]) insere o eixo de tamanho 1.

# TODO 5: `diaria_norm` = cada dia dividido pelo SEU próprio pico (máximo
#         do dia). Dica: torch.amax(diaria, dim=1) dá o pico por dia com
#         shape (365,) — ajuste o shape para o broadcasting funcionar.
diaria_norm = ...

check("5a: shape", tuple(diaria_norm.shape) == (365, 96))
check("5b: picos = 1", torch.allclose(diaria_norm.amax(dim=1),
                                      torch.ones(365)))


# ---------------------------------------------------------------------------
# 5. Juntando tudo: o baseline M0 do projeto
# ---------------------------------------------------------------------------
# M0 = casa sem bateria. O fluxo com a rede em cada passo é
#     liquido = demanda_kW - pv_scale * pv_kW          (PV1000 escalado!)
# Importa quando liquido > 0 (paga tarifa cheia); exporta quando < 0
# (recebe 20% da tarifa). Custo do passo:
#     custo = (import - 0.2 * export) * tarifa * 0.25
#
# Funções úteis: torch.clamp(t, min=0) zera os negativos — com ela você
# obtém import e export sem nenhum if (vetorização: nunca escreva loop
# sobre 35040 passos!).

pv = torch.from_numpy(df["produced_electricity_rate_W"].to_numpy()).float() / 1000
tarifa = torch.from_numpy(df["tar_rtp"].to_numpy()).float()
PV_SCALE = 5.0

# TODO 6: calcule `custo_anual_m0` (escalar Python, EUR) como descrito.
custo_anual_m0 = ...

check("6: custo M0", abs(custo_anual_m0 - 2706.2) < 1.0)
print(f"\n>>> M0 (sem bateria, PV x5, tarifa RTP): EUR {custo_anual_m0:.2f}/ano")
print(">>> Este é o número que o SAC precisa bater. Guarde-o.")


# ---------------------------------------------------------------------------
# 6. GPU — só para ver com os próprios olhos
# ---------------------------------------------------------------------------
# `.to('cuda')` move o tensor para a VRAM. Operações entre tensores exigem
# que TODOS estejam no mesmo device — o erro "expected all tensors on the
# same device" será seu companheiro nas próximas lições.

if torch.cuda.is_available():
    # TODO 7: mova `diaria` para a GPU em `diaria_gpu` e confirme o device.
    diaria_gpu = ...

    check("7: na GPU", diaria_gpu.device.type == "cuda")
    print(f"tensor de {tuple(diaria_gpu.shape)} vivendo na",
          torch.cuda.get_device_name(0))

print("\nLIÇÃO 1 COMPLETA — me avise para revisarmos e abrir a lição 2 (autograd).")

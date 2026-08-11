# Aprendendo PyTorch com o smart-hems

Trilha prática: cada lição é um script com exercícios `TODO` que se
auto-verificam (asserts). Rode com o venv do projeto:

```
.venv\Scripts\python.exe learning\01_tensores.py
```

O script para no primeiro exercício incompleto e diz o que falta. Quando
todos passarem, me avise que eu reviso suas soluções e abrimos a próxima
lição.

## Trilha

| # | Lição | Conceitos PyTorch | Artefato do projeto que ela constrói |
|---|-------|-------------------|--------------------------------------|
| 1 | Tensores | criação, dtype, shape, indexação, broadcasting, reduções, GPU | custo anual do baseline **M0** (sem bateria) |
| 2 | Autograd | `requires_grad`, `backward()`, grafo computacional, `no_grad` | gradiente do custo em relação ao dimensionamento do PV |
| 3 | `nn.Module` + Dataset | `Linear`, `Dataset`/`DataLoader`, loop de treino, loss, otimizador | baseline de previsão por regressão linear |
| 4 | Redes recorrentes | `nn.GRU`, sequências, empacotamento, early stopping, checkpoint | o **forecaster** de demanda/PV |
| 5 | GPU e performance | `.to('cuda')`, batching, mixed precision, profiling | treino do forecaster na RTX 3060 |
| 6 | RL: as sutilezas | `detach`, target networks, dois otimizadores, log-prob com tanh | os testes de controle sem previsão e com futuro perfeito |

Regras do formato combinado:
- Eu escrevo a estrutura e as explicações; **você preenche os `TODO`**.
- Os asserts usam valores de referência calculados independentemente — se
  passou, está certo.
- Não há vergonha em espiar a documentação: https://pytorch.org/docs/stable/torch.html

# Ideia de pesquisa: meta-RL orientado por previsao

## Estado

Ideia para orientar o desenvolvimento e a contribuicao do paper. Ainda nao e
uma especificacao fechada de implementacao.

## Motivacao

O dataset permite observar as mesmas configuracoes residenciais sob clima
atual, clima futuro e diferentes perfis de uso do veiculo eletrico. A
oportunidade de inovacao em reinforcement learning nao esta apenas em aplicar
um algoritmo de meta-learning ao SAC, mas em estudar como o controlador:

1. decide quanto confiar nas previsoes de demanda e geracao fotovoltaica;
2. reconhece uma mudanca de contexto;
3. adapta-se com poucos dias de dados sem retreinar toda a politica.

## Gap candidato

Meta-learning ja foi aplicado a controle de edificios, previsao de carga e
ambientes residenciais. O gap candidato esta na intersecao entre:

- residencia com PV, bateria e veiculo eletrico;
- previsao explicita de demanda e PV;
- transferencia de extremos climaticos atuais para extremos futuros;
- adaptacao com poucos dados;
- separacao entre erro de previsao e erro de controle;
- teste adicional sob maior demanda de mobilidade.

Uma formulacao provisoria da contribuicao e:

> Um controlador meta-RL condicionado por contexto que adapta seletivamente o
> uso de previsoes de demanda e PV diante de mudancas climaticas e de
> mobilidade, preservando o nucleo da politica e reduzindo os dados necessarios
> para adaptacao.

Essa formulacao deve ser validada por uma revisao sistematica antes de qualquer
alegacao de ineditismo.

## Metodo proposto

Separar a observacao do controlador em tres grupos:

### Estado fisico

- demanda, geracao PV e tarifa atuais;
- estado de carga da bateria;
- estado de carga, conexao e necessidade do veiculo eletrico.

### Futuro disponivel

- demanda prevista para as proximas 24 horas;
- geracao PV prevista para as proximas 24 horas;
- tarifa futura.

### Contexto

- perfil familiar e configuracao do PV;
- comportamento climatico recente;
- residuos recentes das previsoes;
- comportamento recente do veiculo eletrico.

Um codificador produz um vetor de contexto a partir da trajetoria recente:

```text
z(t) = contexto(ultimos dias, residuos da previsao, configuracao da casa)
```

O ator e os criticos do SAC recebem o estado, a previsao e esse contexto. A
adaptacao deve alterar apenas o vetor de contexto ou um pequeno modulo
adaptador. O nucleo do ator e dos criticos permanece congelado durante a
adaptacao inicial.

Isso permite testar se uma adaptacao pequena e direcionada e mais eficiente e
estavel do que fazer fine-tuning de toda a rede.

## Construcao das tarefas

Uma tarefa de meta-treinamento pode ser definida por:

```text
familia x configuracao do PV x periodo climatico x qualidade da previsao
```

As tarefas devem ser construidas somente com os cenarios autorizados para
treinamento. Variacoes na qualidade da previsao podem incluir:

- previsao normal da GRU;
- persistencia;
- ausencia de previsao;
- previsao com ruido;
- previsao com vies;
- previsao com atraso;
- perda parcial do horizonte previsto.

O futuro perfeito nao deve ser usado para tornar a politica irrealisticamente
dependente de informacao indisponivel. Ele deve funcionar principalmente como
limite superior experimental.

## Protocolo de dados

O protocolo base continua sendo:

- REF e anos tipicos atual e futuro: treinamento da previsao;
- extremos atuais frios e quentes: treinamento e validacao do controlador;
- extremos futuros frios e quentes: avaliacao de transferencia climatica;
- perfil de EV com alta demanda: teste separado de generalizacao de
  mobilidade.

Para medir adaptacao few-shot, cada cenario futuro precisa ser dividido sem
sobreposicao:

```text
support: 1, 3 ou 7 dias para adaptacao
query:   periodo posterior usado apenas na avaliacao
```

O resultado deve ser apresentado como uma curva de desempenho para zero, um,
tres e sete dias de adaptacao. Se nenhum dado do cenario futuro for usado, o
experimento mede generalizacao zero-shot, e nao adaptacao few-shot.

## Comparacoes

Os baselines minimos sao:

1. SAC treinado do zero;
2. SAC compartilhado sem adaptacao;
3. SAC compartilhado com fine-tuning completo;
4. SAC robusto treinado com perturbacoes de previsao;
5. meta-SAC adaptando toda a rede;
6. meta-SAC adaptando apenas contexto ou um pequeno adaptador.

Cada controlador deve ser avaliado com:

- sem informacao futura;
- persistencia;
- previsao da GRU;
- futuro perfeito.

## Hipoteses

1. A adaptacao do contexto exige menos interacoes do que o fine-tuning de toda
   a politica.
2. A politica condicionada pela confiabilidade da previsao degrada menos
   quando a GRU encontra um clima futuro fora da distribuicao de treinamento.
3. A meta-adaptacao reduz o custo e as violacoes de restricao nos extremos
   futuros em comparacao ao SAC compartilhado sem adaptacao.
4. O perfil de EV com alta demanda revela se o contexto aprendido responde a
   uma mudanca de mobilidade diferente da mudanca climatica.

## Metricas

- custo total de energia;
- importacao no horario de ponta;
- autoconsumo e aproveitamento de PV;
- energia curta ou necessidade do EV nao atendida;
- violacoes dos limites da bateria e do EV;
- desempenho nos piores dias ou semanas;
- numero de dias ou transicoes necessarios para adaptacao;
- arrependimento em relacao ao controlador com futuro perfeito.

O arrependimento operacional pode ser definido como:

```text
regret = custo do controlador avaliado - custo com futuro perfeito
```

Essa comparacao ajuda a separar a perda causada pela previsao da perda causada
pela politica de controle.

## Limites que devem ser declarados

- As 18 configuracoes compartilham poucas trajetorias climaticas; 36
  combinacoes de casa e clima nao representam 36 climas independentes.
- Se as mesmas casas aparecem no treino e no teste, o experimento demonstra
  transferencia temporal para clima futuro, nao generalizacao para residencias
  desconhecidas.
- O perfil de EV com alta demanda e apenas uma condicao fora da distribuicao e
  nao representa toda a diversidade de mobilidade.
- Dividir um mesmo ano em semanas cria tarefas correlacionadas; essas semanas
  nao devem ser apresentadas como ambientes totalmente independentes.
- Os resultados sao obtidos em simulacao e precisam ser descritos dessa forma.

## Escopo recomendado

Na primeira versao:

- manter a GRU como forecaster convencional;
- aplicar meta-learning somente ao controlador;
- adaptar apenas contexto ou um modulo pequeno;
- medir explicitamente a velocidade de adaptacao;
- decompor o valor da previsao usando os quatro modos de futuro;
- manter a alta demanda do EV como teste externo.

Meta-learning simultaneo na GRU e no SAC deve ficar para uma extensao, pois
dificultaria identificar a origem das melhorias.

## Referencias iniciais

- [Meta-Reinforcement Learning for Building Energy Management System](https://arxiv.org/abs/2210.12590)
- [Few-Shot Load Forecasting Under Data Scarcity in Smart Grids](https://arxiv.org/abs/2406.05887)
- [A Meta-Learning Approach for Multi-Objective Reinforcement Learning in Sustainable Home Environments](https://arxiv.org/abs/2407.11489)
- [Meta-RL with Shared Representations Enables Fast Adaptation in Energy Systems](https://arxiv.org/abs/2603.08418)
- [Meta-reinforcement learning for multi-energy building microgrids](https://www.sciencedirect.com/science/article/pii/S0196890426006217)


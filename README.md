# Detecção de Anomalias em Campanhas de E-mail Marketing

[![CI](https://github.com/gustavoprehl/email-campaign-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/gustavoprehl/email-campaign-anomaly-detection/actions/workflows/ci.yml)

Projeto de portfólio que simula dados de campanhas de e-mail marketing (estilo
CRM automotivo multimarca) e compara duas abordagens de detecção de anomalias
de performance — queda de deliverability, cliques anômalos e picos de
engajamento orgânico — uma estatística simples (z-score) e uma de Machine
Learning (Isolation Forest).

## Contexto e motivação

Times de CRM/marketing automation monitoram diariamente métricas de funil
(taxa de abertura, CTR, CTOR) por campanha e disparo. Na prática, esses
alertas costumam ser regras fixas ("abertura caiu abaixo de X%"), que não se
adaptam ao baseline de cada campanha nem distinguem uma queda real de
deliverability de uma variação normal — ou pior, de um pico de engajamento
bom (ex.: uma campanha de Black Friday) que não deveria disparar alarme
nenhum.

Este projeto usa esse cenário como pretexto para comparar, de forma
controlada, duas abordagens de detecção: uma totalmente interpretável
(z-score sobre uma janela móvel) e uma orientada a dados (Isolation Forest).
Como os dados são sintéticos, existe um **gabarito** de quais disparos
realmente têm um problema injetado — o que permite calcular precisão, recall
e F1 de cada método, algo que normalmente não é possível fazer com dados de
produção.

## Arquitetura

```
┌─────────────────────┐     ┌──────────────────┐     ┌───────────────────────────┐     ┌──────────────────┐
│   generate_data.py   │ --> │   aggregate.py   │ --> │  detect_anomalies_*.py    │ --> │   evaluate.py     │
│                      │     │                  │     │  (zscore / isolation_fst) │     │                  │
│ dim_marca            │     │ fato_agregado_   │     │ outputs/anomalias_*.csv   │     │ outputs/          │
│ dim_campanha         │     │ diario.csv       │     │                           │     │ comparativo_      │
│ dim_subscriber       │     │ (grão: disparo)  │     │                           │     │ metodos.csv       │
│ fato_evento          │     │                  │     │                           │     │                  │
│ gabarito_anomalias   │     │                  │     │                           │     │                  │
└─────────────────────┘     └──────────────────┘     └───────────────────────────┘     └──────────────────┘
                                                                                                  │
                                                                                                  v
                                                                            notebooks/exploratory_analysis.ipynb
                                                                            (séries temporais + overlay dos 2 métodos)
```

- **`generate_data.py`**: gera as dimensões (marca, campanha, subscriber) e o
  funil de eventos (`sent → bounce/open → click → unsubscribe`) com seed fixa
  (42), disparando a cada `INTERVALO_DISPARO_DIAS` (3) dias dentro da janela
  de cada campanha. Injeta anomalias em ~18-20% dos disparos
  (`queda_deliverability`, `clique_bot`, `pico_engajamento`) e salva o
  gabarito separadamente.
- **`aggregate.py`**: agrega `fato_evento` (grão evento) para o grão
  `(id_campanha, id_disparo)`, calculando `taxa_abertura`, `ctr` e `ctor`.
- **`detect_anomalies_zscore.py`**: z-score móvel **por campanha**, sobre cada
  métrica isoladamente.
- **`detect_anomalies_ml.py`**: Isolation Forest **por campanha**, treinado no
  vetor `[taxa_abertura, ctr, ctor, qtd_enviados]`.
- **`evaluate.py`**: cruza as duas detecções com o gabarito e calcula matriz
  de confusão, precisão, recall e F1 por método.
- **`notebooks/exploratory_analysis.ipynb`**: visualiza as séries temporais de
  `taxa_abertura` e `ctr`, sobrepondo gabarito e as duas detecções.

## Como rodar

Requer Python 3.11+ (testado com 3.13).

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv venv
source venv/Scripts/activate      # Git Bash / Linux / macOS
# ou, no PowerShell:
# .\venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar o pipeline, na ordem
python src/generate_data.py
python src/aggregate.py
python src/detect_anomalies_zscore.py
python src/detect_anomalies_ml.py
python src/evaluate.py

# 4. (Opcional) Sensibilidade dos parametros de deteccao (z_threshold, contamination)
python src/threshold_sensitivity.py

# 5. (Opcional) Rodar o notebook de visualização
jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory_analysis.ipynb
# ou abra normalmente no Jupyter/VSCode e rode as células
```

Todas as etapas usam seed fixa (42), então rodar novamente reproduz
exatamente os mesmos dados e resultados.

## Testes

```bash
pytest -v
```

Os testes cobrem as funções puras de cada módulo (agregação, cálculo de
z-score, formato de saída do Isolation Forest, matriz de confusão e o
tratamento de `pico_engajamento` como negativo em `evaluate.py`), usando
fixtures pequenas e determinísticas — não dependem de rodar o pipeline
completo. Rodam automaticamente via GitHub Actions a cada push/PR para
`main` (ver badge no topo deste README).

## Resultados

Rodagem de referência (dados sintéticos gerados com seed 42): 625 disparos
avaliados (cadência de 3 em 3 dias por campanha, ver decisões de design), 112
(17,9%) com anomalia injetada — 27 `queda_deliverability`, 43 `clique_bot`,
42 `pico_engajamento`.

A matriz de confusão trata `pico_engajamento` como **negativo esperado**
(não deveria virar alerta de problema, ver seção de design abaixo):

| Método | Precisão | Recall | F1 | Recall `queda_deliverability` | Recall `clique_bot` | Falso alarme em `pico_engajamento` |
|---|---|---|---|---|---|---|
| z-score | 43,8% | 70,0% | 53,8% | 70,4% | 69,8% | 83,3% |
| Isolation Forest | 47,6% | 70,0% | 56,7% | 55,6% | 79,1% | 81,0% |

**Leitura**: com mais disparos por campanha (ver decisão de design abaixo), o
recall do z-score subiu bastante frente à rodagem anterior com cadência
semanal (50,0% → 70,0%) — a perda estrutural dos primeiros disparos de cada
campanha (sem histórico suficiente) agora pesa menos sobre o total. Os dois
métodos convergem para o mesmo recall geral (70,0%), com o Isolation Forest
levando no F1 por ter mais precisão.

Um efeito colateral notável: a taxa de falso alarme em `pico_engajamento` do
z-score **piorou** (45,5% → 83,3%). Com mais disparos recentes na janela
móvel, a estimativa de desvio-padrão do baseline fica mais "apertada" (menos
variância residual não explicada) — então qualquer desvio grande, bom ou
ruim, cruza o limiar de `|z| > 2.5` com mais facilidade. Mais histórico
ajuda o z-score a achar problemas reais, mas também o deixa mais nervoso com
picos de engajamento legítimos. Isso reforça o ponto de design original:
**nenhum dos dois métodos usa a direção do desvio** — ambos reagem a
qualquer desvio grande em relação ao baseline, seja para cima ou para baixo,
"bom" ou "ruim". Um refinamento natural (fora do escopo deste projeto) seria
incorporar a direção do desvio na regra de alerta, por exemplo, só tratando
quedas de abertura como potencial problema.

Resultado completo em [`outputs/comparativo_metodos.csv`](outputs/comparativo_metodos.csv)
e visualização em [`notebooks/exploratory_analysis.ipynb`](notebooks/exploratory_analysis.ipynb).

### Sensibilidade dos parâmetros (`z_threshold` e `contamination`)

Os parâmetros padrão (`z_threshold=2.5`, `contamination=0.15`) não foram só
"escolhas razoáveis" — `src/threshold_sensitivity.py` varre uma faixa de
valores para cada um, reavaliando precisão/recall/F1 com o mesmo critério de
`evaluate.py` (rodar com `python src/threshold_sensitivity.py`):

| `z_threshold` | Precisão | Recall | F1 |
|---|---|---|---|
| 1.5 | 28,5% | 81,4% | 42,2% |
| 2.0 | 35,6% | 75,7% | 48,4% |
| **2.5** | **43,8%** | **70,0%** | **53,8%** |
| 3.0 | 43,8% | 55,7% | 49,1% |
| 3.5 | 49,3% | 51,4% | 50,3% |
| 4.0 | 50,0% | 50,0% | 50,0% |

| `contamination` | Precisão | Recall | F1 |
|---|---|---|---|
| 0.05 | 57,4% | 38,6% | 46,2% |
| 0.10 | 54,1% | 57,1% | 55,6% |
| **0.15** | **47,6%** | **70,0%** | **56,6%** |
| 0.20 | 44,4% | 84,3% | **58,1%** |
| 0.25 | 39,5% | 91,4% | 55,2% |
| 0.30 | 35,6% | 98,6% | 52,3% |

**Leitura**: para o z-score, `2.5` continua sendo o melhor F1 dentre os
valores testados. Para o Isolation Forest, `contamination=0.20` agora bate
o F1 de `0.15` (58,1% vs. 56,6%) — com mais dados, o ponto ótimo de F1 se
deslocou de `0.25` (na rodagem anterior, com menos disparos) para `0.20`,
reforçando que calibrar `contamination` pela proporção real de anomalias
(`0.15` aqui) é um atalho razoável, mas não é garantia do melhor trade-off
em nenhuma das duas rodagens. O script também gera um gráfico comparando as
três curvas (precisão, recall, F1) em
`outputs/charts/sensibilidade_threshold.png` — não versionado (é regenerado
a cada execução do script).

## Decisões de design

**Z-score por campanha, não global.** Cada campanha tem um baseline de
engajamento próprio — tipo de campanha, marca, mix de segmentos da lista
(lead frio vs. cliente engajado, por exemplo). Normalizar globalmente
misturaria essas distribuições e geraria falsos positivos/negativos
sistemáticos entre campanhas com públicos muito diferentes. O mesmo raciocínio
vale para o Isolation Forest, também treinado por campanha.

**Janela móvel com `shift(1)` antes do `rolling`.** O baseline de cada
disparo usa apenas os disparos *anteriores* da mesma campanha — nunca o
próprio valor avaliado —, para que um disparo anômalo não infle seu próprio
baseline e mascare a própria detecção.

**Cadência de disparo a cada 3 dias, não semanal.** Com disparo semanal e
campanhas de 30-90 dias, cada campanha tinha só 4-12 disparos, e os 3
primeiros de cada uma ficavam sem histórico suficiente para o z-score
(`MIN_DISPAROS_HISTORICO=3`) — uma perda estrutural de recall que não tem a
ver com a qualidade do método, só com falta de dados. Aumentar a frequência
de disparo (sem alterar `N_SUBSCRIBERS` nem a duração das campanhas) deu
mais histórico por campanha e mediu diretamente o efeito: o recall do
z-score subiu de 50,0% para 70,0% (ver Resultados). Também expôs um
trade-off novo — mais histórico deixa o baseline mais "apertado", então o
z-score passou a confundir `pico_engajamento` com problema com mais
frequência (45,5% → 83,3% de falso alarme).

**Isolation Forest com `contamination=0.15`.** Alinhado à proporção real de
anomalias injetadas (~18-20%), o que é uma calibração "otimista": em um
cenário de produção real essa proporção não seria conhecida de antemão. Está
documentado assim no código propositalmente, para deixar claro que essa é
uma simplificação didática, não uma prática recomendada em produção.

**`pico_engajamento` como caso de teste, tratado como negativo esperado na
avaliação.** Um pico de engajamento orgânico (ex.: campanha de Black Friday)
é estatisticamente uma anomalia, mas é uma anomalia **boa** — não deveria
virar alerta de problema para o time de CRM. Incluir esse tipo de anomalia no
gabarito, mas classificá-la como negativo na matriz de confusão, testa
diretamente se cada método confunde sazonalidade boa com anomalia ruim (ver
resultados acima: ambos confundem, o Isolation Forest bem mais).

**Amostra versionada em vez de Git LFS.** `data/raw/` e `data/processed/`
são ignorados pelo git (dados grandes e 100% reprodutíveis via seed fixa).
Em vez de Git LFS — que exige que qualquer pessoa clonando o repositório
instale uma ferramenta extra —, uma amostra pequena de cada tabela fica
versionada em `data/sample/`, só para inspeção rápida sem precisar rodar o
gerador.

## Tecnologias usadas

- **Python 3.13**
- **pandas / numpy** — geração, agregação e manipulação dos dados
- **Faker** (`pt_BR`) — nomes e e-mails fictícios dos subscribers
- **scikit-learn** (`IsolationForest`) — detecção de anomalias via ML
- **matplotlib** — visualização no notebook
- **Jupyter (`notebook` + `ipykernel`)** — notebook de análise exploratória
- **pytest** — testes automatizados
- **GitHub Actions** — CI (roda a suíte de testes a cada push/PR para `main`)

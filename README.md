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
  (42). Injeta anomalias em ~18-20% dos disparos (`queda_deliverability`,
  `clique_bot`, `pico_engajamento`) e salva o gabarito separadamente.
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

# 4. (Opcional) Rodar o notebook de visualização
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

Rodagem de referência (dados sintéticos gerados com seed 42): 260 disparos
avaliados, 51 (19,6%) com anomalia injetada — 20 `queda_deliverability`, 20
`clique_bot`, 11 `pico_engajamento`.

A matriz de confusão trata `pico_engajamento` como **negativo esperado**
(não deveria virar alerta de problema, ver seção de design abaixo):

| Método | Precisão | Recall | F1 | Recall `queda_deliverability` | Recall `clique_bot` | Falso alarme em `pico_engajamento` |
|---|---|---|---|---|---|---|
| z-score | 51,3% | 50,0% | 50,6% | 50% | 50% | 45,5% |
| Isolation Forest | 51,9% | 67,5% | 58,7% | 45% | 90% | 81,8% |

**Leitura**: o Isolation Forest tem recall e F1 melhores no geral — não
depende de janela histórica, então funciona desde o primeiro disparo de cada
campanha, e captura bem `clique_bot` (padrão multivariado: CTR sobe sem
abertura proporcional). Em compensação, ele é bem pior em não confundir
`pico_engajamento` com anomalia ruim (81,8% de falso alarme, contra 45,5% do
z-score). Isso acontece porque **nenhum dos dois métodos usa a direção do
desvio** — ambos reagem a qualquer desvio grande em relação ao baseline,
seja para cima ou para baixo, "bom" ou "ruim". Um refinamento natural (fora
do escopo deste projeto) seria incorporar a direção do desvio na regra de
alerta, por exemplo, só tratando quedas de abertura como potencial problema.

Resultado completo em [`outputs/comparativo_metodos.csv`](outputs/comparativo_metodos.csv)
e visualização em [`notebooks/exploratory_analysis.ipynb`](notebooks/exploratory_analysis.ipynb).

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

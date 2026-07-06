"""Dashboard Streamlit do projeto de deteccao de anomalias em campanhas de
e-mail marketing.

Roda o pipeline inteiro (geracao de dados -> agregacao -> deteccao ->
avaliacao -> sensibilidade) EM MEMORIA a cada cold start, sem depender de
arquivos previamente gerados em data/ ou outputs/ (que nao sao versionados
no git, ver README). Isso mantem o app auto-contido e pronto para deploy
(ex.: Streamlit Community Cloud) so com o codigo do repositorio - os dados
sao 100% reprodutiveis via seed fixa (42).

O resultado do pipeline fica em cache (st.cache_data), entao so roda de
fato uma vez por processo/deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import aggregate as agg
import detect_anomalies_ml as dml
import detect_anomalies_zscore as dz
import evaluate as ev
import generate_data as gd
import threshold_sensitivity as ts
import visualization as viz

st.set_page_config(
    page_title="Anomalias em Campanhas de E-mail Marketing",
    page_icon="📧",
    layout="wide",
)


@st.cache_data(show_spinner="Gerando dados sinteticos e rodando o pipeline (so na primeira vez)...")
def carregar_pipeline_completo() -> dict[str, pd.DataFrame]:
    rng, faker = gd._rng_setup(gd.SEED)

    dim_marca = gd.gerar_dim_marca(rng)
    dim_campanha = gd.gerar_dim_campanha(rng)
    dim_subscriber = gd.gerar_dim_subscriber(rng, faker)
    fato_evento, gabarito = gd.gerar_fato_evento(dim_campanha, dim_subscriber, rng)

    agregado = agg.agregar_por_disparo(fato_evento)
    gabarito_ajustado = ev.ajustar_gabarito(gabarito)

    zscore = dz.detectar_anomalias_zscore(agregado)
    isolation_forest = dml.detectar_anomalias_ml(agregado)
    deteccoes = ev.reduzir_deteccoes_por_disparo(zscore, isolation_forest)

    comparativo = pd.DataFrame(
        [ev.avaliar_metodo(deteccoes, gabarito_ajustado, metodo) for metodo in ["zscore", "isolation_forest"]]
    )

    sensibilidade_zscore = ts.avaliar_zscore_em_varios_thresholds(agregado, gabarito_ajustado, ts.Z_THRESHOLDS)
    sensibilidade_if = ts.avaliar_isolation_forest_em_varios_contaminations(
        agregado, gabarito_ajustado, ts.CONTAMINATIONS
    )
    sensibilidade = pd.concat([sensibilidade_zscore, sensibilidade_if], ignore_index=True)

    return {
        "dim_marca": dim_marca,
        "dim_campanha": dim_campanha,
        "agregado": agregado,
        "gabarito": gabarito_ajustado,
        "zscore": zscore,
        "isolation_forest": isolation_forest,
        "comparativo": comparativo,
        "sensibilidade": sensibilidade,
    }


def formatar_rotulo_campanha(id_campanha: int, dim_campanha: pd.DataFrame, dim_marca: pd.DataFrame) -> str:
    linha = dim_campanha.loc[dim_campanha["id_campanha"] == id_campanha].iloc[0]
    nome_marca = dim_marca.loc[dim_marca["id_marca"] == linha["id_marca"], "nome_marca"].iloc[0]
    return f"Campanha {id_campanha} — {linha['nome_jornada']} ({nome_marca})"


dados = carregar_pipeline_completo()

st.title("📧 Detecção de Anomalias em Campanhas de E-mail Marketing")
st.caption(
    "Dados sintéticos (seed fixa = 42) — comparação entre z-score por campanha e Isolation Forest. "
    "Ver o [README](https://github.com/gustavoprehl/email-campaign-anomaly-detection) para o contexto completo."
)

aba_visao_geral, aba_campanhas = st.tabs(["Visão Geral", "Campanhas"])

with aba_visao_geral:
    gabarito = dados["gabarito"]
    comparativo = dados["comparativo"]

    n_disparos = len(dados["agregado"])
    n_anomalias = int(gabarito["flg_anomalia_injetada"].sum())
    contagem_tipos = (
        gabarito.loc[gabarito["flg_anomalia_injetada"], "tipo_anomalia_injetada"].value_counts()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Disparos avaliados", f"{n_disparos:,}".replace(",", "."))
    col2.metric("Anomalias injetadas", f"{n_anomalias} ({n_anomalias / n_disparos:.1%})")
    col3.metric("queda_deliverability", int(contagem_tipos.get("queda_deliverability", 0)))
    col4.metric("clique_bot / pico_engajamento", f"{int(contagem_tipos.get('clique_bot', 0))} / {int(contagem_tipos.get('pico_engajamento', 0))}")

    st.subheader("Comparativo dos métodos")
    st.caption(
        "`pico_engajamento` é tratado como negativo esperado (não deveria virar alerta de problema) — "
        "ver Decisões de Design no README."
    )
    tabela = comparativo.set_index("metodo")[
        ["tp", "fp", "tn", "fn", "precisao", "recall", "f1"]
    ].copy()
    for coluna in ["precisao", "recall", "f1"]:
        tabela[coluna] = (tabela[coluna] * 100).round(1).astype(str) + "%"
    st.dataframe(tabela, use_container_width=True)

    zscore_row = comparativo.set_index("metodo").loc["zscore"]
    if_row = comparativo.set_index("metodo").loc["isolation_forest"]
    st.markdown(
        f"""
**Leitura**: os dois métodos convergem para o mesmo recall geral
(z-score {zscore_row['recall']:.1%}, Isolation Forest {if_row['recall']:.1%}),
mas o Isolation Forest tem mais precisão ({if_row['precisao']:.1%} vs.
{zscore_row['precisao']:.1%}), levando no F1. O z-score tem um efeito
colateral notável: quanto mais histórico por campanha, mais "apertado" fica
o baseline — e qualquer desvio grande, bom ou ruim, cruza o limiar com mais
facilidade. Isso aparece na taxa de falso alarme em `pico_engajamento`
({zscore_row['taxa_falso_alarme_pico_engajamento']:.1%} no z-score vs.
{if_row['taxa_falso_alarme_pico_engajamento']:.1%} no Isolation Forest):
**nenhum dos dois métodos usa a direção do desvio**, então um pico de
engajamento bom pode ser confundido com anomalia ruim.
"""
    )

    st.subheader("Sensibilidade dos parâmetros")
    st.caption(
        "Como precisão/recall/F1 mudam ao variar o limiar do z-score (`z_threshold`) e o "
        "`contamination` do Isolation Forest — os parâmetros padrão usados acima são "
        "`z_threshold=2.5` e `contamination=0.15`."
    )
    fig_sensibilidade = viz.plotar_sensibilidade(dados["sensibilidade"])
    st.pyplot(fig_sensibilidade)

with aba_campanhas:
    dim_campanha = dados["dim_campanha"]
    dim_marca = dados["dim_marca"]
    gabarito = dados["gabarito"]

    todas_campanhas = sorted(dim_campanha["id_campanha"].tolist())
    rotulos = {
        id_campanha: formatar_rotulo_campanha(id_campanha, dim_campanha, dim_marca)
        for id_campanha in todas_campanhas
    }
    campanhas_exemplo = viz.selecionar_campanhas_exemplo(gabarito, n=4)

    selecionadas = st.multiselect(
        "Campanhas para visualizar (taxa_abertura e ctr, com anomalia real e detecções sobrepostas)",
        options=todas_campanhas,
        default=campanhas_exemplo,
        format_func=lambda id_campanha: rotulos[id_campanha],
    )

    if not selecionadas:
        st.info("Selecione ao menos uma campanha para visualizar os gráficos.")

    for id_campanha in selecionadas:
        st.markdown(f"#### {rotulos[id_campanha]}")
        fig = viz.plotar_campanha(
            id_campanha,
            dados["agregado"],
            gabarito,
            dados["zscore"],
            dados["isolation_forest"],
        )
        st.pyplot(fig)

"""Funcoes de visualizacao compartilhadas entre o notebook de analise
exploratoria (notebooks/exploratory_analysis.ipynb) e o app Streamlit
(app.py) - um unico lugar para a logica de plotagem, para as duas
interfaces nunca divergirem visualmente.

Toda funcao de plot aqui RETORNA a Figure em vez de chamar plt.show(),
para funcionar tanto em notebook (display automatico da ultima expressao)
quanto em Streamlit (st.pyplot(fig)).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

# Paleta categorica validada (contraste + distincao em daltonismo - CVD), com
# forma de marcador distinta por fonte de deteccao. Cor nunca e o unico canal
# de identidade (ver skill de dataviz do projeto).
COR_LINHA = "#2a78d6"              # azul - valor da metrica
COR_ANOMALIA_REAL = "#e34948"      # vermelho - gabarito (anomalia realmente injetada)
COR_ZSCORE = "#4a3aa7"             # violeta - detectado pelo zscore
COR_ISOLATION_FOREST = "#eb6834"   # laranja - detectado pelo isolation forest
COR_PRECISAO = "#2a78d6"
COR_RECALL = "#4a3aa7"
COR_F1 = "#eb6834"

METRICAS_VISUALIZACAO = ["taxa_abertura", "ctr"]


def aplicar_estilo_padrao() -> None:
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.edgecolor"] = "#c3c2b7"
    plt.rcParams["axes.labelcolor"] = "#0b0b0b"
    plt.rcParams["text.color"] = "#0b0b0b"
    plt.rcParams["xtick.color"] = "#52514e"
    plt.rcParams["ytick.color"] = "#52514e"
    plt.rcParams["grid.color"] = "#e1e0d9"
    plt.rcParams["font.size"] = 10


def selecionar_campanhas_exemplo(gabarito: pd.DataFrame, n: int = 4) -> list[int]:
    """Seleciona as `n` campanhas com maior diversidade de tipos de
    anomalia injetada no gabarito, para ilustrar os tres cenarios
    (queda_deliverability, clique_bot, pico_engajamento) no menor numero
    de exemplos possivel."""
    anomalias_reais = gabarito[gabarito["flg_anomalia_injetada"]]
    diversidade = (
        anomalias_reais.groupby("id_campanha")["tipo_anomalia_injetada"]
        .nunique()
        .sort_values(ascending=False)
    )
    return diversidade.head(n).index.tolist()


def plotar_campanha(
    id_campanha: int,
    agregado: pd.DataFrame,
    gabarito: pd.DataFrame,
    zscore: pd.DataFrame,
    isolation_forest: pd.DataFrame,
) -> plt.Figure:
    """Plota taxa_abertura e ctr de uma campanha, sobrepondo anomalia real
    (gabarito) e deteccoes de zscore e Isolation Forest."""
    dados = agregado[agregado["id_campanha"] == id_campanha].sort_values("id_disparo")
    dados_por_disparo = dados.set_index("id_disparo")

    gab_campanha = gabarito[gabarito["id_campanha"] == id_campanha]
    z_campanha = zscore[zscore["id_campanha"] == id_campanha]
    if_campanha = isolation_forest[isolation_forest["id_campanha"] == id_campanha]

    fig, eixos = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for eixo, metrica in zip(eixos, METRICAS_VISUALIZACAO):
        eixo.plot(
            dados["id_disparo"], dados[metrica],
            color=COR_LINHA, linewidth=2, marker="o", markersize=4,
            label=metrica, zorder=2,
        )

        disparos_reais = gab_campanha.loc[gab_campanha["flg_anomalia_injetada"], "id_disparo"]
        eixo.scatter(
            disparos_reais, dados_por_disparo.loc[disparos_reais, metrica],
            color=COR_ANOMALIA_REAL, marker="X", s=160, linewidths=0,
            label="Anomalia real (gabarito)", zorder=5,
        )

        disparos_zscore = z_campanha.loc[
            (z_campanha["metrica_avaliada"] == metrica) & z_campanha["flg_anomalia_detectada"],
            "id_disparo",
        ]
        eixo.scatter(
            disparos_zscore, dados_por_disparo.loc[disparos_zscore, metrica],
            facecolors="none", edgecolors=COR_ZSCORE, marker="^", s=180, linewidths=2,
            label="Detectado - zscore", zorder=4,
        )

        disparos_if = if_campanha.loc[if_campanha["flg_anomalia_detectada"], "id_disparo"]
        eixo.scatter(
            disparos_if, dados_por_disparo.loc[disparos_if, metrica],
            facecolors="none", edgecolors=COR_ISOLATION_FOREST, marker="s", s=180, linewidths=2,
            label="Detectado - Isolation Forest", zorder=3,
        )

        eixo.set_ylabel(metrica)
        eixo.grid(True, linewidth=0.6)

    eixos[-1].set_xlabel("id_disparo")
    eixos[0].set_title(f"Campanha {id_campanha}")
    eixos[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def plotar_sensibilidade(sensibilidade: pd.DataFrame) -> plt.Figure:
    """Curvas de precisao/recall/F1 por valor de parametro, para zscore
    (z_threshold) e Isolation Forest (contamination) lado a lado."""
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

    paineis = [
        ("zscore", "z-score (limiar |z|)"),
        ("isolation_forest", "Isolation Forest (contamination)"),
    ]
    for eixo, (metodo, titulo) in zip(eixos, paineis):
        subset = sensibilidade.loc[sensibilidade["metodo"] == metodo].sort_values("valor")
        eixo.plot(subset["valor"], subset["precisao"], color=COR_PRECISAO, marker="o", label="Precisao")
        eixo.plot(subset["valor"], subset["recall"], color=COR_RECALL, marker="o", label="Recall")
        eixo.plot(subset["valor"], subset["f1"], color=COR_F1, marker="o", linewidth=2.5, label="F1")
        eixo.set_title(titulo)
        eixo.set_xlabel("valor do parametro")
        eixo.set_ylim(0, 1)
        eixo.grid(True, linewidth=0.5, color="#e1e0d9")

    eixos[0].set_ylabel("score")
    eixos[0].legend(frameon=False)
    fig.tight_layout()
    return fig

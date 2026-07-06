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

ROTULOS_METRICA = {
    "taxa_abertura": "Taxa de abertura",
    "ctr": "CTR (cliques ÷ entregues)",
}

ROTULOS_TIPO_ANOMALIA = {
    "queda_deliverability": "queda de entregabilidade",
    "clique_bot": "clique suspeito (possível bot)",
    "pico_engajamento": "pico de engajamento (bom)",
}


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
    (gabarito) e deteccoes de zscore e Isolation Forest. Uma unica legenda
    (com rotulos amigaveis) fica embaixo da figura, compartilhada pelos
    dois paineis."""
    dados = agregado[agregado["id_campanha"] == id_campanha].sort_values("id_disparo")
    dados_por_disparo = dados.set_index("id_disparo")

    gab_campanha = gabarito[gabarito["id_campanha"] == id_campanha]
    z_campanha = zscore[zscore["id_campanha"] == id_campanha]
    if_campanha = isolation_forest[isolation_forest["id_campanha"] == id_campanha]

    fig, eixos = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)

    for eixo, metrica in zip(eixos, METRICAS_VISUALIZACAO):
        eixo.plot(
            dados["id_disparo"], dados[metrica],
            color=COR_LINHA, linewidth=2, marker="o", markersize=4,
            zorder=2,
        )

        disparos_reais = gab_campanha.loc[gab_campanha["flg_anomalia_injetada"], "id_disparo"]
        eixo.scatter(
            disparos_reais, dados_por_disparo.loc[disparos_reais, metrica],
            color=COR_ANOMALIA_REAL, marker="X", s=140, linewidths=0,
            label="Anomalia real", zorder=5,
        )

        disparos_zscore = z_campanha.loc[
            (z_campanha["metrica_avaliada"] == metrica) & z_campanha["flg_anomalia_detectada"],
            "id_disparo",
        ]
        eixo.scatter(
            disparos_zscore, dados_por_disparo.loc[disparos_zscore, metrica],
            facecolors="none", edgecolors=COR_ZSCORE, marker="^", s=160, linewidths=2,
            label="Detecção: Z-score", zorder=4,
        )

        disparos_if = if_campanha.loc[if_campanha["flg_anomalia_detectada"], "id_disparo"]
        eixo.scatter(
            disparos_if, dados_por_disparo.loc[disparos_if, metrica],
            facecolors="none", edgecolors=COR_ISOLATION_FOREST, marker="s", s=160, linewidths=2,
            label="Detecção: Isolation Forest", zorder=3,
        )

        eixo.set_ylabel(ROTULOS_METRICA.get(metrica, metrica))
        eixo.grid(True, linewidth=0.6)

    eixos[-1].set_xlabel("Disparo")
    fig.suptitle(f"Campanha {id_campanha}")

    handles, labels = eixos[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plotar_sensibilidade(sensibilidade: pd.DataFrame) -> plt.Figure:
    """Curvas de precisao/recall/F1 por valor de parametro, para zscore
    (z_threshold) e Isolation Forest (contamination) lado a lado."""
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.5))

    paineis = [
        ("zscore", "z-score (limiar |z|)"),
        ("isolation_forest", "Isolation Forest (contamination)"),
    ]
    for eixo, (metodo, titulo) in zip(eixos, paineis):
        subset = sensibilidade.loc[sensibilidade["metodo"] == metodo].sort_values("valor")
        eixo.plot(subset["valor"], subset["precisao"], color=COR_PRECISAO, marker="o", label="Precisão")
        eixo.plot(subset["valor"], subset["recall"], color=COR_RECALL, marker="o", label="Recall")
        eixo.plot(subset["valor"], subset["f1"], color=COR_F1, marker="o", linewidth=2.5, label="F1")
        eixo.set_title(titulo, fontsize=10)
        eixo.set_xlabel("Valor do parâmetro")
        eixo.set_ylim(0, 1)
        eixo.grid(True, linewidth=0.5, color="#e1e0d9")

    eixos[0].set_ylabel("Score")
    handles, labels = eixos[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def gerar_insight_campanha(
    id_campanha: int,
    agregado: pd.DataFrame,
    zscore: pd.DataFrame,
    isolation_forest: pd.DataFrame,
    gabarito: pd.DataFrame | None = None,
) -> str:
    """Resumo em markdown do que os detectores encontraram numa campanha,
    complementando o grafico com uma leitura em texto.

    Sempre descreve o sinal bruto de cada disparo flagado (metrica, valor e
    desvio no caso do z-score) - isso so depende de `agregado` e das saidas
    dos detectores, que sempre existirao, seja com dado sintetico ou com
    dado real vindo de um DW (GCP, Snowflake etc.).

    Se `gabarito` for informado, adiciona uma segunda camada cruzando com a
    anomalia REAL injetada (disponivel so em dados sinteticos, onde se
    conhece a verdade de antemao). Com `gabarito=None`, a funcao nao faz
    nenhuma afirmacao sobre "acerto" - so relata o que foi observado.
    """
    agregado_campanha = agregado.loc[agregado["id_campanha"] == id_campanha].set_index("id_disparo")
    z_campanha = zscore.loc[zscore["id_campanha"] == id_campanha]
    if_campanha = isolation_forest.loc[isolation_forest["id_campanha"] == id_campanha].set_index("id_disparo")

    linhas_zscore: list[str] = []
    for _, linha in z_campanha.loc[z_campanha["flg_anomalia_detectada"]].sort_values("id_disparo").iterrows():
        id_disparo = int(linha["id_disparo"])
        metrica = linha["metrica_avaliada"]
        valor = agregado_campanha.loc[id_disparo, metrica]
        rotulo_metrica = ROTULOS_METRICA.get(metrica, metrica)
        linhas_zscore.append(
            f"- 🔎 Disparo {id_disparo}: {rotulo_metrica.lower()} em {valor:.1%}, "
            f"{abs(linha['score_anomalia']):.1f} desvios-padrão fora do histórico recente "
            f"da campanha — **Z-score** marcou como anomalia."
        )

    linhas_if: list[str] = []
    disparos_if = if_campanha.loc[if_campanha["flg_anomalia_detectada"]].index.sort_values()
    for id_disparo in disparos_if:
        linha = agregado_campanha.loc[id_disparo]
        linhas_if.append(
            f"- 🔎 Disparo {int(id_disparo)}: combinação de taxa de abertura "
            f"({linha['taxa_abertura']:.1%}) e CTR ({linha['ctr']:.1%}) destoante das demais "
            f"da campanha — **Isolation Forest** marcou como anomalia."
        )

    partes = ["#### O que os detectores encontraram"]
    if linhas_zscore or linhas_if:
        partes.extend(linhas_zscore)
        partes.extend(linhas_if)
    else:
        partes.append("- Nenhum sinal fora do padrão foi detectado nesta campanha.")

    if gabarito is not None:
        gab_campanha = gabarito.loc[gabarito["id_campanha"] == id_campanha]
        anomalias_reais = gab_campanha.loc[gab_campanha["flg_anomalia_injetada"]].sort_values("id_disparo")

        partes.append("\n#### O que de fato aconteceu (gabarito)")
        if anomalias_reais.empty:
            partes.append("- Nenhuma anomalia real foi injetada nos disparos desta campanha.")
        else:
            z_flag_por_disparo = z_campanha.groupby("id_disparo")["flg_anomalia_detectada"].any()
            for _, linha in anomalias_reais.iterrows():
                id_disparo = int(linha["id_disparo"])
                tipo = linha["tipo_anomalia_injetada"]
                detectou_zscore = bool(z_flag_por_disparo.get(id_disparo, False))
                detectou_if = bool(if_campanha["flg_anomalia_detectada"].get(id_disparo, False))

                if detectou_zscore and detectou_if:
                    veredito = "detectada por **ambos os métodos**"
                elif detectou_zscore:
                    veredito = "detectada só pelo **Z-score**"
                elif detectou_if:
                    veredito = "detectada só pelo **Isolation Forest**"
                else:
                    veredito = "**não detectada** por nenhum método"

                marcador = "📈" if tipo == "pico_engajamento" else "⚠️"
                rotulo_tipo = ROTULOS_TIPO_ANOMALIA.get(tipo, tipo)
                partes.append(f"- {marcador} Disparo {id_disparo} — {rotulo_tipo}: {veredito}.")

    return "\n".join(partes)

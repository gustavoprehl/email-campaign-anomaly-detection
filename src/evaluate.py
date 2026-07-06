"""Avalia e compara os dois metodos de deteccao de anomalias (z-score vs.
Isolation Forest) contra o gabarito de anomalias injetadas.

Tratamento especial de "pico_engajamento": um pico de engajamento organico
(ex.: campanha de Black Friday) e uma anomalia estatistica, mas e uma
anomalia BOA — nao deveria virar alerta de problema para o time de CRM.
Por isso, no gabarito ajustado usado na matriz de confusao, disparos com
tipo_anomalia_injetada == "pico_engajamento" sao tratados como negativos
(nao-anomalia-problema), mesmo tendo flg_anomalia_injetada=True no
gabarito bruto. Isso testa diretamente se cada metodo confunde
sazonalidade boa com anomalia ruim — e, como os detectores atuais usam
o valor ABSOLUTO do desvio (nao a direcao), e esperado que ambos gerem
falsos positivos aqui; isso e um resultado valido a discutir no README,
nao um bug a esconder.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

TIPOS_ANOMALIA_PROBLEMA = ["queda_deliverability", "clique_bot"]


def ajustar_gabarito(gabarito: pd.DataFrame) -> pd.DataFrame:
    """Cria flg_anomalia_problema, que exclui pico_engajamento do conjunto
    de positivos (ver docstring do modulo). Funcao pura (sem I/O) para ser
    reutilizada tanto pelo pipeline em disco quanto pelo app Streamlit,
    que roda tudo em memoria."""
    gabarito = gabarito.copy()
    gabarito["flg_anomalia_problema"] = gabarito["flg_anomalia_injetada"] & gabarito[
        "tipo_anomalia_injetada"
    ].isin(TIPOS_ANOMALIA_PROBLEMA)
    return gabarito


def carregar_gabarito_ajustado() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "gabarito_anomalias.csv")
    return ajustar_gabarito(df)


def reduzir_deteccoes_por_disparo(zscore: pd.DataFrame, isolation_forest: pd.DataFrame) -> pd.DataFrame:
    """Reduz as deteccoes de ambos os metodos para uma linha por
    (id_campanha, id_disparo, metodo): um disparo e considerado
    'detectado' pelo z-score se QUALQUER uma das 3 metricas (taxa_abertura,
    ctr, ctor) estourou o limiar — o z-score avalia cada metrica
    isoladamente, entao a agregacao por 'any' e o equivalente a perguntar
    'esse disparo teve algum sinal fora do padrao'. O Isolation Forest ja
    produz uma linha por disparo (metrica_avaliada='combinado').

    Funcao pura (sem I/O) para ser reutilizada tanto pelo pipeline em disco
    quanto pelo app Streamlit."""
    zscore_por_disparo = (
        zscore.groupby(["id_campanha", "id_disparo"])["flg_anomalia_detectada"]
        .any()
        .reset_index()
    )
    zscore_por_disparo["metodo"] = "zscore"

    if_por_disparo = isolation_forest[["id_campanha", "id_disparo", "flg_anomalia_detectada"]].copy()
    if_por_disparo["metodo"] = "isolation_forest"

    return pd.concat([zscore_por_disparo, if_por_disparo], ignore_index=True)


def carregar_deteccoes_por_disparo() -> pd.DataFrame:
    zscore = pd.read_csv(OUTPUTS_DIR / "anomalias_zscore.csv")
    isolation_forest = pd.read_csv(OUTPUTS_DIR / "anomalias_isolation_forest.csv")
    return reduzir_deteccoes_por_disparo(zscore, isolation_forest)


def calcular_matriz_confusao(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    tp = int(((y_true) & (y_pred)).sum())
    fp = int(((~y_true) & (y_pred)).sum())
    tn = int(((~y_true) & (~y_pred)).sum())
    fn = int(((y_true) & (~y_pred)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def calcular_metricas(matriz: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = matriz["tp"], matriz["fp"], matriz["fn"]
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0
    return {"precisao": precisao, "recall": recall, "f1": f1}


def recall_por_tipo(cruzado: pd.DataFrame, tipo: str) -> float:
    subset = cruzado.loc[cruzado["tipo_anomalia_injetada"] == tipo]
    if len(subset) == 0:
        return float("nan")
    return subset["flg_anomalia_detectada"].mean()


def avaliar_metodo(deteccoes: pd.DataFrame, gabarito: pd.DataFrame, metodo: str) -> dict:
    det_metodo = deteccoes.loc[deteccoes["metodo"] == metodo]
    cruzado = gabarito.merge(det_metodo, on=["id_campanha", "id_disparo"], how="left")
    cruzado["flg_anomalia_detectada"] = cruzado["flg_anomalia_detectada"].fillna(False)

    matriz = calcular_matriz_confusao(cruzado["flg_anomalia_problema"], cruzado["flg_anomalia_detectada"])
    metricas = calcular_metricas(matriz)

    taxa_deteccao_pico = recall_por_tipo(cruzado, "pico_engajamento")

    resultado = {
        "metodo": metodo,
        **matriz,
        **metricas,
        "recall_queda_deliverability": recall_por_tipo(cruzado, "queda_deliverability"),
        "recall_clique_bot": recall_por_tipo(cruzado, "clique_bot"),
        "taxa_falso_alarme_pico_engajamento": taxa_deteccao_pico,
    }
    return resultado


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    gabarito = carregar_gabarito_ajustado()
    deteccoes = carregar_deteccoes_por_disparo()

    resultados = [avaliar_metodo(deteccoes, gabarito, metodo) for metodo in ["zscore", "isolation_forest"]]
    comparativo = pd.DataFrame(resultados)

    colunas_ordenadas = [
        "metodo",
        "tp",
        "fp",
        "tn",
        "fn",
        "precisao",
        "recall",
        "f1",
        "recall_queda_deliverability",
        "recall_clique_bot",
        "taxa_falso_alarme_pico_engajamento",
    ]
    comparativo = comparativo[colunas_ordenadas]

    saida = OUTPUTS_DIR / "comparativo_metodos.csv"
    comparativo.to_csv(saida, index=False)

    print("=== Comparativo de metodos (anomalia = 'problema', pico_engajamento tratado como negativo) ===\n")
    for resultado in resultados:
        print(f"--- {resultado['metodo']} ---")
        print(
            f"Matriz de confusao: TP={resultado['tp']} FP={resultado['fp']} "
            f"TN={resultado['tn']} FN={resultado['fn']}"
        )
        print(
            f"Precisao={resultado['precisao']:.2%}  Recall={resultado['recall']:.2%}  "
            f"F1={resultado['f1']:.2%}"
        )
        print(
            f"Recall por tipo -> queda_deliverability={resultado['recall_queda_deliverability']:.2%}  "
            f"clique_bot={resultado['recall_clique_bot']:.2%}"
        )
        print(
            f"Taxa de falso alarme em pico_engajamento (quanto MENOR, melhor o metodo distingue "
            f"sazonalidade boa de anomalia ruim): {resultado['taxa_falso_alarme_pico_engajamento']:.2%}\n"
        )

    print(f"Arquivo salvo em {saida}")


if __name__ == "__main__":
    main()

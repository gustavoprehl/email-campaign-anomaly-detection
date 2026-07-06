"""Analise de sensibilidade dos parametros de deteccao: como precisao,
recall e F1 mudam ao variar o limiar do z-score (|z| > threshold) e o
`contamination` do Isolation Forest.

Usa o mesmo criterio de avaliacao de evaluate.py (pico_engajamento tratado
como negativo esperado), para que os resultados sejam diretamente
comparaveis com o comparativo_metodos.csv da rodagem com os parametros
padrao (z_threshold=2.5, contamination=0.15).

Por que isso importa: os dois parametros hoje sao escolhas fixas (2.5 e
0.15) sem justificativa alem de "parecia razoavel". Esta analise mostra o
trade-off real entre precisao e recall ao variar cada um, e explicita que
"contamination=0.15" so funciona bem porque, neste dataset sintetico,
coincide com a proporcao real de anomalias injetadas — algo que nao se
saberia de antemao em producao (ver detect_anomalies_ml.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import detect_anomalies_ml as dml
import detect_anomalies_zscore as dz
import evaluate as ev
import visualization as viz

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"

Z_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
CONTAMINATIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def carregar_agregado() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "fato_agregado_diario.csv", parse_dates=["dat_referencia"])
    return df.sort_values(["id_campanha", "id_disparo"]).reset_index(drop=True)


def avaliar_zscore_em_varios_thresholds(
    agregado: pd.DataFrame,
    gabarito: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Calcula o z-score de cada metrica uma unica vez (nao muda com o
    threshold) e reavalia o flag de deteccao para cada valor testado."""
    z_por_metrica = {metrica: dz.calcular_zscore_movel(agregado, metrica) for metrica in dz.METRICAS}

    resultados = []
    for threshold in thresholds:
        flg_por_metrica = pd.DataFrame(
            {metrica: z.abs() > threshold for metrica, z in z_por_metrica.items()}
        ).fillna(False)
        flg_disparo = flg_por_metrica.any(axis=1)

        deteccoes = pd.DataFrame(
            {
                "id_campanha": agregado["id_campanha"],
                "id_disparo": agregado["id_disparo"],
                "flg_anomalia_detectada": flg_disparo,
            }
        )
        cruzado = gabarito.merge(deteccoes, on=["id_campanha", "id_disparo"], how="left")
        cruzado["flg_anomalia_detectada"] = cruzado["flg_anomalia_detectada"].fillna(False)

        matriz = ev.calcular_matriz_confusao(cruzado["flg_anomalia_problema"], cruzado["flg_anomalia_detectada"])
        metricas = ev.calcular_metricas(matriz)
        resultados.append({"metodo": "zscore", "parametro": "z_threshold", "valor": threshold, **matriz, **metricas})

    return pd.DataFrame(resultados)


def avaliar_isolation_forest_em_varios_contaminations(
    agregado: pd.DataFrame,
    gabarito: pd.DataFrame,
    contaminations: list[float],
) -> pd.DataFrame:
    """Retreina o Isolation Forest para cada contamination testado. O score
    bruto (decision_function) nao muda com contamination — so o limiar de
    corte do predict() muda —, mas retreinar aqui mantem a analise fiel ao
    que detect_anomalies_ml.py de fato roda em producao."""
    resultados = []
    for contamination in contaminations:
        deteccoes = dml.detectar_anomalias_ml(agregado, contamination=contamination)
        cruzado = gabarito.merge(
            deteccoes[["id_campanha", "id_disparo", "flg_anomalia_detectada"]],
            on=["id_campanha", "id_disparo"],
            how="left",
        )
        cruzado["flg_anomalia_detectada"] = cruzado["flg_anomalia_detectada"].fillna(False)

        matriz = ev.calcular_matriz_confusao(cruzado["flg_anomalia_problema"], cruzado["flg_anomalia_detectada"])
        metricas = ev.calcular_metricas(matriz)
        resultados.append(
            {
                "metodo": "isolation_forest",
                "parametro": "contamination",
                "valor": contamination,
                **matriz,
                **metricas,
            }
        )

    return pd.DataFrame(resultados)


def salvar_grafico_sensibilidade(sensibilidade: pd.DataFrame) -> None:
    """Gera o grafico via visualization.plotar_sensibilidade (compartilhado
    com o app Streamlit) e salva em disco para uso no CLI/README."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = viz.plotar_sensibilidade(sensibilidade)

    caminho = CHARTS_DIR / "sensibilidade_threshold.png"
    fig.savefig(caminho, dpi=150)
    print(f"Grafico salvo em {caminho}")


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    agregado = carregar_agregado()
    gabarito = ev.carregar_gabarito_ajustado()

    sensibilidade_zscore = avaliar_zscore_em_varios_thresholds(agregado, gabarito, Z_THRESHOLDS)
    sensibilidade_if = avaliar_isolation_forest_em_varios_contaminations(agregado, gabarito, CONTAMINATIONS)
    sensibilidade = pd.concat([sensibilidade_zscore, sensibilidade_if], ignore_index=True)

    saida = OUTPUTS_DIR / "sensibilidade_threshold.csv"
    sensibilidade.to_csv(saida, index=False)
    salvar_grafico_sensibilidade(sensibilidade)

    print("=== Sensibilidade ao threshold/contamination ===\n")
    for metodo in ["zscore", "isolation_forest"]:
        subset = sensibilidade.loc[sensibilidade["metodo"] == metodo]
        melhor = subset.loc[subset["f1"].idxmax()]
        print(f"--- {metodo} ---")
        print(subset[["valor", "precisao", "recall", "f1"]].to_string(index=False))
        print(f"Melhor F1: {melhor['f1']:.2%} em valor={melhor['valor']}\n")

    print(f"Arquivo salvo em {saida}")


if __name__ == "__main__":
    main()

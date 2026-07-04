"""Deteccao de anomalias via Isolation Forest, treinado POR CAMPANHA
(mesma justificativa do z-score: cada campanha tem baseline de
engajamento proprio, entao um unico modelo global misturaria
distribuicoes muito diferentes).

Diferenca de abordagem em relacao ao z-score: o Isolation Forest e
multivariado — avalia o vetor (taxa_abertura, ctr, ctor, qtd_enviados)
em conjunto, produzindo UM score por disparo, em vez de um score por
metrica isolada. Por isso, no resultado, metrica_avaliada="combinado"
(nao ha um score de IF "so para ctr", por exemplo).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

FEATURES = ["taxa_abertura", "ctr", "ctor", "qtd_enviados"]
SEED = 42

# Contaminacao alinhada a proporcao real de disparos com anomalia injetada
# (~18-20%, ver generate_data.py). Em um cenario de producao real essa
# proporcao NAO seria conhecida de antemao; aqui ela e usada de forma
# consciente, como um cenario "otimista" de calibracao, para efeito de
# comparacao didatica com o z-score em evaluate.py.
CONTAMINATION = 0.15


def carregar_agregado() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "fato_agregado_diario.csv", parse_dates=["dat_referencia"])
    df = df.sort_values(["id_campanha", "id_disparo"]).reset_index(drop=True)
    print(f"[fato_agregado_diario] {len(df)} disparos carregados.")
    return df


def detectar_anomalias_ml(
    df: pd.DataFrame,
    contamination: float = CONTAMINATION,
) -> pd.DataFrame:
    """Treina um IsolationForest por campanha e retorna formato longo
    (id_campanha, id_disparo, dat_referencia, metrica_avaliada,
    score_anomalia, flg_anomalia_detectada, metodo), no mesmo schema
    usado por detect_anomalies_zscore.py."""
    resultados = []

    for id_campanha, grupo in df.groupby("id_campanha"):
        X = grupo[FEATURES].to_numpy()

        modelo = IsolationForest(contamination=contamination, random_state=SEED)
        modelo.fit(X)

        # decision_function: valores mais baixos = mais anomalo.
        # Invertemos o sinal para que score_anomalia maior = mais anomalo,
        # mantendo a mesma convencao de leitura do |z-score|.
        score_anomalia = -modelo.decision_function(X)
        flg_anomalia_detectada = modelo.predict(X) == -1

        parcial = pd.DataFrame(
            {
                "id_campanha": grupo["id_campanha"].to_numpy(),
                "id_disparo": grupo["id_disparo"].to_numpy(),
                "dat_referencia": grupo["dat_referencia"].to_numpy(),
                "metrica_avaliada": "combinado",
                "score_anomalia": score_anomalia,
                "flg_anomalia_detectada": flg_anomalia_detectada,
                "metodo": "isolation_forest",
            }
        )
        resultados.append(parcial)

    resultado = pd.concat(resultados, ignore_index=True).sort_values(["id_campanha", "id_disparo"]).reset_index(drop=True)
    n_anomalias = resultado["flg_anomalia_detectada"].sum()
    print(
        f"[isolation_forest] {len(resultado)} disparos avaliados (features: {FEATURES}), "
        f"{n_anomalias} marcados como anomalia (contamination={contamination})."
    )
    return resultado


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = carregar_agregado()
    resultado = detectar_anomalias_ml(df)

    saida = OUTPUTS_DIR / "anomalias_isolation_forest.csv"
    resultado.to_csv(saida, index=False)
    print(f"\nArquivo salvo em {saida}")


if __name__ == "__main__":
    main()

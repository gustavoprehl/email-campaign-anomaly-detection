import pandas as pd

import detect_anomalies_zscore as dz


def _agregado_fixture() -> pd.DataFrame:
    """Campanha unica com 6 disparos estaveis (~0.20) seguidos de uma queda
    forte no 6o disparo e recuperacao no 7o."""
    valores_abertura = [0.20, 0.21, 0.19, 0.20, 0.22, 0.02, 0.20]
    n = len(valores_abertura)
    return pd.DataFrame(
        {
            "id_campanha": [1] * n,
            "id_disparo": range(1, n + 1),
            "dat_referencia": pd.date_range("2026-01-01", periods=n, freq="7D"),
            "taxa_abertura": valores_abertura,
            "ctr": [0.05] * n,
            "ctor": [0.25] * n,
        }
    )


def test_calcular_zscore_movel_ignora_disparo_atual_no_baseline():
    df = _agregado_fixture()
    z = dz.calcular_zscore_movel(df, "taxa_abertura")

    # Disparos 1-3: historico insuficiente (min 3 disparos anteriores).
    assert z.iloc[:3].isna().all()
    # Disparo 6 (valor 0.02, muito abaixo do baseline ~0.20).
    assert z.iloc[5] < -2.5


def test_detectar_anomalias_zscore_marca_apenas_o_disparo_anomalo():
    df = _agregado_fixture()
    resultado = dz.detectar_anomalias_zscore(df, z_threshold=2.5)

    esperado_colunas = {
        "id_campanha",
        "id_disparo",
        "metrica_avaliada",
        "score_anomalia",
        "flg_anomalia_detectada",
        "metodo",
    }
    assert esperado_colunas <= set(resultado.columns)
    assert set(resultado["metodo"].unique()) == {"zscore"}

    marcados_abertura = resultado.loc[
        (resultado["metrica_avaliada"] == "taxa_abertura") & resultado["flg_anomalia_detectada"],
        "id_disparo",
    ].tolist()
    assert marcados_abertura == [6]

    # Metricas constantes (ctr, ctor) nao devem gerar falso positivo por
    # divisao por desvio-padrao zero.
    marcados_ctr = resultado.loc[
        (resultado["metrica_avaliada"] == "ctr") & resultado["flg_anomalia_detectada"]
    ]
    assert marcados_ctr.empty

import pandas as pd

import detect_anomalies_ml as ml


def _agregado_fixture() -> pd.DataFrame:
    """Campanha unica com 12 disparos normais e 1 disparo com CTR muito
    acima do padrao (ctor/abertura constantes) - um outlier obvio no
    espaco multivariado."""
    n_normais = 12
    linhas = []
    for i in range(1, n_normais + 1):
        linhas.append(
            {
                "id_campanha": 1,
                "id_disparo": i,
                "dat_referencia": pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * (i - 1)),
                "taxa_abertura": 0.20 + (i % 3) * 0.01,
                "ctr": 0.05 + (i % 2) * 0.005,
                "ctor": 0.25,
                "qtd_enviados": 500,
            }
        )
    linhas.append(
        {
            "id_campanha": 1,
            "id_disparo": n_normais + 1,
            "dat_referencia": pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * n_normais),
            "taxa_abertura": 0.20,
            "ctr": 0.30,
            "ctor": 0.25,
            "qtd_enviados": 500,
        }
    )
    return pd.DataFrame(linhas)


def test_detectar_anomalias_ml_formato_de_saida():
    df = _agregado_fixture()
    resultado = ml.detectar_anomalias_ml(df, contamination=0.1)

    esperado_colunas = {
        "id_campanha",
        "id_disparo",
        "dat_referencia",
        "metrica_avaliada",
        "score_anomalia",
        "flg_anomalia_detectada",
        "metodo",
    }
    assert esperado_colunas <= set(resultado.columns)
    assert (resultado["metrica_avaliada"] == "combinado").all()
    assert (resultado["metodo"] == "isolation_forest").all()
    assert len(resultado) == len(df)


def test_detectar_anomalias_ml_marca_o_disparo_fora_do_padrao():
    df = _agregado_fixture()
    resultado = ml.detectar_anomalias_ml(df, contamination=0.1)

    disparo_outlier = df["id_disparo"].max()
    linha_outlier = resultado.loc[resultado["id_disparo"] == disparo_outlier].iloc[0]
    assert bool(linha_outlier["flg_anomalia_detectada"]) is True

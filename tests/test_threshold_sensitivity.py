import pandas as pd

import threshold_sensitivity as ts


def _agregado_fixture() -> pd.DataFrame:
    """Campanha unica com 6 disparos estaveis (~0.20) e uma queda forte no
    6o disparo - mesmo cenario de tests/test_detect_anomalies_zscore.py."""
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
            "qtd_enviados": [500] * n,
        }
    )


def _gabarito_fixture(agregado: pd.DataFrame) -> pd.DataFrame:
    gabarito = agregado[["id_campanha", "id_disparo"]].copy()
    gabarito["flg_anomalia_problema"] = gabarito["id_disparo"] == 6
    return gabarito


def test_avaliar_zscore_em_varios_thresholds_recall_nao_aumenta_com_threshold_maior():
    agregado = _agregado_fixture()
    gabarito = _gabarito_fixture(agregado)

    resultado = ts.avaliar_zscore_em_varios_thresholds(agregado, gabarito, [1.0, 2.5, 100.0])

    recalls = resultado.set_index("valor")["recall"]
    assert recalls[1.0] >= recalls[2.5] >= recalls[100.0]
    # threshold impossivel de atingir -> nenhuma deteccao -> recall zero.
    assert recalls[100.0] == 0.0
    assert set(resultado["metodo"].unique()) == {"zscore"}


def test_avaliar_isolation_forest_em_varios_contaminations_recall_nao_diminui_com_contamination_maior():
    agregado = _agregado_fixture()
    gabarito = _gabarito_fixture(agregado)

    resultado = ts.avaliar_isolation_forest_em_varios_contaminations(agregado, gabarito, [0.05, 0.5])

    recalls = resultado.set_index("valor")["recall"]
    assert recalls[0.05] <= recalls[0.5]
    assert set(resultado["metodo"].unique()) == {"isolation_forest"}

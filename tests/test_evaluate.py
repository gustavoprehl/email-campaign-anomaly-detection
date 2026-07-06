import pandas as pd
import pytest

import evaluate as ev


def test_calcular_matriz_confusao():
    y_true = pd.Series([True, True, False, False, True])
    y_pred = pd.Series([True, False, False, True, True])

    matriz = ev.calcular_matriz_confusao(y_true, y_pred)

    assert matriz == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}


def test_calcular_metricas():
    matriz = {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    metricas = ev.calcular_metricas(matriz)

    assert metricas["precisao"] == pytest.approx(2 / 3)
    assert metricas["recall"] == pytest.approx(2 / 3)
    assert metricas["f1"] == pytest.approx(2 / 3)


def test_calcular_metricas_sem_positivos_previstos_nao_gera_divisao_por_zero():
    matriz = {"tp": 0, "fp": 0, "tn": 5, "fn": 3}
    metricas = ev.calcular_metricas(matriz)

    assert metricas["precisao"] == 0.0
    assert metricas["recall"] == 0.0
    assert metricas["f1"] == 0.0


def test_gabarito_ajustado_trata_pico_engajamento_como_negativo():
    gabarito = pd.DataFrame(
        [
            {
                "id_campanha": 1,
                "id_disparo": 1,
                "flg_anomalia_injetada": True,
                "tipo_anomalia_injetada": "pico_engajamento",
            },
            {
                "id_campanha": 1,
                "id_disparo": 2,
                "flg_anomalia_injetada": True,
                "tipo_anomalia_injetada": "queda_deliverability",
            },
            {
                "id_campanha": 1,
                "id_disparo": 3,
                "flg_anomalia_injetada": False,
                "tipo_anomalia_injetada": None,
            },
        ]
    )
    gabarito["flg_anomalia_problema"] = gabarito["flg_anomalia_injetada"] & gabarito[
        "tipo_anomalia_injetada"
    ].isin(ev.TIPOS_ANOMALIA_PROBLEMA)

    assert gabarito["flg_anomalia_problema"].tolist() == [False, True, False]


def test_avaliar_metodo_trata_falso_alarme_em_pico_engajamento_como_fp():
    gabarito = pd.DataFrame(
        [
            {
                "id_campanha": 1,
                "id_disparo": 1,
                "tipo_anomalia_injetada": "pico_engajamento",
                "flg_anomalia_problema": False,
            },
            {
                "id_campanha": 1,
                "id_disparo": 2,
                "tipo_anomalia_injetada": "queda_deliverability",
                "flg_anomalia_problema": True,
            },
        ]
    )
    deteccoes = pd.DataFrame(
        [
            {"id_campanha": 1, "id_disparo": 1, "flg_anomalia_detectada": True, "metodo": "zscore"},
            {"id_campanha": 1, "id_disparo": 2, "flg_anomalia_detectada": True, "metodo": "zscore"},
        ]
    )

    resultado = ev.avaliar_metodo(deteccoes, gabarito, "zscore")

    # disparo 1 (pico_engajamento, detectado) -> falso positivo.
    # disparo 2 (queda_deliverability, detectado) -> verdadeiro positivo.
    assert resultado["tp"] == 1
    assert resultado["fp"] == 1
    assert resultado["precisao"] == pytest.approx(0.5)

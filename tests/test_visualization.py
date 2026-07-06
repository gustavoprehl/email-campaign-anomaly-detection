import pandas as pd

import visualization as viz


def _agregado_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id_campanha": 1, "id_disparo": 1, "taxa_abertura": 0.05, "ctr": 0.05},
            {"id_campanha": 1, "id_disparo": 2, "taxa_abertura": 0.20, "ctr": 0.30},
            {"id_campanha": 1, "id_disparo": 3, "taxa_abertura": 0.45, "ctr": 0.40},
            {"id_campanha": 1, "id_disparo": 4, "taxa_abertura": 0.06, "ctr": 0.05},
        ]
    )


def _zscore_fixture() -> pd.DataFrame:
    # Disparo 1: zscore marca taxa_abertura. Disparo 3: zscore marca ctr.
    # Disparo 2 e 4: zscore nao marca nada.
    return pd.DataFrame(
        [
            {"id_campanha": 1, "id_disparo": 1, "metrica_avaliada": "taxa_abertura", "score_anomalia": -3.5, "flg_anomalia_detectada": True},
            {"id_campanha": 1, "id_disparo": 1, "metrica_avaliada": "ctr", "score_anomalia": -0.2, "flg_anomalia_detectada": False},
            {"id_campanha": 1, "id_disparo": 2, "metrica_avaliada": "taxa_abertura", "score_anomalia": 0.1, "flg_anomalia_detectada": False},
            {"id_campanha": 1, "id_disparo": 2, "metrica_avaliada": "ctr", "score_anomalia": 0.3, "flg_anomalia_detectada": False},
            {"id_campanha": 1, "id_disparo": 3, "metrica_avaliada": "taxa_abertura", "score_anomalia": 1.0, "flg_anomalia_detectada": False},
            {"id_campanha": 1, "id_disparo": 3, "metrica_avaliada": "ctr", "score_anomalia": 3.1, "flg_anomalia_detectada": True},
            {"id_campanha": 1, "id_disparo": 4, "metrica_avaliada": "taxa_abertura", "score_anomalia": -0.5, "flg_anomalia_detectada": False},
            {"id_campanha": 1, "id_disparo": 4, "metrica_avaliada": "ctr", "score_anomalia": -0.1, "flg_anomalia_detectada": False},
        ]
    )


def _isolation_forest_fixture() -> pd.DataFrame:
    # Disparo 1: IF tambem marca. Disparo 2: so IF marca. Disparos 3 e 4: IF nao marca.
    return pd.DataFrame(
        [
            {"id_campanha": 1, "id_disparo": 1, "flg_anomalia_detectada": True},
            {"id_campanha": 1, "id_disparo": 2, "flg_anomalia_detectada": True},
            {"id_campanha": 1, "id_disparo": 3, "flg_anomalia_detectada": False},
            {"id_campanha": 1, "id_disparo": 4, "flg_anomalia_detectada": False},
        ]
    )


def _gabarito_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id_campanha": 1, "id_disparo": 1, "flg_anomalia_injetada": True, "tipo_anomalia_injetada": "queda_deliverability"},
            {"id_campanha": 1, "id_disparo": 2, "flg_anomalia_injetada": True, "tipo_anomalia_injetada": "clique_bot"},
            {"id_campanha": 1, "id_disparo": 3, "flg_anomalia_injetada": True, "tipo_anomalia_injetada": "pico_engajamento"},
            {"id_campanha": 1, "id_disparo": 4, "flg_anomalia_injetada": True, "tipo_anomalia_injetada": "queda_deliverability"},
        ]
    )


def test_gerar_insight_campanha_com_gabarito_reporta_deteccao_por_metodo():
    texto = viz.gerar_insight_campanha(
        1, _agregado_fixture(), _zscore_fixture(), _isolation_forest_fixture(), gabarito=_gabarito_fixture()
    )

    assert "Disparo 1" in texto and "ambos os métodos" in texto
    assert "Disparo 2" in texto and "só pelo **Isolation Forest**" in texto
    assert "Disparo 3" in texto and "só pelo **Z-score**" in texto
    assert "Disparo 4" in texto and "não detectada" in texto

    # secao de sinal bruto dos detectores tambem deve mencionar os disparos flagados
    assert "detectores encontraram" in texto
    assert "**Z-score** marcou" in texto
    assert "**Isolation Forest** marcou" in texto


def test_gerar_insight_campanha_sem_gabarito_nao_menciona_verdade():
    texto = viz.gerar_insight_campanha(
        1, _agregado_fixture(), _zscore_fixture(), _isolation_forest_fixture(), gabarito=None
    )

    assert "gabarito" not in texto.lower()
    assert "anomalia real" not in texto.lower()
    # a camada de "o que o detector viu" deve continuar presente
    assert "detectores encontraram" in texto
    assert "**Z-score** marcou" in texto
    assert "**Isolation Forest** marcou" in texto


def test_gerar_insight_campanha_sem_deteccoes_retorna_mensagem_neutra():
    agregado = _agregado_fixture()
    zscore_vazio = _zscore_fixture().assign(flg_anomalia_detectada=False)
    if_vazio = _isolation_forest_fixture().assign(flg_anomalia_detectada=False)

    texto = viz.gerar_insight_campanha(1, agregado, zscore_vazio, if_vazio, gabarito=None)

    assert "Nenhum sinal fora do padrão foi detectado" in texto

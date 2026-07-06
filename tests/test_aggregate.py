import pandas as pd
import pytest

import aggregate as agg


def _fato_evento_fixture() -> pd.DataFrame:
    """Disparo unico: 4 sent, 1 bounce, 2 open (de 3 entregues), 1 click."""
    registros = [
        {"id_subscriber": 101, "tipo_evento": "sent", "dat_evento": "2026-01-01T08:00:00"},
        {"id_subscriber": 101, "tipo_evento": "open", "dat_evento": "2026-01-01T09:00:00"},
        {"id_subscriber": 101, "tipo_evento": "click", "dat_evento": "2026-01-01T10:00:00"},
        {"id_subscriber": 102, "tipo_evento": "sent", "dat_evento": "2026-01-01T08:05:00"},
        {"id_subscriber": 102, "tipo_evento": "open", "dat_evento": "2026-01-01T09:10:00"},
        {"id_subscriber": 103, "tipo_evento": "sent", "dat_evento": "2026-01-01T08:10:00"},
        {"id_subscriber": 103, "tipo_evento": "bounce", "dat_evento": "2026-01-01T08:11:00"},
        {"id_subscriber": 104, "tipo_evento": "sent", "dat_evento": "2026-01-01T08:15:00"},
    ]
    df = pd.DataFrame(registros)
    df["id_campanha"] = 1
    df["id_disparo"] = 1
    df["dat_evento"] = pd.to_datetime(df["dat_evento"])
    df["id_evento"] = range(1, len(df) + 1)
    return df


def test_agregar_por_disparo_calcula_taxas_corretas():
    fato_evento = _fato_evento_fixture()
    agregado = agg.agregar_por_disparo(fato_evento)

    assert len(agregado) == 1
    linha = agregado.iloc[0]
    assert linha["qtd_enviados"] == 4
    assert linha["qtd_entregues"] == 3
    assert linha["qtd_abertos"] == 2
    assert linha["qtd_clicks"] == 1
    assert linha["taxa_abertura"] == pytest.approx(2 / 3)
    assert linha["ctr"] == pytest.approx(1 / 3)
    assert linha["ctor"] == pytest.approx(1 / 2)
    assert linha["dat_referencia"] == pd.Timestamp("2026-01-01")


def test_anexar_gabarito_preserva_todas_as_linhas_do_agregado():
    fato_evento = _fato_evento_fixture()
    agregado = agg.agregar_por_disparo(fato_evento)
    gabarito = pd.DataFrame(
        [
            {
                "id_campanha": 1,
                "id_disparo": 1,
                "flg_anomalia_injetada": True,
                "tipo_anomalia_injetada": "clique_bot",
            }
        ]
    )

    resultado = agg.anexar_gabarito(agregado, gabarito)

    assert len(resultado) == len(agregado)
    assert bool(resultado.loc[0, "flg_anomalia_injetada"]) is True
    assert resultado.loc[0, "tipo_anomalia_injetada"] == "clique_bot"

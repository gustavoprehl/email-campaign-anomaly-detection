import pandas as pd

import generate_data as gd


def test_gerar_dim_marca_tem_ids_e_nomes_unicos():
    rng, _ = gd._rng_setup(42)
    df = gd.gerar_dim_marca(rng)

    assert len(df) == gd.N_MARCAS
    assert df["id_marca"].is_unique
    assert df["nome_marca"].is_unique


def test_gerar_dim_campanha_datas_dentro_do_periodo():
    rng, _ = gd._rng_setup(42)
    df = gd.gerar_dim_campanha(rng)

    assert len(df) == gd.N_CAMPANHAS
    dat_inicio = pd.to_datetime(df["dat_inicio"])
    dat_fim = pd.to_datetime(df["dat_fim"])
    assert (dat_inicio >= gd.PERIODO_INICIO).all()
    assert (dat_fim <= gd.PERIODO_FIM).all()
    assert (dat_fim > dat_inicio).all()


def test_gerar_fato_evento_funil_e_consistente():
    rng, faker = gd._rng_setup(42)
    dim_campanha = gd.gerar_dim_campanha(rng)
    dim_subscriber = gd.gerar_dim_subscriber(rng, faker)
    fato_evento, gabarito = gd.gerar_fato_evento(dim_campanha, dim_subscriber, rng)

    contagens = fato_evento["tipo_evento"].value_counts()
    assert contagens["sent"] > 0
    assert contagens.get("open", 0) <= contagens["sent"]
    assert contagens.get("click", 0) <= contagens.get("open", 0)

    # open nunca pode ocorrer antes do sent do mesmo subscriber+disparo.
    pivot = fato_evento.pivot_table(
        index=["id_campanha", "id_disparo", "id_subscriber"],
        columns="tipo_evento",
        values="dat_evento",
        aggfunc="first",
    ).dropna(subset=["sent", "open"])
    assert (pd.to_datetime(pivot["open"]) >= pd.to_datetime(pivot["sent"])).all()

    # O gabarito deve cobrir exatamente os (id_campanha, id_disparo) gerados.
    disparos_evento = fato_evento[["id_campanha", "id_disparo"]].drop_duplicates()
    assert len(gabarito) == len(disparos_evento)


def test_gerar_dados_e_deterministico_com_mesma_seed():
    rng1, faker1 = gd._rng_setup(42)
    dim_subscriber_1 = gd.gerar_dim_subscriber(rng1, faker1)

    rng2, faker2 = gd._rng_setup(42)
    dim_subscriber_2 = gd.gerar_dim_subscriber(rng2, faker2)

    pd.testing.assert_frame_equal(dim_subscriber_1, dim_subscriber_2)

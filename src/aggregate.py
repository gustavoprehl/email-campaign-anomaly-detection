"""Pipeline de agregacao: transforma fato_evento (grao evento) em fato_agregado_diario
(grao disparo), calculando as metricas de funil usadas pelos detectores de anomalia.

Le de data/raw/ e salva data/processed/fato_agregado_diario.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def carregar_fato_evento() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "fato_evento.csv", parse_dates=["dat_evento"])
    print(f"[fato_evento] {len(df)} eventos carregados.")
    return df


def carregar_gabarito_anomalias() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "gabarito_anomalias.csv")
    print(f"[gabarito_anomalias] {len(df)} disparos carregados.")
    return df


def agregar_por_disparo(fato_evento: pd.DataFrame) -> pd.DataFrame:
    """Agrega eventos por (id_campanha, id_disparo), calculando dat_referencia
    como a data do envio (todo 'sent' de um mesmo disparo ocorre no mesmo dia,
    pela forma como os disparos sao gerados) e as metricas de funil.

    Cada disparo vira exatamente uma linha: opens/clicks sao contados pelo
    id_disparo de origem, independente de quantos dias depois do envio
    ocorreram (a cauda de abertura/clique vai ate 7 dias).
    """
    contagens = (
        fato_evento.groupby(["id_campanha", "id_disparo", "tipo_evento"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["sent", "bounce", "open", "click", "unsubscribe"], fill_value=0)
    )

    dat_referencia = (
        fato_evento.loc[fato_evento["tipo_evento"] == "sent"]
        .groupby(["id_campanha", "id_disparo"])["dat_evento"]
        .min()
        .dt.normalize()
        .rename("dat_referencia")
    )

    agregado = contagens.join(dat_referencia).reset_index()
    agregado = agregado.rename(
        columns={
            "sent": "qtd_enviados",
            "bounce": "qtd_bounce",
            "open": "qtd_abertos",
            "click": "qtd_clicks",
            "unsubscribe": "qtd_unsubscribe",
        }
    )
    agregado["qtd_entregues"] = agregado["qtd_enviados"] - agregado["qtd_bounce"]

    agregado["taxa_abertura"] = agregado["qtd_abertos"] / agregado["qtd_entregues"]
    agregado["ctr"] = agregado["qtd_clicks"] / agregado["qtd_entregues"]
    agregado["ctor"] = agregado["qtd_clicks"] / agregado["qtd_abertos"]

    colunas_finais = [
        "id_campanha",
        "id_disparo",
        "dat_referencia",
        "qtd_enviados",
        "qtd_entregues",
        "qtd_abertos",
        "qtd_clicks",
        "taxa_abertura",
        "ctr",
        "ctor",
    ]
    agregado = agregado[colunas_finais].sort_values(["id_campanha", "id_disparo"]).reset_index(drop=True)

    print(f"[fato_agregado_diario] {len(agregado)} disparos agregados.")
    return agregado


def anexar_gabarito(agregado: pd.DataFrame, gabarito: pd.DataFrame) -> pd.DataFrame:
    """Anexa o gabarito de anomalias ao agregado.

    IMPORTANTE: flg_anomalia_injetada e tipo_anomalia_injetada existem
    SOMENTE para a etapa de avaliacao (evaluate.py). Os detectores
    (z-score e Isolation Forest) nao devem usar essas colunas como
    feature de treino/deteccao, sob pena de vazamento de dados (data leakage).
    """
    return agregado.merge(gabarito, on=["id_campanha", "id_disparo"], how="left")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fato_evento = carregar_fato_evento()
    gabarito = carregar_gabarito_anomalias()

    agregado = agregar_por_disparo(fato_evento)
    agregado_com_gabarito = anexar_gabarito(agregado, gabarito)

    saida = PROCESSED_DIR / "fato_agregado_diario.csv"
    agregado_com_gabarito.to_csv(saida, index=False)
    print(f"\nArquivo salvo em {saida}")


if __name__ == "__main__":
    main()

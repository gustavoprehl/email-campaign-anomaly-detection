"""Gera dados sinteticos de campanhas de e-mail marketing (estilo CRM automotivo multimarca).

Tabelas geradas em data/raw/:
    - dim_marca.csv
    - dim_campanha.csv
    - dim_subscriber.csv

Todo o processo usa seed fixo (SEED = 42) para reprodutibilidade.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

N_MARCAS = 5
N_CAMPANHAS = 30
N_SUBSCRIBERS = 5000
PERIODO_INICIO = datetime(2025, 7, 1)
PERIODO_FIM = datetime(2026, 6, 30)

SEGMENTOS = ["lead_frio", "cliente_ativo", "pos_venda", "engajado"]


def _rng_setup(seed: int = SEED) -> tuple[np.random.Generator, Faker]:
    random.seed(seed)
    np.random.seed(seed)
    faker = Faker("pt_BR")
    Faker.seed(seed)
    return np.random.default_rng(seed), faker


def gerar_dim_marca(rng: np.random.Generator) -> pd.DataFrame:
    nomes = [f"Marca {chr(ord('A') + i)}" for i in range(N_MARCAS)]
    df = pd.DataFrame({"id_marca": range(1, N_MARCAS + 1), "nome_marca": nomes})
    print(f"[dim_marca] {len(df)} marcas geradas.")
    return df


def gerar_dim_campanha(rng: np.random.Generator) -> pd.DataFrame:
    prefixos_jornada = [
        "Reimpactacao_LeadFrio",
        "PosVenda_Revisao",
        "Promocional_Lancamento",
        "Institucional_Marca",
        "PosVenda_Aniversario",
        "Promocional_BlackFriday",
        "Reimpactacao_Carrinho",
    ]
    trimestres = ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]

    registros = []
    dias_periodo = (PERIODO_FIM - PERIODO_INICIO).days
    for id_campanha in range(1, N_CAMPANHAS + 1):
        id_marca = int(rng.integers(1, N_MARCAS + 1))
        prefixo = rng.choice(prefixos_jornada)
        trimestre = rng.choice(trimestres)
        nome_jornada = f"{prefixo}_{trimestre}_{id_campanha:02d}"

        if "Promocional" in prefixo:
            tipo_campanha = "promocional"
        elif "PosVenda" in prefixo:
            tipo_campanha = "pos_venda"
        elif "Reimpactacao" in prefixo:
            tipo_campanha = "reimpactacao"
        else:
            tipo_campanha = "institucional"

        offset_inicio = int(rng.integers(0, dias_periodo - 60))
        dat_inicio = PERIODO_INICIO + timedelta(days=offset_inicio)
        duracao_dias = int(rng.integers(30, 90))
        dat_fim_calc = dat_inicio + timedelta(days=duracao_dias)
        dat_fim = min(dat_fim_calc, PERIODO_FIM)

        registros.append(
            {
                "id_campanha": id_campanha,
                "id_marca": id_marca,
                "nome_jornada": nome_jornada,
                "tipo_campanha": tipo_campanha,
                "dat_inicio": dat_inicio.date().isoformat(),
                "dat_fim": dat_fim.date().isoformat(),
            }
        )

    df = pd.DataFrame(registros)
    print(f"[dim_campanha] {len(df)} campanhas geradas entre {PERIODO_INICIO.date()} e {PERIODO_FIM.date()}.")
    return df


def gerar_dim_subscriber(rng: np.random.Generator, faker: Faker) -> pd.DataFrame:
    registros = []
    dias_periodo = (PERIODO_FIM - PERIODO_INICIO).days
    for id_subscriber in range(1, N_SUBSCRIBERS + 1):
        nome_fake = faker.name()
        email_fake = faker.email()
        id_marca_principal = int(rng.integers(1, N_MARCAS + 1))
        segmento = rng.choice(SEGMENTOS, p=[0.30, 0.35, 0.20, 0.15])
        offset_cadastro = int(rng.integers(0, dias_periodo))
        dat_cadastro = PERIODO_INICIO + timedelta(days=offset_cadastro)

        registros.append(
            {
                "id_subscriber": id_subscriber,
                "nome_fake": nome_fake,
                "email_fake": email_fake,
                "id_marca_principal": id_marca_principal,
                "segmento": segmento,
                "dat_cadastro": dat_cadastro.date().isoformat(),
            }
        )

    df = pd.DataFrame(registros)
    print(f"[dim_subscriber] {len(df)} subscribers gerados.")
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng, faker = _rng_setup(SEED)

    dim_marca = gerar_dim_marca(rng)
    dim_campanha = gerar_dim_campanha(rng)
    dim_subscriber = gerar_dim_subscriber(rng, faker)

    dim_marca.to_csv(RAW_DIR / "dim_marca.csv", index=False)
    dim_campanha.to_csv(RAW_DIR / "dim_campanha.csv", index=False)
    dim_subscriber.to_csv(RAW_DIR / "dim_subscriber.csv", index=False)

    print(f"\nArquivos salvos em {RAW_DIR}")


if __name__ == "__main__":
    main()

"""Gera dados sinteticos de campanhas de e-mail marketing (estilo CRM automotivo multimarca).

Tabelas geradas em data/raw/:
    - dim_marca.csv
    - dim_campanha.csv
    - dim_subscriber.csv
    - fato_evento.csv
    - gabarito_anomalias.csv   (gabarito de avaliacao, NAO usar como feature nos modelos)

Todo o processo usa seed fixo (SEED = 42) para reprodutibilidade.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
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

# Cadencia de disparo dentro de cada campanha. Antes era semanal (7 dias,
# piso de 4 disparos); reduzida para aumentar o numero de disparos por
# campanha sem alterar N_SUBSCRIBERS nem a duracao das campanhas (ver
# _gerar_disparos_da_campanha).
INTERVALO_DISPARO_DIAS = 3
MIN_DISPAROS_POR_CAMPANHA = 8

TIPOS_CAMPANHA = ["promocional", "pos_venda", "reimpactacao", "institucional"]
SEGMENTOS = ["lead_frio", "cliente_ativo", "pos_venda", "engajado"]

# Multiplicadores de taxa de abertura por segmento, relativos a base_abertura.
# Ordem de engajamento: engajado > cliente_ativo > pos_venda > lead_frio.
MULT_ABERTURA_SEGMENTO = {
    "engajado": 1.35,
    "cliente_ativo": 1.10,
    "pos_venda": 0.90,
    "lead_frio": 0.65,
}

# Proporcao de (campanha, disparo) que recebe anomalia injetada.
PCT_DISPAROS_COM_ANOMALIA = 0.18
TIPOS_ANOMALIA = ["queda_deliverability", "clique_bot", "pico_engajamento"]


@dataclass
class DisparoConfig:
    """Parametros normais (sem anomalia) de um disparo, usados como baseline
    antes de aplicar eventual anomalia injetada."""

    taxa_bounce: float
    taxa_abertura_base: float
    ctor_base: float
    taxa_unsub: float


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


def _gerar_disparos_da_campanha(
    campanha: pd.Series, rng: np.random.Generator
) -> list[datetime]:
    """Gera as datas de disparo dentro da janela da campanha, a cada
    INTERVALO_DISPARO_DIAS dias (nao mais semanal).

    Por que: com cadencia semanal e campanhas de 30-90 dias, cada campanha
    tinha so 4-12 disparos, e os 3 primeiros de cada uma ficavam sem
    historico suficiente para o z-score (MIN_DISPAROS_HISTORICO=3) — uma
    perda estrutural de recall que nao tem a ver com a qualidade do metodo.
    Aumentar a frequencia de disparo (sem mexer no numero de subscribers
    nem na duracao da campanha) da mais historico por campanha, diluindo
    essa perda estrutural."""
    dat_inicio = datetime.fromisoformat(campanha["dat_inicio"])
    dat_fim = datetime.fromisoformat(campanha["dat_fim"])
    duracao_dias = max((dat_fim - dat_inicio).days, INTERVALO_DISPARO_DIAS)
    n_disparos = max(duracao_dias // INTERVALO_DISPARO_DIAS, MIN_DISPAROS_POR_CAMPANHA)

    disparos = [dat_inicio + timedelta(days=INTERVALO_DISPARO_DIAS * i) for i in range(n_disparos)]
    return disparos


def _timestamp_pos_evento(dat_base: datetime, rng: np.random.Generator) -> datetime:
    """Timestamp posterior ao envio, com maioria nas primeiras 48h e long tail ate 7 dias.

    Usa mistura: 80% das ocorrencias em uma exponencial de escala curta (horas),
    20% em uma cauda mais longa (dias), truncada em 7 dias.
    """
    if rng.random() < 0.8:
        horas = rng.exponential(scale=10.0)
        horas = min(horas, 48.0)
    else:
        horas = 48.0 + rng.exponential(scale=36.0)
        horas = min(horas, 24.0 * 7)
    return dat_base + timedelta(hours=float(horas))


def gerar_fato_evento(
    dim_campanha: pd.DataFrame,
    dim_subscriber: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera fato_evento com logica de funil sent -> bounce/open -> click -> unsubscribe,
    e injeta anomalias em uma fracao dos (id_campanha, id_disparo).

    Retorna (fato_evento, gabarito_anomalias).
    """
    eventos: list[dict] = []
    gabarito: list[dict] = []
    id_evento_seq = 1

    subs_por_marca: dict[int, np.ndarray] = {
        id_marca: dim_subscriber.loc[dim_subscriber["id_marca_principal"] == id_marca, "id_subscriber"].to_numpy()
        for id_marca in dim_subscriber["id_marca_principal"].unique()
    }
    segmento_por_subscriber = dim_subscriber.set_index("id_subscriber")["segmento"].to_dict()

    for _, campanha in dim_campanha.iterrows():
        id_campanha = int(campanha["id_campanha"])
        id_marca = int(campanha["id_marca"])
        candidatos = subs_por_marca.get(id_marca, np.array([], dtype=int))
        if len(candidatos) == 0:
            continue

        disparos = _gerar_disparos_da_campanha(campanha, rng)

        for id_disparo, dat_disparo in enumerate(disparos, start=1):
            tamanho_lista = int(rng.integers(int(len(candidatos) * 0.3), int(len(candidatos) * 0.6) + 1))
            tamanho_lista = max(tamanho_lista, 50)
            destinatarios = rng.choice(candidatos, size=min(tamanho_lista, len(candidatos)), replace=False)

            # --- baseline (sem anomalia) ---
            taxa_bounce = rng.uniform(0.02, 0.05)
            base_abertura = rng.uniform(0.18, 0.30)
            ctor_base = rng.uniform(0.15, 0.25)
            taxa_unsub = rng.uniform(0.001, 0.003)

            # --- decide se este disparo recebe anomalia injetada ---
            tem_anomalia = rng.random() < PCT_DISPAROS_COM_ANOMALIA
            tipo_anomalia = None
            mult_abertura_anomalia = 1.0
            mult_ctor_anomalia = 1.0

            if tem_anomalia:
                tipo_anomalia = rng.choice(TIPOS_ANOMALIA)
                if tipo_anomalia == "queda_deliverability":
                    # Abertura cai para 20-40% do valor normal do disparo.
                    mult_abertura_anomalia = rng.uniform(0.20, 0.40)
                elif tipo_anomalia == "clique_bot":
                    # CTR sobe 3-5x sem aumento proporcional de abertura -> mexe no CTOR.
                    mult_ctor_anomalia = rng.uniform(3.0, 5.0)
                elif tipo_anomalia == "pico_engajamento":
                    # Aumento organico simultaneo de abertura E clique (falso positivo intencional).
                    mult_abertura_anomalia = rng.uniform(1.5, 2.5)
                    mult_ctor_anomalia = rng.uniform(1.5, 2.5)

            gabarito.append(
                {
                    "id_campanha": id_campanha,
                    "id_disparo": id_disparo,
                    "flg_anomalia_injetada": bool(tem_anomalia),
                    "tipo_anomalia_injetada": tipo_anomalia,
                }
            )

            for id_subscriber in destinatarios:
                segmento = segmento_por_subscriber[id_subscriber]
                mult_segmento = MULT_ABERTURA_SEGMENTO[segmento]

                dat_sent = dat_disparo + timedelta(
                    hours=float(rng.uniform(0, 6))
                )
                eventos.append(
                    {
                        "id_evento": id_evento_seq,
                        "id_campanha": id_campanha,
                        "id_subscriber": int(id_subscriber),
                        "tipo_evento": "sent",
                        "dat_evento": dat_sent.isoformat(),
                        "id_disparo": id_disparo,
                    }
                )
                id_evento_seq += 1

                if rng.random() < taxa_bounce:
                    eventos.append(
                        {
                            "id_evento": id_evento_seq,
                            "id_campanha": id_campanha,
                            "id_subscriber": int(id_subscriber),
                            "tipo_evento": "bounce",
                            "dat_evento": _timestamp_pos_evento(dat_sent, rng).isoformat(),
                            "id_disparo": id_disparo,
                        }
                    )
                    id_evento_seq += 1
                    continue  # entregue = sent - bounce; quem sofreu bounce nao abre.

                prob_abertura = min(base_abertura * mult_segmento * mult_abertura_anomalia, 0.98)
                abriu = rng.random() < prob_abertura
                if abriu:
                    dat_open = _timestamp_pos_evento(dat_sent, rng)
                    eventos.append(
                        {
                            "id_evento": id_evento_seq,
                            "id_campanha": id_campanha,
                            "id_subscriber": int(id_subscriber),
                            "tipo_evento": "open",
                            "dat_evento": dat_open.isoformat(),
                            "id_disparo": id_disparo,
                        }
                    )
                    id_evento_seq += 1

                    prob_clique = min(ctor_base * mult_ctor_anomalia, 0.95)
                    if rng.random() < prob_clique:
                        dat_click = dat_open + timedelta(hours=float(rng.exponential(scale=2.0)))
                        eventos.append(
                            {
                                "id_evento": id_evento_seq,
                                "id_campanha": id_campanha,
                                "id_subscriber": int(id_subscriber),
                                "tipo_evento": "click",
                                "dat_evento": dat_click.isoformat(),
                                "id_disparo": id_disparo,
                            }
                        )
                        id_evento_seq += 1

                if rng.random() < taxa_unsub:
                    eventos.append(
                        {
                            "id_evento": id_evento_seq,
                            "id_campanha": id_campanha,
                            "id_subscriber": int(id_subscriber),
                            "tipo_evento": "unsubscribe",
                            "dat_evento": _timestamp_pos_evento(dat_sent, rng).isoformat(),
                            "id_disparo": id_disparo,
                        }
                    )
                    id_evento_seq += 1

    df_eventos = pd.DataFrame(eventos)
    df_gabarito = pd.DataFrame(gabarito)
    print(f"[fato_evento] {len(df_eventos)} eventos gerados.")
    print(
        f"[gabarito_anomalias] {len(df_gabarito)} disparos avaliados, "
        f"{df_gabarito['flg_anomalia_injetada'].sum()} com anomalia injetada "
        f"({df_gabarito['flg_anomalia_injetada'].mean():.1%})."
    )
    print(df_gabarito.loc[df_gabarito["flg_anomalia_injetada"], "tipo_anomalia_injetada"].value_counts())
    return df_eventos, df_gabarito


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng, faker = _rng_setup(SEED)

    dim_marca = gerar_dim_marca(rng)
    dim_campanha = gerar_dim_campanha(rng)
    dim_subscriber = gerar_dim_subscriber(rng, faker)
    fato_evento, gabarito_anomalias = gerar_fato_evento(dim_campanha, dim_subscriber, rng)

    dim_marca.to_csv(RAW_DIR / "dim_marca.csv", index=False)
    dim_campanha.to_csv(RAW_DIR / "dim_campanha.csv", index=False)
    dim_subscriber.to_csv(RAW_DIR / "dim_subscriber.csv", index=False)
    fato_evento.to_csv(RAW_DIR / "fato_evento.csv", index=False)
    gabarito_anomalias.to_csv(RAW_DIR / "gabarito_anomalias.csv", index=False)

    print(f"\nArquivos salvos em {RAW_DIR}")


if __name__ == "__main__":
    main()

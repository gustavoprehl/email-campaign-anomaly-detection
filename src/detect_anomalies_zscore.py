"""Deteccao de anomalias via z-score movel, calculado POR CAMPANHA.

Por que por campanha e nao globalmente: cada campanha tem um baseline de
engajamento proprio (tipo de campanha, marca, mix de segmentos da lista).
Comparar um disparo de reimpactacao de lead frio contra a media global de
todas as campanhas (que inclui campanhas de clientes engajados) geraria
falsos positivos/negativos sistematicos. Normalizando dentro do historico
da propria campanha, o z-score reflete desvio em relacao ao comportamento
recente DAQUELA jornada especifica.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

METRICAS = ["taxa_abertura", "ctr", "ctor"]
JANELA_DISPAROS = 6  # tamanho maximo da janela movel (5-7 disparos anteriores)
MIN_DISPAROS_HISTORICO = 3  # minimo de disparos anteriores para calcular z-score
Z_THRESHOLD_PADRAO = 2.5


def carregar_agregado() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "fato_agregado_diario.csv", parse_dates=["dat_referencia"])
    df = df.sort_values(["id_campanha", "id_disparo"]).reset_index(drop=True)
    print(f"[fato_agregado_diario] {len(df)} disparos carregados.")
    return df


def calcular_zscore_movel(
    df: pd.DataFrame,
    metrica: str,
    janela: int = JANELA_DISPAROS,
    min_periodos: int = MIN_DISPAROS_HISTORICO,
) -> pd.Series:
    """Z-score de `metrica` no disparo atual vs. media/desvio movel dos
    disparos ANTERIORES da mesma campanha (shift(1) evita usar o proprio
    valor atual no calculo do baseline, o que inflaria artificialmente o
    baseline em disparos anomalos e mascararia a anomalia)."""
    valores_anteriores = df.groupby("id_campanha")[metrica].shift(1)
    media_movel = valores_anteriores.groupby(df["id_campanha"]).transform(
        lambda s: s.rolling(janela, min_periods=min_periodos).mean()
    )
    std_movel = valores_anteriores.groupby(df["id_campanha"]).transform(
        lambda s: s.rolling(janela, min_periods=min_periodos).std()
    )
    z = (df[metrica] - media_movel) / std_movel
    return z


def detectar_anomalias_zscore(
    df: pd.DataFrame,
    z_threshold: float = Z_THRESHOLD_PADRAO,
) -> pd.DataFrame:
    """Roda o z-score movel para cada metrica e retorna formato longo:
    (id_campanha, id_disparo, dat_referencia, metrica_avaliada,
    score_anomalia, flg_anomalia_detectada, metodo)."""
    resultados = []
    for metrica in METRICAS:
        z = calcular_zscore_movel(df, metrica)
        parcial = pd.DataFrame(
            {
                "id_campanha": df["id_campanha"],
                "id_disparo": df["id_disparo"],
                "dat_referencia": df["dat_referencia"],
                "metrica_avaliada": metrica,
                "score_anomalia": z,
                "flg_anomalia_detectada": z.abs() > z_threshold,
                "metodo": "zscore",
            }
        )
        # Disparos sem historico suficiente (std NaN ou 0) nao tem z-score valido.
        parcial["flg_anomalia_detectada"] = parcial["flg_anomalia_detectada"].fillna(False)
        resultados.append(parcial)

    resultado = pd.concat(resultados, ignore_index=True)
    n_disparos_com_anomalia = resultado.loc[resultado["flg_anomalia_detectada"], "id_disparo"].count()
    print(
        f"[zscore] {len(resultado)} linhas avaliadas ({len(METRICAS)} metricas x {df.shape[0]} disparos), "
        f"{n_disparos_com_anomalia} marcacoes de anomalia (|z| > {z_threshold})."
    )
    print(resultado.loc[resultado["flg_anomalia_detectada"], "metrica_avaliada"].value_counts())
    return resultado


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = carregar_agregado()
    resultado = detectar_anomalias_zscore(df)

    saida = OUTPUTS_DIR / "anomalias_zscore.csv"
    resultado.to_csv(saida, index=False)
    print(f"\nArquivo salvo em {saida}")


if __name__ == "__main__":
    main()

"""Baixa IPCA, taxa de desocupação e dívida pública/PIB do Banco Central (SGS) desde 2010,
calcula as correlações de Pearson e Spearman entre dívida/PIB e IPCA 12 meses e salva tudo em Excel."""
import os
import sys

import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr

# Pasta onde o Excel será salvo (pode passar outra como argumento: python baixar_dados.py "C:\outra\pasta")
PASTA_PADRAO = r"C:\Users\casem\OneDrive\Análises Estatísticas\Economia\AM"
DATA_INICIAL = "01/01/2010"

# Séries do Banco Central (SGS)
SERIES = {
    433: "IPCA_mensal_%",      # IPCA - variação mensal
    13522: "IPCA_12m_%",       # IPCA - acumulado em 12 meses
    24369: "Desocupacao_%",    # Taxa de desocupação - PNAD Contínua (trimestre móvel)
    13762: "Divida_Bruta_PIB_%",  # Dívida Bruta do Governo Geral (% do PIB)
}


def baixar_serie(cod: int, nome: str) -> pd.DataFrame:
    url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{cod}/dados"
           f"?formato=json&dataInicial={DATA_INICIAL}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    d = pd.DataFrame(resp.json())
    d["data"] = pd.to_datetime(d["data"], dayfirst=True)
    d[nome] = d["valor"].astype(float)
    print(f"{nome}: {len(d)} meses (até {d['data'].max():%m/%Y})")
    return d.set_index("data")[[nome]]


def main() -> None:
    pasta = sys.argv[1] if len(sys.argv) > 1 else PASTA_PADRAO
    df = pd.concat([baixar_serie(c, n) for c, n in SERIES.items()], axis=1)
    df = df.sort_index().reset_index()
    df["data"] = df["data"].dt.strftime("%m/%Y")

    correl = correlacao(df, "Divida_Bruta_PIB_%", "IPCA_12m_%")

    os.makedirs(pasta, exist_ok=True)
    arquivo = os.path.join(pasta, "Inflacao_Desocupacao_Brasil.xlsx")
    with pd.ExcelWriter(arquivo) as w:
        df.to_excel(w, sheet_name="Dados", index=False)
        correl.to_excel(w, sheet_name="Correlacao", index=False)
    print("Salvo em:", arquivo)


def classificar(r: float, p: float) -> str:
    forca = "forte" if abs(r) >= 0.7 else "moderada" if abs(r) >= 0.4 else "fraca"
    sentido = "positiva" if r > 0 else "negativa"
    signif = "significativa" if p < 0.05 else "não significativa"
    return f"Correlação {forca} e {sentido}, estatisticamente {signif} a 5%"


def correlacao(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Correlações de Pearson e Spearman entre duas colunas, usando só os meses em que as duas têm dado."""
    pares = df[["data", x, y]].dropna()
    r_p, p_p = pearsonr(pares[x], pares[y])
    r_s, p_s = spearmanr(pares[x], pares[y])
    periodo = f"{pares['data'].iloc[0]} a {pares['data'].iloc[-1]}"

    print(f"\n{x} x {y} | {periodo} | n = {len(pares)}")
    print(f"Pearson : r   = {r_p:.4f} | p-valor = {p_p:.4g} | {classificar(r_p, p_p)}")
    print(f"Spearman: rho = {r_s:.4f} | p-valor = {p_s:.4g} | {classificar(r_s, p_s)}")

    return pd.DataFrame({
        "Método": ["Pearson", "Spearman"],
        "Variável X": [x, x],
        "Variável Y": [y, y],
        "Período": [periodo, periodo],
        "Nº de meses (n)": [len(pares), len(pares)],
        "Coeficiente": [round(r_p, 4), round(r_s, 4)],
        "p-valor": [p_p, p_s],
        "Interpretação": [classificar(r_p, p_p), classificar(r_s, p_s)],
    })


if __name__ == "__main__":
    main()
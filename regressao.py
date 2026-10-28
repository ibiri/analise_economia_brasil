"""Regressão linear múltipla: Desocupacao_% ~ IPCA_12m_% + Divida_Bruta_PIB_%.

Lê os dados de Inflacao_Desocupacao_Brasil.xlsx (gerado por eco_am.py) e ajusta três modelos por MQO:

1. Nível (mensal): as séries como estão.
2. Diferenças trimestrais: variação de um trimestre para o outro, usando só o último mês de cada
   trimestre (03, 06, 09, 12). A desocupação da PNAD é média móvel de 3 meses, então meses seguidos
   se sobrepõem; pegar um mês por trimestre evita isso. Diferenciar remove a tendência das séries,
   que torna a regressão em nível espúria.
3. Diferenças + sazonalidade: o modelo 2 com dummies de trimestre (base = 1º trimestre).

Para cada modelo testa os pressupostos e salva tudo em Regressao_Desocupacao.xlsx; os gráficos de
diagnóstico vão para regressao_diagnostico_<nivel|dif|difsaz>.png."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller

PASTA = os.path.dirname(os.path.abspath(__file__))
Y = "Desocupacao_%"
X = ["IPCA_12m_%", "Divida_Bruta_PIB_%"]
ALFA = 0.05
# Valor crítico de Engle-Granger (MacKinnon) para 2 regressoras + constante a 5%
EG_CRITICO = -3.74


def conclusao(p: float, ok: str, violado: str) -> str:
    return violado if p < ALFA else ok


def ajustar(df: pd.DataFrame, nome: str, lags_autocorr: int, periodicidade: str, em_nivel: bool,
            controles: list[str] | None = None) -> dict:
    """Ajusta o MQO, roda os testes de pressupostos e devolve as tabelas e os dados para os gráficos.
    `controles` são regressoras extras (ex.: dummies de trimestre) que entram no modelo mas não nos
    testes de VIF e estacionariedade."""
    controles = controles or []
    periodo = f"{df['data'].iloc[0]} a {df['data'].iloc[-1]}"
    n = len(df)
    y = df[Y]
    Xc = sm.add_constant(df[X + controles])
    modelo = sm.OLS(y, Xc).fit()
    # Erros-padrão robustos a heterocedasticidade e autocorrelação (Newey-West)
    lag_hac = int(np.floor(4 * (n / 100) ** (2 / 9)))
    modelo_hac = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lag_hac})
    resid = modelo.resid

    print(f"\n{'=' * 30} Modelo: {nome} | {periodo} | n = {n} {periodicidade} {'=' * 30}")
    print(modelo.summary())

    coef = pd.DataFrame({
        "Variável": modelo.params.index,
        "Coeficiente": modelo.params.values,
        "Erro-padrão (MQO)": modelo.bse.values,
        "t (MQO)": modelo.tvalues.values,
        "p-valor (MQO)": modelo.pvalues.values,
        f"Erro-padrão (Newey-West, {lag_hac} lags)": modelo_hac.bse.values,
        "z (Newey-West)": modelo_hac.tvalues.values,
        "p-valor (Newey-West)": modelo_hac.pvalues.values,
    })

    ajuste = {
        "Período": periodo, f"n ({periodicidade})": n, "R²": modelo.rsquared,
        "R² ajustado": modelo.rsquared_adj, "F (MQO)": modelo.fvalue, "p-valor F (MQO)": modelo.f_pvalue,
        "AIC": modelo.aic, "BIC": modelo.bic, "Desvio-padrão dos resíduos": np.sqrt(modelo.scale),
    }

    testes = []

    def add(pressuposto, teste, estat, p, h0, conc):
        testes.append({"Pressuposto": pressuposto, "Teste": teste, "Estatística": estat,
                       "p-valor": p, "H0": h0, "Conclusão (5%)": conc})

    reset = linear_reset(modelo, power=3, use_f=True)
    add("Linearidade / especificação", "RESET de Ramsey (ŷ² e ŷ³)", float(reset.fvalue), float(reset.pvalue),
        "Forma funcional linear adequada",
        conclusao(reset.pvalue, "OK: sem evidência de má especificação",
                  "Violado: forma funcional inadequada ou variável omitida"))

    sw = st.shapiro(resid)
    add("Normalidade dos resíduos", "Shapiro-Wilk", sw.statistic, sw.pvalue, "Resíduos normais",
        conclusao(sw.pvalue, "OK: resíduos normais", "Violado: resíduos não normais"))
    jb, jb_p, assim, curt = jarque_bera(resid)
    add("Normalidade dos resíduos", f"Jarque-Bera (assimetria {assim:.2f}, curtose {curt:.2f})", jb, jb_p,
        "Resíduos normais", conclusao(jb_p, "OK: resíduos normais", "Violado: resíduos não normais"))

    bp = het_breuschpagan(resid, Xc)
    add("Homocedasticidade", "Breusch-Pagan", bp[0], bp[1], "Variância constante",
        conclusao(bp[1], "OK: variância constante", "Violado: heterocedasticidade"))
    wh = het_white(resid, Xc)
    add("Homocedasticidade", "White", wh[0], wh[1], "Variância constante",
        conclusao(wh[1], "OK: variância constante", "Violado: heterocedasticidade"))

    dw = durbin_watson(resid)
    add("Independência dos resíduos", "Durbin-Watson", dw, np.nan, "Sem autocorrelação de 1ª ordem (DW ≈ 2)",
        "OK: DW perto de 2" if 1.5 <= dw <= 2.5
        else "Violado: autocorrelação positiva" if dw < 1.5 else "Violado: autocorrelação negativa")
    bg = acorr_breusch_godfrey(modelo, nlags=lags_autocorr)
    add("Independência dos resíduos", f"Breusch-Godfrey ({lags_autocorr} lags = 1 ano)", bg[0], bg[1],
        "Sem autocorrelação", conclusao(bg[1], "OK: sem autocorrelação", "Violado: resíduos autocorrelacionados"))

    for i, col in enumerate(X, start=1):
        vif = variance_inflation_factor(Xc.values, i)
        add("Ausência de multicolinearidade", f"VIF – {col}", vif, np.nan, "VIF < 5",
            "Multicolinearidade grave" if vif >= 10 else "Multicolinearidade moderada" if vif >= 5
            else "OK: sem multicolinearidade relevante")

    cook = modelo.get_influence().cooks_distance[0]
    lim_cook = 4 / n
    n_infl = int((cook > lim_cook).sum())
    add("Observações influentes", f"Distância de Cook > 4/n ({lim_cook:.3f})", n_infl, np.nan,
        "Nenhum ponto domina o ajuste",
        f"{n_infl} períodos acima do limite (máx. Cook = {cook.max():.3f} em {df['data'].iloc[cook.argmax()]})")

    # Regressão entre séries não estacionárias pode ser espúria
    sufixo = "nível" if em_nivel else "diferença"
    for col in [Y, *X]:
        adf = adfuller(df[col], autolag="AIC")
        add("Estacionariedade das séries", f"ADF – {col} ({sufixo})", adf[0], adf[1],
            "Série tem raiz unitária (não estacionária)",
            conclusao(adf[1], "Violado: não estacionária", "OK: estacionária"))
    if em_nivel:
        adf_res = adfuller(resid, autolag="AIC")
        add("Estacionariedade das séries", "Engle-Granger: ADF nos resíduos (crítico 5% ≈ -3,74)",
            adf_res[0], np.nan, "Sem cointegração (regressão pode ser espúria)",
            "OK: cointegradas, relação de longo prazo válida" if adf_res[0] < EG_CRITICO
            else "Violado: não cointegradas, risco de regressão espúria")

    testes = pd.DataFrame(testes)
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print("\n", testes.to_string(index=False))
        print("\n", coef.to_string(index=False))

    residuos = pd.DataFrame({"data": df["data"], "Observado": y, "Ajustado": modelo.fittedvalues,
                             "Resíduo": resid, "Cook": cook})
    formula = f"{Y} ~ {' + '.join(X)}" + (" + dummies de trimestre" if controles else "")
    return {"nome": nome, "formula": formula, "periodo": periodo, "n": n, "coef": coef, "ajuste": ajuste,
            "testes": testes, "residuos": residuos, "lags_acf": 2 * lags_autocorr}


def graficos(r: dict, arquivo: str, rotulo_y: str) -> None:
    res = r["residuos"]
    resid, ajust = res["Resíduo"], res["Ajustado"]
    datas = pd.to_datetime(res["data"], format="%m/%Y")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Diagnóstico – {r['nome']}: {r['formula']}  ({r['periodo']}, n = {r['n']})", fontsize=13)

    ax[0, 0].scatter(ajust, resid, s=14, alpha=0.7)
    ax[0, 0].axhline(0, color="gray", lw=1)
    ax[0, 0].set(title="Resíduos x valores ajustados", xlabel="Ajustado", ylabel="Resíduo")

    sm.qqplot(resid, line="s", ax=ax[0, 1], markersize=3)
    ax[0, 1].set_title("Q-Q plot dos resíduos")

    ax[0, 2].hist(resid, bins=min(25, r["n"] // 3), density=True, alpha=0.7)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax[0, 2].plot(xs, st.norm.pdf(xs, resid.mean(), resid.std()), color="C3")
    ax[0, 2].set(title="Histograma dos resíduos", xlabel="Resíduo")

    ax[1, 0].plot(datas, resid, lw=1)
    ax[1, 0].axhline(0, color="gray", lw=1)
    ax[1, 0].set(title="Resíduos ao longo do tempo", ylabel="Resíduo")

    sm.graphics.tsa.plot_acf(resid, lags=min(r["lags_acf"], r["n"] // 2 - 1), ax=ax[1, 1])
    ax[1, 1].set_title("Autocorrelação dos resíduos (ACF)")

    ax[1, 2].plot(datas, res["Observado"], label="Observado", lw=1.5)
    ax[1, 2].plot(datas, ajust, label="Ajustado", lw=1.5)
    ax[1, 2].set(title=f"{rotulo_y}: observado x ajustado", ylabel="p.p." if "Variação" in rotulo_y else "%")
    ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig(arquivo, dpi=120)
    plt.close(fig)


def main() -> None:
    dados = pd.read_excel(os.path.join(PASTA, "Inflacao_Desocupacao_Brasil.xlsx"), sheet_name="Dados")
    dados = dados[["data", Y, *X]].dropna().reset_index(drop=True)

    # Modelo 1: nível mensal
    nivel = ajustar(dados, "Nível (mensal)", lags_autocorr=12, periodicidade="meses", em_nivel=True)

    # Modelo 2: diferenças trimestrais, só com o último mês de cada trimestre
    mes = dados["data"].str[:2].astype(int)
    trim = dados[mes % 3 == 0].reset_index(drop=True)
    dif = trim[[Y, *X]].diff()
    dif.insert(0, "data", trim["data"])
    dif = dif.dropna().reset_index(drop=True)
    difer = ajustar(dif, "Diferenças trimestrais", lags_autocorr=4, periodicidade="trimestres", em_nivel=False)

    # Modelo 3: diferenças trimestrais + dummies de trimestre (base = 1º trimestre), para tirar a
    # sazonalidade da desocupação (ela sobe todo 1º trimestre, depois das contratações de fim de ano)
    trimestre = dif["data"].str[:2].astype(int) // 3
    dummies = []
    for t in (2, 3, 4):
        dummies.append(f"Trimestre_{t}")
        dif[f"Trimestre_{t}"] = (trimestre == t).astype(float)
    sazonal = ajustar(dif, "Diferenças + sazonalidade", lags_autocorr=4, periodicidade="trimestres",
                      em_nivel=False, controles=dummies)

    modelos = [(nivel, "Nivel", "Desocupação"),
               (difer, "Dif", "Variação da desocupação"),
               (sazonal, "DifSaz", "Variação da desocupação")]
    for r, s, rotulo in modelos:
        graficos(r, os.path.join(PASTA, f"regressao_diagnostico_{s.lower()}.png"), rotulo)

    # Comparação lado a lado
    chaves = list(nivel["ajuste"].keys())
    comp_ajuste = pd.DataFrame({"Medida": chaves[:1] + ["n"] + chaves[2:]})
    # Os testes saem na mesma ordem em todos os modelos; só o de nível tem Engle-Granger no fim
    comp_testes = nivel["testes"][["Pressuposto", "Teste"]].copy()
    comp_testes["Teste"] = (comp_testes["Teste"].str.replace(r" \((nível|assimetria.*|12 lags.*)\)$", "", regex=True)
                            .str.replace(r" > 4/n.*", " > 4/n", regex=True))
    for r, _, _ in modelos:
        comp_ajuste[r["nome"]] = list(r["ajuste"].values())
        comp_testes[r["nome"]] = r["testes"]["Conclusão (5%)"].reindex(comp_testes.index).fillna("Não se aplica")

    arquivo = os.path.join(PASTA, "Regressao_Desocupacao.xlsx")
    with pd.ExcelWriter(arquivo) as w:
        comp_ajuste.to_excel(w, sheet_name="Comparacao_Ajuste", index=False)
        comp_testes.to_excel(w, sheet_name="Comparacao_Pressupostos", index=False)
        for r, s, _ in modelos:
            r["coef"].to_excel(w, sheet_name=f"{s}_Coeficientes", index=False)
            r["testes"].to_excel(w, sheet_name=f"{s}_Pressupostos", index=False)
            r["residuos"].to_excel(w, sheet_name=f"{s}_Residuos", index=False)
    print("\nSalvo em:", arquivo)
    print("Gráficos:", ", ".join(f"regressao_diagnostico_{s.lower()}.png" for _, s, _ in modelos))


if __name__ == "__main__":
    main()

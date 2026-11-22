"""Modèles « toutes choses égales par ailleurs » de la sanction électorale.

- ``table_modele`` : sortants têtes de liste aux deux élections, dans des
  communes disputées les deux fois (champ du notebook 02), avec les
  caractéristiques des communes (src/covariables.py) ;
- ``ols`` / ``logit`` : régressions avec erreurs-types groupées par commune
  (une commune apparaît en 2020 et en 2026) ;
- ``appariement`` : appariement sur le score de propension (k plus proches
  voisins, avec caliper), effet moyen sur les traités et équilibre des
  covariables avant / après.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import affaires as A  # noqa: E402
import analyse as An  # noqa: E402
import covariables as Cov  # noqa: E402
import panel as P  # noqa: E402

TRAITEMENTS = ["affaire_connue", "affaire_mandat", "affaire_apres", "personnalite_affaires",
               "affaire_probite_connue", "condamne_avant"]

# Contrôles communs à tous les modèles. Le score précédent capte le retour vers
# la moyenne (un sortant très bien élu a plus à perdre).
CONTROLES = ("pct_voix_exp_prec + log_inscrits + n_listes + C(annee) + C(bloc_prec) + pct_diplome_sup"
             " + pct_cadres + pct_retraites + pct_extreme_droite_pres + participation_pres")
COVARIABLES_PS = ["pct_voix_exp_prec", "log_inscrits", "n_listes", "pct_diplome_sup", "pct_cadres",
                  "pct_retraites", "pct_extreme_droite_pres", "participation_pres", "election_2026"]


def table_modele() -> pd.DataFrame:
    """Une ligne par sortant tête de liste aux deux élections, commune disputée les deux fois."""
    p = pd.read_parquet(P.PROCESSED_DIR / "panel_affaires.parquet") \
        if (P.PROCESSED_DIR / "panel_affaires.parquet").exists() else A.build()[0]
    p = p[~p["perimetre_modifie"]].merge(Cov.load_covariables(), on=["annee", "code_insee"], how="left")
    d = p[(p["n_listes"] > 1) & p["sortant_tete_liste"].fillna(False) & p["tete_liste_prec"]].copy()
    d = d[~d["affaire_ambigue"]]  # affaire dont on ne sait pas si elle précède le vote
    d = d.dropna(subset=["delta_pct_voix_exp", "pct_diplome_sup", "pct_extreme_droite_pres"])
    d["gagne"] = d["liste_sortant_gagnante"].astype(float)
    d["log_inscrits"] = np.log(d["inscrits"])
    d["taille"] = An.tranche_taille(d["inscrits"])
    d["election_2026"] = (d["annee"] == 2026).astype(int)
    d["bloc_prec"] = d["bloc_prec"].fillna("non_nuance").astype(str)
    for c in TRAITEMENTS:
        d[c] = d[c].astype(int)
    d["affaire_avant_mandat"] = d["affaire_connue"] - d["affaire_mandat"]
    vote = pd.to_datetime(d["annee"].map(P.DATES_PREMIER_TOUR))
    d["delai_annees"] = (vote - d["premiere_affaire"]).dt.days / 365.25
    return d.reset_index(drop=True)


def _groupes(data: pd.DataFrame) -> np.ndarray:
    return pd.factorize(data["code_insee"])[0]


def ols(formule: str, data: pd.DataFrame):
    return smf.ols(formule, data=data).fit(cov_type="cluster", cov_kwds={"groups": _groupes(data)})


def logit(formule: str, data: pd.DataFrame):
    return smf.logit(formule, data=data).fit(cov_type="cluster", cov_kwds={"groups": _groupes(data)}, disp=0, maxiter=300)


def coefficients(res, motif: str = "affaire|personnalite") -> pd.DataFrame:
    """Coefficients d'intérêt avec IC 95 % ; pour un logit, effets marginaux moyens."""
    if hasattr(res, "get_margeff"):
        m = res.get_margeff().summary_frame()
        t = pd.DataFrame({"coef": m["dy/dx"], "ic_bas": m["Conf. Int. Low"],
                          "ic_haut": m["Cont. Int. Hi."], "p": m["Pr(>|z|)"]})
    else:
        ic = res.conf_int()
        t = pd.DataFrame({"coef": res.params, "ic_bas": ic[0], "ic_haut": ic[1], "p": res.pvalues})
    return t[t.index.str.contains(motif)]


# --------------------------------------------------------------------------
# Appariement sur le score de propension
# --------------------------------------------------------------------------
def _design(d: pd.DataFrame) -> np.ndarray:
    x = pd.get_dummies(d[COVARIABLES_PS + ["bloc_prec"]], columns=["bloc_prec"], drop_first=True)
    return StandardScaler().fit_transform(x.astype(float))


def ecart_standardise(x: pd.Series, t: np.ndarray, poids: np.ndarray | None = None) -> float:
    """Différence des moyennes traités - témoins en écarts-types (équilibre si |.| < 0,1)."""
    xt, xc = x[t], x[~t]
    mc = np.average(xc, weights=poids) if poids is not None else xc.mean()
    return float((xt.mean() - mc) / np.sqrt((xt.var() + xc.var()) / 2))


def appariement(d: pd.DataFrame, traitement: str, resultats: list[str], k: int = 5,
                caliper: float = 0.2, n_boot: int = 1000, seed: int = 0) -> dict:
    """Effet moyen sur les traités par appariement sur le logit du score de propension.

    Chaque traité est comparé à ses ``k`` témoins les plus proches ; il est écarté
    si l'un d'eux est à plus de ``caliper`` écart-type du logit (hors support
    commun). IC 95 % par bootstrap des écarts appariés (score de propension tenu
    fixe, donc un peu trop étroit).
    """
    t = d[traitement].to_numpy() == 1
    ps = LogisticRegression(max_iter=5000).fit(_design(d), t).predict_proba(_design(d))[:, 1]
    lp = np.log(ps / (1 - ps))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(lp[~t].reshape(-1, 1)).kneighbors(lp[t].reshape(-1, 1))
    dans_support = dist.max(axis=1) <= caliper * lp.std()

    rng = np.random.default_rng(seed)
    effets = {}
    for y in resultats:
        yt = d.loc[t, y].to_numpy()[dans_support]
        yc = d.loc[~t, y].to_numpy()[idx[dans_support]].mean(axis=1)
        ecarts = yt - yc
        boot = [rng.choice(ecarts, size=len(ecarts)).mean() for _ in range(n_boot)]
        effets[y] = {"effet": ecarts.mean(), "ic_bas": np.percentile(boot, 2.5),
                     "ic_haut": np.percentile(boot, 97.5)}

    # Équilibre : témoins pondérés par le nombre de fois où ils servent.
    poids = np.bincount(idx[dans_support].ravel(), minlength=(~t).sum()).astype(float)
    t_ok = t.copy()
    t_ok[np.flatnonzero(t)[~dans_support]] = False
    equilibre = pd.DataFrame({
        c: {"avant": ecart_standardise(d[c].astype(float).to_numpy(), t),
            "apres": ecart_standardise(np.r_[d[c].to_numpy()[t_ok], d[c].to_numpy()[~t]].astype(float),
                                       np.r_[np.ones(t_ok.sum(), bool), np.zeros((~t).sum(), bool)],
                                       poids if poids.sum() else None)}
        for c in COVARIABLES_PS
    }).T
    return {"n_traites": int(t.sum()), "n_apparies": int(dans_support.sum()),
            "effets": pd.DataFrame(effets).T, "equilibre": equilibre,
            "score": pd.DataFrame({"logit_ps": lp, "traite": t})}


# --------------------------------------------------------------------------
# Inférence par permutation
# --------------------------------------------------------------------------
def permutation(d: pd.DataFrame, formule: str, variable: str, parmi: str, n_perm: int = 500,
                logistique: bool = False, seed: int = 0) -> dict:
    """p-valeur de permutation du coefficient de ``variable``.

    On redistribue au hasard ``variable`` entre les lignes où ``parmi`` vaut 1
    (par exemple : l'affaire révélée pendant le mandat entre toutes les
    personnalités recensées), en gardant le nombre de traités. Sous l'hypothèse
    d'absence d'effet, le coefficient observé est un tirage comme un autre :
    la p-valeur est la part des permutations qui donnent un coefficient au
    moins aussi grand en valeur absolue.

    Logit : un tirage dont l'estimation ne converge pas (quasi-séparation, par
    exemple une catégorie rare où toutes les listes gagnent) est écarté et compté.
    """
    def ajuster(x: pd.DataFrame) -> float:
        if not logistique:
            return smf.ols(formule, data=x).fit().params[variable]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # non-convergence : testée juste après
            r = smf.logit(formule, data=x).fit(disp=0, maxiter=300)
        return r.params[variable] if r.mle_retvals["converged"] else np.nan

    observe = ajuster(d)
    rng = np.random.default_rng(seed)
    lignes = np.flatnonzero(d[parmi].to_numpy() == 1)
    valeurs = d[variable].to_numpy()[lignes]
    tirages = np.empty(n_perm)
    x = d.copy()
    for i in range(n_perm):
        v = x[variable].to_numpy().copy()
        v[lignes] = rng.permutation(valeurs)
        x[variable] = v
        tirages[i] = ajuster(x)
    valides = tirages[~np.isnan(tirages)]
    return {"coef": observe, "p_permutation": float((np.abs(valides) >= abs(observe)).mean()),
            "tirages": valides, "non_convergents": int(np.isnan(tirages).sum())}

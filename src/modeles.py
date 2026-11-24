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
               "affaire_probite_connue", "condamne_avant",
               "affaire_niveau", "affaire_soupcon", "affaire_procedure", "affaire_condamnation",
               "affaire_blanchi"]

# Gradation de l'affaire au vote (src/affaires.py) : score ordinal 0-3 (effet
# « dose-réponse », un seul coefficient) ou une indicatrice par niveau. Les élus
# déjà blanchis forment un groupe à part, comparé aux sortants sans affaire.
# Soupçon et procédure sont regroupés (« avant jugement ») : chacun ne compte que
# quelques sortants, tous ou presque réélus, ce qui empêche le logit de converger.
GRADATION_ORDINALE = "affaire_niveau + affaire_blanchi + personnalite_affaires"
GRADATION_NIVEAUX = "affaire_avant_jugement + affaire_condamnation + affaire_blanchi + personnalite_affaires"

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
    d["affaire_avant_jugement"] = d["affaire_soupcon"] + d["affaire_procedure"]
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


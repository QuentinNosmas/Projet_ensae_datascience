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



# --------------------------------------------------------------------------
# Analyses de synthèse (partagées par main.py et les notebooks 00 et 04)
# --------------------------------------------------------------------------
APPROCHES = ["1. à taille égale", "2. régression, maires ordinaires",
             "3. appariement (score de propension)", "4. régression, à notoriété égale"]


def _ligne(effet: float, bas: float, haut: float, echelle: float = 1) -> dict:
    return {"effet": effet * echelle, "ic_bas": bas * echelle, "ic_haut": haut * echelle}


def _coef(res, variable: str = "affaire_mandat", echelle: float = 1) -> dict:
    c = coefficients(res).loc[variable]
    return _ligne(c.coef, c.ic_bas, c.ic_haut, echelle)


def effets_par_approche(d: pd.DataFrame) -> pd.DataFrame:
    """Effet d'une affaire révélée pendant le mandat, selon quatre approches.

    1. écart avec les sortants sans affaire des communes de même taille, à la même
       élection (stratification, IC bootstrap) ;
    2. régression avec contrôles, contre les maires ordinaires ;
    3. appariement sur le score de propension (5 plus proches voisins) ;
    4. régression avec contrôles et indicatrice de notoriété : comparaison avec les
       personnalités dont l'affaire n'éclate qu'après le vote.

    Renvoie une table (approche x {score, victoire} x {effet, ic_bas, ic_haut}),
    en points de % (la victoire est un effet sur la probabilité, x 100).
    """
    strat = d[(d["affaire_mandat"] == 1) | (d["personnalite_affaires"] == 0)]
    f_ord = "{y} ~ affaire_mandat + " + CONTROLES
    f_not = "{y} ~ affaire_mandat + affaire_avant_mandat + personnalite_affaires + " + CONTROLES
    app = appariement(strat.reset_index(drop=True), "affaire_mandat", ["delta_pct_voix_exp", "gagne"])

    score, victoire = {}, {}
    s = An.ecart_stratifie(strat, "delta_pct_voix_exp", "affaire_mandat", ["annee", "taille"])
    score[APPROCHES[0]] = _ligne(s.ecart, s.ic_bas, s.ic_haut)
    s = An.ecart_stratifie(strat, "gagne", "affaire_mandat", ["annee", "taille"])
    victoire[APPROCHES[0]] = _ligne(s.ecart, s.ic_bas, s.ic_haut, 100)
    score[APPROCHES[1]] = _coef(ols(f_ord.format(y="delta_pct_voix_exp"), strat))
    victoire[APPROCHES[1]] = _coef(logit(f_ord.format(y="gagne"), strat), echelle=100)
    e = app["effets"]
    score[APPROCHES[2]] = _ligne(*e.loc["delta_pct_voix_exp", ["effet", "ic_bas", "ic_haut"]])
    victoire[APPROCHES[2]] = _ligne(*e.loc["gagne", ["effet", "ic_bas", "ic_haut"]], 100)
    score[APPROCHES[3]] = _coef(ols(f_not.format(y="delta_pct_voix_exp"), d))
    victoire[APPROCHES[3]] = _coef(logit(f_not.format(y="gagne"), d), echelle=100)
    return pd.concat({"Évolution du score": pd.DataFrame(score).T,
                      "Probabilité de victoire": pd.DataFrame(victoire).T}, axis=1)


def camp_politique(d: pd.DataFrame) -> pd.Series:
    """Bord politique de la liste du sortant à l'élection précédente : gauche, droite, autre."""
    return d["bloc_prec"].map({"gauche": "gauche", "droite": "droite"}).fillna("autre")


def par_bord_politique(d: pd.DataFrame) -> dict:
    """Effet d'une affaire connue selon le bord du sortant, à notoriété égale.

    Renvoie les effectifs descriptifs, les coefficients (score et victoire) et le
    test de l'écart gauche - droite (en points pour le score, sur l'échelle logit
    pour la victoire).
    """
    camp = camp_politique(d)
    x = d.assign(**{f"affaire_{c}": d["affaire_connue"] * (camp == c) for c in ("gauche", "droite", "autre")})
    descriptif = (x[x["affaire_connue"] == 1].assign(camp=camp)
                  .groupby("camp").agg(sortants=("code_insee", "size"),
                                       evolution_score=("delta_pct_voix_exp", "mean"),
                                       victoire=("gagne", "mean")))
    formule = "~ affaire_gauche + affaire_droite + affaire_autre + personnalite_affaires + " + CONTROLES
    res = {"évolution du score (pts)": ols("delta_pct_voix_exp " + formule, x),
           "victoire (effet marginal)": logit("gagne " + formule, x)}
    tests = {}
    for nom, r in res.items():
        t = r.t_test("affaire_gauche - affaire_droite = 0")
        tests[nom.split(" (")[0]] = {"ecart_gauche_droite": float(t.effect[0]), "p": float(t.pvalue)}
    return {"descriptif": descriptif, "coefficients": pd.concat({k: coefficients(r) for k, r in res.items()}),
            "tests": pd.DataFrame(tests).T}


def stade_au_vote(d: pd.DataFrame) -> pd.Series:
    """Groupe de chaque sortant selon le stade de son affaire au 1er tour."""
    return pd.Series(np.select(
        [d["affaire_blanchi"] == 1, d["affaire_condamnation"] == 1, d["affaire_procedure"] == 1,
         d["affaire_soupcon"] == 1, d["personnalite_affaires"] == 1],
        ["blanchi", "3 condamnation", "2 procédure", "1 soupçon", "0 personnalité, affaire future"],
        "sortant ordinaire"), index=d.index)


def par_stade(d: pd.DataFrame) -> dict:
    """Effet de l'affaire selon son stade au vote (soupçon, procédure, condamnation, blanchi).

    Deux spécifications : score ordinal 0-3 (« dose-réponse ») et une indicatrice
    par niveau (soupçon et procédure regroupés). Test : la condamnation coûte-t-elle
    plus qu'une affaire pas encore jugée ?
    """
    descriptif = (d.assign(stade=stade_au_vote(d)).groupby("stade")
                  .agg(sortants=("code_insee", "size"), dont_2026=("election_2026", "sum"),
                       inscrits=("inscrits", "median"), score_precedent=("pct_voix_exp_prec", "mean"),
                       evolution_score=("delta_pct_voix_exp", "mean"), victoire=("gagne", "mean")))
    res = {}
    for spec, rhs in {"ordinal": GRADATION_ORDINALE, "par niveau": GRADATION_NIVEAUX}.items():
        res[("évolution du score (pts)", spec)] = ols(f"delta_pct_voix_exp ~ {rhs} + {CONTROLES}", d)
        res[("victoire (effet marginal)", spec)] = logit(f"gagne ~ {rhs} + {CONTROLES}", d)
    tests = {}
    for (resultat, spec), r in res.items():
        if spec == "par niveau":
            t = r.t_test("affaire_condamnation - affaire_avant_jugement = 0")
            tests[resultat.split(" (")[0]] = {"condamnation_moins_avant_jugement": float(t.effect[0]),
                                              "p": float(t.pvalue)}
    return {"descriptif": descriptif, "resultats": res,
            "coefficients": pd.concat({k: coefficients(r) for k, r in res.items()}),
            "tests": pd.DataFrame(tests).T}


def par_election(d: pd.DataFrame) -> pd.DataFrame:
    """Robustesse : effet d'une affaire révélée pendant le mandat sur le score, élection par
    élection, avec et sans l'indicatrice de notoriété."""
    lignes = {}
    controles = CONTROLES.replace(" + C(annee)", "")
    for annee in (2020, 2026):
        x = d[d["annee"] == annee]
        for libelle, notoriete in (("sans notoriété", ""), ("avec notoriété", " + personnalite_affaires")):
            f = "delta_pct_voix_exp ~ affaire_mandat + affaire_avant_mandat" + notoriete + " + " + controles
            lignes[(annee, libelle)] = coefficients(ols(f, x)).loc["affaire_mandat"]
    return pd.DataFrame(lignes).T

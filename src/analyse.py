"""Comparaisons « à taille de commune égale » entre sortants avec et sans affaire.

Les sortants visés par une affaire dirigent surtout de grandes villes : les
comparer à l'ensemble des sortants mélangerait l'effet de l'affaire et celui
de la taille. On compare donc à l'intérieur de strates (élection x tranche
d'inscrits), puis on moyenne les écarts avec le poids de chaque strate chez
les traités (estimateur de l'effet sur les traités par stratification exacte).

Avec quelques dizaines de traités, l'incertitude est grande : intervalle de
confiance à 95 % par bootstrap (tirage avec remise à l'intérieur de chaque
groupe et de chaque strate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TAILLES = [0, 1000, 3500, 10_000, 30_000, np.inf]
LIBELLES_TAILLE = ["< 1 000", "1 000-3 500", "3 500-10 000", "10 000-30 000", "30 000 +"]


def tranche_taille(inscrits: pd.Series) -> pd.Series:
    return pd.cut(inscrits, TAILLES, labels=LIBELLES_TAILLE, right=False)


def _ecart(y: np.ndarray, traite: np.ndarray, strate: np.ndarray) -> float:
    """Écart moyen traités - témoins, strate par strate, pondéré par les traités.
    Les strates sans témoin sont ignorées (leurs traités ne sont pas comparables)."""
    ecarts, poids = [], []
    for s in np.unique(strate[traite]):
        t = y[traite & (strate == s)]
        c = y[~traite & (strate == s)]
        if len(c):
            ecarts.append(t.mean() - c.mean())
            poids.append(len(t))
    return float(np.average(ecarts, weights=poids)) if poids else np.nan


def ecart_stratifie(df: pd.DataFrame, y: str, traite: str, strates: list[str],
                    n_boot: int = 2000, seed: int = 0) -> pd.Series:
    """Écart traités - témoins sur ``y`` à strates égales, avec IC 95 % bootstrap.

    ``df`` ne contient que les traités et leurs témoins ; lignes où ``y`` manque retirées.
    """
    d = df.dropna(subset=[y])
    yv = d[y].astype(float).to_numpy()
    tv = d[traite].astype(bool).to_numpy()
    sv = d[strates].astype(str).agg("|".join, axis=1).to_numpy()
    est = _ecart(yv, tv, sv)

    rng = np.random.default_rng(seed)
    groupes = [np.flatnonzero((sv == s) & (tv == t)) for s in np.unique(sv) for t in (True, False)]
    groupes = [g for g in groupes if len(g)]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g)) for g in groupes])
        boot[b] = _ecart(yv[idx], tv[idx], sv[idx])
    bas, haut = np.nanpercentile(boot, [2.5, 97.5])
    return pd.Series({"n_traites": int(tv.sum()), "n_temoins": int((~tv).sum()),
                      "ecart": est, "ic_bas": bas, "ic_haut": haut})

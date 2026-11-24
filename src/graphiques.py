"""Charte graphique et graphiques réutilisés par plusieurs notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402

COULEURS = {2020: "#2a78d6", 2026: "#eb6834"}
ENCRE, NEUTRE, FOND = "#52514e", "#8a8984", "#f1f0ec"
ISSUES = {"liste gagnante": "#2a78d6", "liste battue": "#eb6834", "ne se représente pas": NEUTRE}


def style() -> None:
    """Axes discrets, grille légère, deux couleurs de série au plus."""
    plt.rcParams.update({
        "figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": NEUTRE, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": ENCRE, "ytick.color": ENCRE,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def contours_metropole():
    """(départements, centroïdes des communes) en Lambert-93, France métropolitaine."""
    import geopandas as gpd

    chemin = P.PROCESSED_DIR / "contours_communes.parquet"
    if not chemin.exists():
        P.load.write_processed("contours_communes")
    contours = gpd.read_parquet(chemin)
    metro = contours[~contours["departement"].str.startswith(("97", "98"))].to_crs(2154)
    return metro.dissolve("departement"), metro.set_index("code_insee").geometry.centroid


def carte_affaires(panel: pd.DataFrame, titre: str = "Sortants dont une affaire était connue au moment du vote"):
    """Une carte par élection : sortants avec affaire connue, colorés selon l'issue."""
    import geopandas as gpd

    departements, centres = contours_metropole()
    issue = np.select(
        [~panel["sortant_candidat"].fillna(False), panel["liste_sortant_gagnante"].fillna(False)],
        ["ne se représente pas", "liste gagnante"], "liste battue")
    points = panel.assign(issue=issue)[panel["affaire_connue"]]
    points = gpd.GeoDataFrame(points, geometry=points["code_insee"].map(centres), crs=2154)
    outre_mer = points.geometry.isna()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    for ax, annee in zip(axes, (2020, 2026)):
        departements.plot(ax=ax, color=FOND, edgecolor="#d4d3cf", linewidth=0.4)
        g = points[(points["annee"] == annee) & ~outre_mer]
        for lib, couleur in ISSUES.items():
            sel = g[g["issue"] == lib]
            if len(sel):
                sel.plot(ax=ax, color=couleur, markersize=34, edgecolor="white", linewidth=1,
                         label=f"{lib} ({len(sel)})", zorder=3)
        n_om = int((outre_mer & (points["annee"] == annee)).sum())
        ax.set_title(f"{annee}" + (f"  (+ {n_om} en outre-mer)" if n_om else ""), fontsize=10)
        ax.set_axis_off()
        ax.legend(loc="lower left", fontsize=8, handletextpad=0.3)
    fig.suptitle(titre, fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def estimations(table: pd.DataFrame, titre: str):
    """Points et intervalles de confiance, une ligne par approche, un panneau par résultat.

    ``table`` : index = approche, colonnes MultiIndex (resultat, {effet, ic_bas, ic_haut}).
    """
    resultats = table.columns.get_level_values(0).unique()
    fig, axes = plt.subplots(1, len(resultats), figsize=(4.6 * len(resultats), 0.55 * len(table) + 1.4),
                             sharey=True)
    for ax, res in zip(np.atleast_1d(axes), resultats):
        t = table[res]
        y = np.arange(len(t))
        ax.hlines(y, t["ic_bas"], t["ic_haut"], color=COULEURS[2026], lw=2)
        ax.plot(t["effet"], y, "o", ms=7, color=COULEURS[2026], mec="white", mew=1.5)
        for yi, v in zip(y, t["effet"]):
            ax.annotate(f"{v:+.1f}", (v, yi), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=ENCRE)
        ax.axvline(0, color=NEUTRE, lw=1)
        ax.set_yticks(y, t.index)
        ax.set_ylim(len(t) - 0.5, -0.7)
        ax.grid(axis="y", visible=False)
        ax.set_title(res, fontsize=10)
        ax.set_xlabel("écart (points de %), IC 95 %")
    fig.suptitle(titre, fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def retour_moyenne(d: pd.DataFrame):
    """Évolution du score selon le score à l'élection précédente.

    Sortants ordinaires en fond (moyenne par tranche de 5 points), sortants dont
    une affaire était connue au vote en couleur. ``d`` : sortie de
    ``modeles.table_modele``.
    """
    ordinaires = d[d["personnalite_affaires"] == 0]
    traites = d[d["affaire_connue"] == 1]
    tranches = np.arange(15, 101, 5)
    moy = (ordinaires.groupby(pd.cut(ordinaires["pct_voix_exp_prec"], tranches), observed=True)
           ["delta_pct_voix_exp"].agg(["mean", "size"]))
    moy = moy[moy["size"] >= 30]
    milieux = [i.mid for i in moy.index]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(ordinaires["pct_voix_exp_prec"], ordinaires["delta_pct_voix_exp"], s=5, color=NEUTRE,
               alpha=0.18, linewidth=0, rasterized=True)
    ax.plot(milieux, moy["mean"], color=ENCRE, lw=2,
            label=f"sortants ordinaires ({len(ordinaires):,} ; moyenne par tranche)".replace(",", " ", 1))
    ax.scatter(traites["pct_voix_exp_prec"], traites["delta_pct_voix_exp"], s=42, color=COULEURS[2026],
               edgecolor="white", linewidth=1.2, zorder=3, label=f"affaire connue au vote ({len(traites)})")
    ax.axhline(0, color=NEUTRE, lw=1)
    for groupe, couleur in ((ordinaires, ENCRE), (traites, COULEURS[2026])):
        m = groupe["pct_voix_exp_prec"].mean()
        ax.axvline(m, color=couleur, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"{m:.0f} %", (m, 0), xycoords=("data", "axes fraction"), xytext=(3, 4),
                    textcoords="offset points", fontsize=8, color=ENCRE)
    ax.set_ylim(np.percentile(d["delta_pct_voix_exp"], [0.5, 99.5]) + np.array([-3, 3]))
    ax.set_xlim(10, 102)
    ax.set_xlabel("Score à l'élection précédente (% des exprimés, 1er tour) ; tirets : moyenne du groupe")
    ax.set_ylabel("Évolution du score (points)")
    ax.set_title("Mieux on était élu, plus on recule : le retour vers la moyenne")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    return fig


STADES_ACTUELS = {"en_cours": "procédure en cours", "condamnation": "condamnation",
                  "abandon": "relaxe, non-lieu, classement"}


def chronologie_affaires(aff: pd.DataFrame):
    """Date publique des affaires rattachées aux sortants, selon leur issue actuelle,
    face aux dates des scrutins municipaux. ``aff`` : sortie de ``affaires.build``."""
    a = aff.dropna(subset=["date_publique"]).drop_duplicates(["source", "personne_id", "intitule", "date_publique"])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for y, (issue, lib) in enumerate(STADES_ACTUELS.items()):
        g = a[a["issue"] == issue]
        couleur = NEUTRE if issue == "abandon" else COULEURS[2026]
        ax.scatter(g["date_publique"], y + rng.uniform(-0.22, 0.22, len(g)), s=20, color=couleur,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"n = {len(g)}", (1.01, y), xycoords=("axes fraction", "data"), va="center",
                    fontsize=8, color=ENCRE)
    for annee, date in P.DATES_PREMIER_TOUR.items():
        ax.axvline(pd.Timestamp(date), color=ENCRE, lw=1, ls=(0, (4, 3)))
        ax.annotate(str(annee), (pd.Timestamp(date), 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                    color=ENCRE)
    ax.set_yticks(range(len(STADES_ACTUELS)), list(STADES_ACTUELS.values()))
    ax.set_ylim(len(STADES_ACTUELS) - 0.5, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Date publique de l'affaire (début de procédure, à défaut verdict)")
    ax.set_title("Affaires rattachées aux maires sortants, selon leur issue actuelle" + "\n"
                 "tirets : 1er tour des municipales", pad=18)
    plt.tight_layout()
    return fig


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion k/n."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - demi, centre + demi


def victoire_par_groupe(d: pd.DataFrame):
    """Taux de victoire (IC 95 % de Wilson) : sortants ordinaires, personnalités sans
    affaire connue, et sortants visés selon le stade de l'affaire au vote."""
    groupes = {
        "sortants ordinaires": (d["personnalite_affaires"] == 0, NEUTRE),
        "personnalités, affaire après le vote": ((d["personnalite_affaires"] == 1) & (d["affaire_connue"] == 0), ENCRE),
        "affaire pas encore jugée": (d["affaire_avant_jugement"] == 1, COULEURS[2026]),
        "condamnation avant le vote": (d["affaire_condamnation"] == 1, COULEURS[2026]),
        "blanchi avant le vote": (d["affaire_blanchi"] == 1, COULEURS[2026]),
    }
    fig, ax = plt.subplots(figsize=(8, 3.4))
    labels = []
    for y, (lib, (masque, couleur)) in enumerate(groupes.items()):
        g = d.loc[masque, "gagne"]
        bas, haut = _wilson(g.sum(), len(g))
        ax.plot([100 * bas, 100 * haut], [y, y], color=couleur, lw=2, solid_capstyle="round")
        ax.plot(100 * g.mean(), y, "o", ms=7, color=couleur, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{100 * g.mean():.0f} %", (100 * g.mean(), y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=ENCRE)
        labels.append(f"{lib} (n = {len(g):,})".replace(f"{len(g):,}", f"{len(g):,}".replace(",", " ")))
    ax.axvline(100 * d.loc[d["personnalite_affaires"] == 0, "gagne"].mean(), color=NEUTRE, lw=1, ls=(0, (4, 3)))
    ax.axhline(1.5, color="#e4e3df", lw=0.8)
    ax.set_yticks(range(len(groupes)), labels)
    ax.set_ylim(len(groupes) - 0.5, -0.7)
    ax.set_xlim(40, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Liste du sortant gagnante (%), IC 95 %")
    ax.set_title("La notoriété pèse plus que l'affaire sur la victoire")
    plt.tight_layout()
    return fig

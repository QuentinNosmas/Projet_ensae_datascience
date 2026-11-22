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

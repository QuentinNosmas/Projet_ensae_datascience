"""Les électeurs sanctionnent-ils les maires visés par une affaire judiciaire ?

Point d'entrée du projet : enchaîne toute la chaîne de traitement, des données
brutes aux résultats, et imprime un résumé détaillé de l'étude. Les chiffres
sont calculés à chaque exécution, aucun n'est écrit en dur.

Étapes :
  1. données     sources publiques téléchargées si absentes (data/raw/),
                 nettoyées (data/processed/) ; affaires : instantané daté
  2. panel       maire sortant x élection (municipales 2020 et 2026)
  3. affaires    rattachement des affaires aux sortants, date et stade au vote
  4. contexte    recensement, revenus, présidentielles de chaque commune
  5. résultats   référence, profil, quatre approches, bord politique, stade de
                 l'affaire, robustesse, conclusion

Usage :
  python main.py                      # tout reconstruire et imprimer le résumé
  python main.py --cache              # réutiliser les tables déjà construites
  python main.py --figures            # enregistrer aussi les figures (sorties/figures/)
  python main.py --figures mon_dossier
  python main.py --markdown           # écrire le résumé dans RESULTATS.md (lisible sur
                                      # GitHub), figures dans docs/resultats/
  python main.py > resume.txt         # garder le résumé du terminal dans un fichier

Premier lancement : environ 550 Mo à télécharger (de 30 minutes à 1 heure).
Ensuite : 2 à 3 minutes.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import textwrap
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # figures écrites dans des fichiers, jamais affichées

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import affaires as A  # noqa: E402
import covariables as Cov  # noqa: E402
import graphiques as G  # noqa: E402
import modeles as M  # noqa: E402
import panel as P  # noqa: E402

LARGEUR = 88
pd.set_option("display.width", LARGEUR + 40)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 40)


# --------------------------------------------------------------------------
# Mise en forme du résumé
# --------------------------------------------------------------------------
class Rendu:
    """Destination du résumé : le terminal (texte justifié) ou un fichier Markdown,
    que GitHub affiche avec titres, tableaux et figures."""

    def __init__(self, markdown: Path | None = None, figures: Path | None = None):
        self.markdown = markdown
        self.figures = figures
        self.lignes: list[str] = []

    def ecrire(self, ligne: str = "") -> None:
        if self.markdown is None:
            print(ligne)
        else:
            self.lignes.append(ligne)

    def enregistrer(self) -> None:
        if self.markdown is not None:
            self.markdown.write_text("\n".join(self.lignes).strip() + "\n", encoding="utf-8")


R = Rendu()  # remplacé dans main() selon les options


def titre(texte: str) -> None:
    if R.markdown is None:
        R.ecrire("\n" + "=" * LARGEUR + f"\n{texte.upper()}\n" + "=" * LARGEUR)
    else:
        R.ecrire(f"\n## {texte}")


def sous_titre(texte: str) -> None:
    if R.markdown is None:
        R.ecrire(f"\n--- {texte} " + "-" * max(0, LARGEUR - len(texte) - 5))
    else:
        R.ecrire(f"\n### {texte[0].upper()}{texte[1:]}")


INSECABLE = {" :": "\u00a0:", " ;": "\u00a0;", " %": "\u00a0%", "« ": "«\u00a0", " »": "\u00a0»"}


def texte(*paragraphes: str) -> None:
    """Paragraphes : justifiés à LARGEUR dans le terminal (sans retour à la ligne avant
    « : », « ; », « % »), d'un seul tenant en Markdown."""
    for p in paragraphes:
        p = " ".join(p.split())
        if R.markdown is not None:
            R.ecrire("\n" + p)
            continue
        for espace, insecable in INSECABLE.items():
            p = p.replace(espace, insecable)
        R.ecrire("\n" + textwrap.fill(p, LARGEUR).replace("\u00a0", " "))


# Noms lisibles des variables, dans les en-têtes et l'index des tableaux.
LIBELLES = {
    "effectif": "effectif", "sortants": "sortants", "dont_2026": "dont 2026",
    "inscrits": "inscrits (médiane)", "inscrits_median": "inscrits (médiane)", "listes": "listes",
    "score_precedent": "score précédent (%)", "evolution_score": "évolution du score (pts)",
    "victoire": "victoire (%)", "victoire_pct": "victoire (%)",
    "effet": "effet", "ic_bas": "IC bas", "ic_haut": "IC haut", "coef": "coef.", "p": "p",
    "affaire_gauche": "affaire, sortant de gauche", "affaire_droite": "affaire, sortant de droite",
    "affaire_autre": "affaire, autre bord", "personnalite_affaires": "notoriété (personnalité recensée)",
    "affaire_niveau": "stade de l'affaire (0 à 3)", "affaire_blanchi": "élu déjà blanchi",
    "affaire_avant_jugement": "affaire pas encore jugée", "affaire_condamnation": "condamnation avant le vote",
    "ecart_gauche_droite": "écart gauche - droite", "condamnation_moins_avant_jugement":
        "condamnation - pas encore jugée",
    "camp": "bord", "stade": "stade de l'affaire au vote", "groupe": "groupe", "annee": "élection",
}


def _libelle(v: object) -> str:
    if isinstance(v, tuple):
        return " / ".join(_libelle(x) for x in v)
    return LIBELLES.get(v, str(v)) if isinstance(v, str) else str(v)


def _lisible(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les noms de variables par des libellés (colonnes, index et leurs noms)."""
    df = df.copy()
    df.columns = df.columns.map(lambda c: tuple(_libelle(x) for x in c) if isinstance(c, tuple) else _libelle(c))
    df.columns.names = [_libelle(n) if n else n for n in df.columns.names]
    df.index = df.index.map(lambda i: tuple(_libelle(x) for x in i) if isinstance(i, tuple) else _libelle(i))
    df.index.names = [_libelle(n) if n else n for n in df.index.names]
    return df


def _cellule(v: object, decimales: int) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "oui" if v else "non"
    if isinstance(v, (int, np.integer)):
        return nombre(v, 0)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return nombre(v, 0 if abs(v) >= 1000 else decimales)
    return str(v)


def _markdown(df: pd.DataFrame, decimales: int) -> str:
    """Table Markdown : index et colonnes à plusieurs niveaux aplatis, nombres à la française."""
    joindre = lambda v: " / ".join(map(str, v)) if isinstance(v, tuple) else str(v)
    entete = [" / ".join(n for n in df.index.names if n)]
    entete += [joindre(c) for c in df.columns]
    lignes = ["| " + " | ".join(entete) + " |", "|" + "---|" * len(entete)]
    for idx, ligne in df.iterrows():
        etiquette = joindre(idx)
        # Lignes « dont » : retrait conservé (espaces insécables, que GitHub affiche).
        retrait = len(etiquette) - len(etiquette.lstrip())
        cellules = ["&nbsp;" * 2 * retrait + etiquette.strip()] + [_cellule(v, decimales) for v in ligne]
        lignes.append("| " + " | ".join(c.replace("|", "/") for c in cellules) + " |")
    return "\n".join(lignes)


def tableau(df: pd.DataFrame | pd.Series, decimales: int = 1) -> None:
    df = _lisible(df.to_frame() if isinstance(df, pd.Series) else df)
    if R.markdown is None:
        R.ecrire("\n" + textwrap.indent(df.round(decimales).to_string(), "  "))
    else:
        R.ecrire("\n" + _markdown(df, decimales))


def figure(nom: str, tracer, legende: str) -> None:
    """Trace une figure si un dossier de figures est demandé ; en Markdown, l'insère."""
    if R.figures is None:
        return
    import matplotlib.pyplot as plt  # noqa: PLC0415

    chemin = R.figures / nom
    fig = tracer()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if R.markdown is None:
        R.ecrire(f"\n  [figure] {chemin.relative_to(ROOT).as_posix() if chemin.is_relative_to(ROOT) else chemin}")
    else:
        lien = Path(os.path.relpath(chemin, R.markdown.parent)).as_posix()
        R.ecrire(f"\n![{legende}]({lien})\n\n*{legende}*")


def nombre(x: float, decimales: int = 1, signe: bool = False) -> str:
    """Nombre à la française : 1 234,5 ; signe explicite si demandé."""
    s = f"{x:+,.{decimales}f}" if signe else f"{x:,.{decimales}f}"
    return s.replace(",", " ").replace(".", ",")


def significativite(p: float) -> str:
    if p < 0.01:
        return "significatif au seuil de 1 %"
    if p < 0.05:
        return "significatif au seuil de 5 %"
    if p < 0.1:
        return "significatif au seuil de 10 % seulement"
    return "non significatif"


def effet(ligne: pd.Series, unite: str = "points") -> str:
    """« -29,3 points (IC 95 % : -57,5 à -1,2) »."""
    return (f"{nombre(ligne['effet'], signe=True)} {unite} "
            f"(IC 95 % : {nombre(ligne['ic_bas'])} à {nombre(ligne['ic_haut'])})")


class Chrono:
    """Affiche la durée de chaque étape de construction."""

    def __init__(self, nom: str):
        self.nom = nom

    def __enter__(self):
        self.debut = time.perf_counter()
        print(f"  {self.nom}...", end="", flush=True)
        return self

    def __exit__(self, *exc):
        print(f" {time.perf_counter() - self.debut:.0f} s")


# --------------------------------------------------------------------------
# Étapes 1 à 4 : construction des données
# --------------------------------------------------------------------------
def construire(cache: bool) -> dict:
    """Construit (ou relit) toutes les tables de l'étude.

    Les sources brutes manquantes sont téléchargées à la volée par src/load.py ;
    les tables dérivées (panel, affaires, covariables) sont reconstruites, sauf
    avec ``cache`` si elles existent déjà.
    """
    if R.markdown is None:
        titre("Construction des données")
    else:
        print("Construction des données :")
    sortie = {}
    with Chrono("panel maire sortant x élection (municipales 2014, 2020, 2026)"):
        sortie["panel_sortants"] = P.load_panel() if cache else P.write_panel()
    with Chrono("affaires judiciaires : harmonisation et rattachement aux sortants"):
        personnes, toutes = A.toutes_affaires()
        couples = A.apparier_sortants(sortie["panel_sortants"], personnes)
        panel, aff = A.build()
        panel.to_parquet(P.PROCESSED_DIR / "panel_affaires.parquet", index=False)
        aff.to_parquet(P.PROCESSED_DIR / "affaires_sortants.parquet", index=False)
    with Chrono("contexte des communes (recensement, revenus, présidentielles)"):
        if not cache or not Cov.COV_PATH.exists():
            Cov.build().to_parquet(Cov.COV_PATH, index=False)
    with Chrono("échantillon des modèles"):
        d = M.table_modele()
    sortie.update(personnes=personnes, toutes=toutes, couples=couples, aff=aff, d=d,
                  panel=panel[~panel["perimetre_modifie"]].copy())
    return sortie


# --------------------------------------------------------------------------
# Étape 5 : résultats
# --------------------------------------------------------------------------
def section_question() -> None:
    titre("Question et stratégie")
    texte(
        """La science politique décrit un avantage du sortant : le maire qui se représente
        gagne le plus souvent. Une affaire judiciaire (enquête, mise en examen,
        condamnation) devrait l'entamer si les électeurs sanctionnent les manquements à la
        probité. On compare donc les maires sortants visés par une affaire connue au moment
        du vote aux autres sortants, aux municipales de 2020 et de 2026.""",
        """Trois résultats sont mesurés : le sortant se représente-t-il ? Comment évolue son
        score au 1er tour par rapport à l'élection précédente ? Sa liste gagne-t-elle ?""",
        """Deux difficultés guident la démarche. Les sortants visés par une affaire ne sont
        pas des maires comme les autres (grandes villes, figures nationales) : il faut
        comparer à caractéristiques égales. Et les sources d'affaires ne recensent que des
        élus connus : il faut séparer l'effet de l'affaire de celui de la notoriété.""",
    )


def section_donnees(b: dict) -> None:
    titre("1. Données")
    texte(
        """Sources : résultats et candidatures des municipales 2014, 2020 et 2026, maires
        sortants 2026 et Répertoire national des élus (ministère de l'Intérieur), Code
        officiel géographique et contours des communes (INSEE, Etalab), recensement 2017 et
        2023 et revenus Filosofi (INSEE), présidentielles 2017 et 2022. Il n'existe pas de
        base officielle nominative des procédures visant des élus : les affaires viennent
        de deux bases citoyennes, Casier Politique (condamnations extraites de Wikipédia)
        et Poligraph (affaires à tous les stades), dont on utilise un instantané daté
        (data/snapshots/) pour que les résultats ne changent pas avec leurs mises à jour.""",
    )
    couples, panel = b["couples"], b["panel"]
    effectifs = pd.Series({
        "communes x élection dans le panel": len(panel),
        "affaires recensées (deux sources)": len(b["toutes"]),
        "personnalités recensées": b["personnes"]["personne_id"].nunique(),
        "rapprochements personnalité-sortant confirmés": int(couples["confirme"].sum()),
        "  rejetés : homonymes probables": int((couples["motif_rejet"] == "homonyme probable").sum()),
        "  rejetés : plus maire au scrutin": int((couples["motif_rejet"] == "plus maire au scrutin").sum()),
        "sortants avec une affaire connue au vote": int(panel["affaire_connue"].sum()),
        "  dont révélée pendant le mandat": int(panel["affaire_mandat"].sum()),
        "  dont atteinte à la probité": int(panel["affaire_probite_connue"].sum()),
        "sortants dont l'affaire n'éclate qu'après le vote": int(panel["affaire_apres"].sum()),
        "échantillon des modèles (têtes de liste, communes disputées)": len(b["d"]),
    }, name="effectif")
    tableau(effectifs, 0)
    texte(
        """Le rapprochement est conservateur : une personnalité n'est rattachée à un sortant
        que si sa date de naissance ou le nom de la commune cité dans sa fiche confirme
        l'identité, et seulement s'il était encore maire au scrutin (beaucoup d'élus
        nationaux ont quitté leur mairie en 2017, avec la fin du cumul des mandats).""",
        """Une affaire est datée par le début de la procédure, à défaut par le verdict :
        c'est une borne tardive, car la presse la révèle souvent plus tôt. Elle est
        « connue au vote » si cette date précède le 1er tour, « révélée pendant le mandat »
        si elle tombe entre les deux élections.""",
    )
    figure("chronologie_affaires.png", lambda: G.chronologie_affaires(b["aff"]),
           "Affaires rattachées aux maires sortants, selon leur date et leur issue actuelle")


def section_reference(b: dict) -> None:
    titre("2. La référence : le maire sortant ordinaire")
    panel, d = b["panel"], b["d"]
    disputees = panel[panel["n_listes"] > 1]
    ref = pd.DataFrame({
        "se représente (%)": 100 * panel.groupby("annee")["sortant_candidat"].mean(),
        "conduit sa liste, s'il se représente (%)":
            100 * panel[panel["sortant_candidat"].fillna(False)].groupby("annee")["sortant_tete_liste"].mean(),
        "sa liste gagne, communes disputées (%)":
            100 * disputees.groupby("annee")["liste_sortant_gagnante"].mean(),
        "évolution médiane du score (points)": d.groupby("annee")["delta_pct_voix_exp"].median(),
    }).T.astype(float)
    tableau(ref)
    texte(
        f"""Un maire sortant ordinaire se représente {nombre(ref.loc['se représente (%)', 2020], 0)} %
        du temps en 2020 et {nombre(ref.loc['se représente (%)', 2026], 0)} % en 2026. Quand il y a
        compétition, sa liste gagne environ 7 fois sur 10 : c'est l'avantage du sortant.
        Son score évolue beaucoup d'une élection à l'autre : c'est le bruit de fond dans
        lequel il faut détecter une éventuelle sanction.""",
    )


def section_profil(b: dict) -> None:
    titre("3. Les sortants visés par une affaire")
    d = b["d"]
    groupe = np.select([d["affaire_connue"] == 1, d["personnalite_affaires"] == 1],
                       ["affaire connue au vote", "personnalité, affaire future"], "autres sortants")
    profil = (d.assign(groupe=groupe).groupby("groupe")
              .agg(effectif=("code_insee", "size"), inscrits_median=("inscrits", "median"),
                   listes=("n_listes", "mean"), score_precedent=("pct_voix_exp_prec", "mean"),
                   evolution_score=("delta_pct_voix_exp", "mean"), victoire_pct=("gagne", "mean")))
    profil["victoire_pct"] *= 100
    figure("carte.png", lambda: G.carte_affaires(b["panel"]),
           "Sortants dont une affaire était connue au moment du vote, selon l'issue du scrutin")
    tableau(profil)
    aff, autres, futur = (profil.loc[g] for g in
                          ("affaire connue au vote", "autres sortants", "personnalité, affaire future"))
    texte(
        f"""Les sortants visés par une affaire dirigent des villes bien plus grandes
        ({nombre(aff.inscrits_median, 0)} inscrits en médiane, contre
        {nombre(autres.inscrits_median, 0)}), avec plus de listes en compétition, et avaient
        été moins bien élus la fois précédente ({nombre(aff.score_precedent, 0)} % contre
        {nombre(autres.score_precedent, 0)} %).""",
    )
    texte(
        f"""Premier piège, le retour vers la moyenne : un sortant très bien élu a plus à
        perdre. Les sortants avec affaire partaient de plus bas ; ils reculent donc moins (évolution
        moyenne du score de {nombre(aff.evolution_score, signe=True)} contre
        {nombre(autres.evolution_score, signe=True)} points) sans que l'affaire y soit pour rien. Il
        faut contrôler le score précédent.""",
    )
    figure("retour_moyenne.png", lambda: G.retour_moyenne(d),
           "Retour vers la moyenne : évolution du score selon le score précédent")
    texte(
        f"""Second piège, la notoriété : les personnalités dont l'affaire n'éclatera qu'après
        le vote leur ressemblent beaucoup, mais gagnent {nombre(futur.victoire_pct, 0)} % du
        temps, contre {nombre(autres.victoire_pct, 0)} % pour les sortants ordinaires. Les élus
        assez connus pour figurer dans nos sources bénéficient d'une prime électorale.""",
    )
    figure("victoire_par_groupe.png", lambda: G.victoire_par_groupe(d),
           "Taux de victoire selon la notoriété et le stade de l'affaire")


def section_approches(b: dict) -> pd.DataFrame:
    titre("4. Une affaire révélée pendant le mandat coûte-t-elle des voix ?")
    texte(
        """Quatre approches, sur le même échantillon (sortants têtes de liste aux deux
        élections, communes disputées les deux fois) : (1) écart avec les sortants sans
        affaire des communes de même taille ; (2) régression contre les maires ordinaires,
        avec contrôle du score précédent, de la taille, du nombre de listes, du bord
        politique, du contexte socio-économique et présidentiel ; (3) appariement sur le
        score de propension (5 « jumeaux » par sortant visé) ; (4) même régression, avec
        une indicatrice de notoriété : la comparaison se fait alors avec des personnalités
        dont l'affaire n'a éclaté qu'après le vote. Effets en points de pourcentage, avec
        intervalle de confiance à 95 %.""",
    )
    res = M.effets_par_approche(b["d"])
    tableau(res)
    figure("estimations.png",
           lambda: G.estimations(res, "Effet d'une affaire révélée pendant le mandat, selon l'approche"),
           "Effet d'une affaire révélée pendant le mandat, selon l'approche (IC 95 %)")
    score, victoire = res["Évolution du score"], res["Probabilité de victoire"]
    contient_zero = ((score["ic_bas"] <= 0) & (score["ic_haut"] >= 0)).all()
    notoriete = victoire.loc[M.APPROCHES[3]]
    texte(
        f"""Score : les estimations vont de {nombre(score['effet'].min(), signe=True)} à
        {nombre(score['effet'].max(), signe=True)} points selon l'approche, et
        {'tous les intervalles contiennent zéro' if contient_zero
         else 'au moins un intervalle exclut zéro'}. La perte observée à taille égale
        ({nombre(score.loc[M.APPROCHES[0], 'effet'], signe=True)}) disparaît une fois le score
        précédent contrôlé ({nombre(score.loc[M.APPROCHES[1], 'effet'], signe=True)}).""",
        f"""Victoire : face à des maires ordinaires, l'affaire ne change pas significativement
        la probabilité de gagner (approches 1 à 3). À notoriété égale (approche 4), l'effet est
        de {effet(notoriete)} : c'est le seul signal de sanction de l'étude, fragile
        (une trentaine de cas, surtout en 2020).""",
    )
    return res


def section_bord(b: dict) -> None:
    titre("5. Selon le bord politique")
    bord = M.par_bord_politique(b["d"])
    sous_titre("sortants avec affaire connue, selon le bord de leur liste")
    descriptif = bord["descriptif"].assign(victoire=lambda t: 100 * t["victoire"])
    tableau(descriptif)
    sous_titre("effets à notoriété égale (score en points, victoire en effet marginal)")
    tableau(bord["coefficients"], 3)
    t = bord["tests"]
    texte(
        f"""Les estimations vont dans le sens d'une sanction plus forte à gauche, mais l'écart
        gauche - droite est {significativite(t.loc['évolution du score', 'p'])} sur le score
        (p = {nombre(t.loc['évolution du score', 'p'], 2)}) et
        {significativite(t.loc['victoire', 'p'])} sur la victoire
        (p = {nombre(t.loc['victoire', 'p'], 2)}).""",
    )


def section_stade(b: dict) -> None:
    titre("6. Selon le stade de l'affaire au moment du vote")
    texte(
        """Les sources ne donnent que le statut actuel de l'affaire ; on reconstruit son
        stade au 1er tour avec la date du verdict : soupçon (pas de juge saisi), procédure
        judiciaire (mise en examen, instruction, procès en cours), condamnation déjà
        prononcée, ou élu déjà blanchi (relaxe, non-lieu, classement avant le vote).""",
    )
    stade = M.par_stade(b["d"])
    descriptif = stade["descriptif"].assign(victoire=lambda t: 100 * t["victoire"])
    tableau(descriptif)
    sous_titre("effets à notoriété égale (score en points, victoire en effet marginal)")
    tableau(stade["coefficients"], 3)
    c = stade["coefficients"].sort_index().loc[("victoire (effet marginal)", "par niveau")]
    t = stade["tests"]
    texte(
        f"""Une condamnation prononcée avant le vote réduirait la probabilité de victoire de
        {nombre(-100 * c.loc['affaire_condamnation', 'coef'], 0)} points
        (p = {nombre(c.loc['affaire_condamnation', 'p'], 2)},
        {significativite(c.loc['affaire_condamnation', 'p'])}), sans écart significatif avec
        une affaire pas encore jugée (p = {nombre(t.loc['victoire', 'p'], 2)}). Les élus déjà
        blanchis ne sont pas sanctionnés de façon détectable
        (p = {nombre(c.loc['affaire_blanchi', 'p'], 2)}).""",
    )


def section_robustesse(b: dict) -> None:
    titre("7. Robustesse : élection par élection")
    texte(
        """Effet d'une affaire révélée pendant le mandat sur le score, estimé séparément pour
        2020 (sortant approché par la tête de la liste gagnante de 2014) et 2026 (liste
        officielle des sortants), avec et sans l'indicatrice de notoriété.""",
    )
    rob = M.par_election(b["d"])
    tableau(rob, 2)
    d = b["d"]
    futurs_2026 = int(((d["annee"] == 2026) & (d["personnalite_affaires"] == 1) & (d["affaire_connue"] == 0)).sum())
    texte(
        f"""Contre des maires ordinaires, aucune élection ne montre de perte de voix. À
        notoriété égale, l'effet n'est négatif qu'en 2020 ; en 2026, la comparaison est
        presque impossible : il ne reste que {futurs_2026} personnalités sans affaire connue,
        les affaires futures n'ayant pas encore eu le temps d'éclater.""",
    )


def section_conclusion(res: pd.DataFrame) -> None:
    titre("Conclusion")
    notoriete = res["Probabilité de victoire"].loc[M.APPROCHES[3]]
    texte(
        """Les électeurs sanctionnent-ils vraiment ? Avec les données disponibles, pas de
        manière détectable. Un maire visé par une affaire judiciaire connue au moment du
        vote ne perd ni plus de voix ni plus souvent sa mairie qu'un maire comparable.""",
        f"""Le seul indice de sanction apparaît quand on le compare à des élus aussi connus
        que lui : {effet(notoriete)} sur la probabilité de victoire. La sanction existerait
        alors, mais serait masquée par l'avantage électoral de la notoriété.""",
        """Ni le bord politique ni le stade de l'affaire (soupçon, procédure, condamnation)
        ne changent significativement ce constat.""",
    )
    sous_titre("limites")
    texte(
        """1. Couverture : seuls les élus ayant une page Wikipédia ou une fiche Poligraph sont
        recensés ; les résultats valent pour les villes moyennes et grandes, pas pour les
        petites communes.""",
        """2. Datation : la date de procédure ou de verdict est postérieure à la révélation
        médiatique ; certaines affaires classées « après le vote » étaient peut-être déjà
        connues.""",
        """3. Sortants 2020 : faute de liste officielle, le sortant est approché par la tête de
        la liste gagnante de 2014 ; les départs en cours de mandat ne sont corrigés que pour
        les personnalités recensées.""",
        """4. Puissance : avec quelques dizaines de cas, seul un effet important serait
        détectable ; l'absence de preuve n'est pas la preuve de l'absence.""",
        """5. Causalité : même à caractéristiques et notoriété égales, les affaires ne sont pas
        attribuées au hasard (un maire sous pression peut renoncer à se représenter).""",
    )


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", action="store_true",
                   help="réutiliser le panel et les covariables déjà construits (plus rapide)")
    p.add_argument("--figures", nargs="?", const="sorties/figures", metavar="DOSSIER",
                   help="enregistrer aussi les figures (par défaut : sorties/figures)")
    p.add_argument("--markdown", nargs="?", const="RESULTATS.md", metavar="FICHIER",
                   help="écrire le résumé en Markdown (par défaut : RESULTATS.md, figures dans "
                        "docs/resultats/) au lieu de l'afficher")
    args = p.parse_args(argv)
    global R
    markdown = (ROOT / args.markdown).resolve() if args.markdown else None
    dossier = args.figures or ("docs/resultats" if markdown else None)
    R = Rendu(markdown, (ROOT / dossier).resolve() if dossier else None)
    if R.figures is not None:
        R.figures.mkdir(parents=True, exist_ok=True)
        G.style()
    # Le détail technique (téléchargements, nettoyage) reste dans les journaux, pas dans le résumé.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s : %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning)

    debut = time.perf_counter()
    print("LES ÉLECTEURS SANCTIONNENT-ILS LES MAIRES VISÉS PAR UNE AFFAIRE JUDICIAIRE ?")
    print("Municipales 2020 et 2026 - résumé de l'étude")
    b = construire(args.cache)
    if markdown is not None:
        R.ecrire("# Les électeurs sanctionnent-ils les maires visés par une affaire judiciaire ?")
        R.ecrire(f"\n*Municipales 2020 et 2026 : résumé de l'étude. Fichier généré par "
                 f"`python main.py --markdown` le {datetime.date.today():%d/%m/%Y}, à partir des "
                 f"données et du code du dépôt ; ne pas le modifier à la main. Détail des "
                 f"analyses : [notebooks/](notebooks/).*")
    section_question()
    section_donnees(b)
    section_reference(b)
    section_profil(b)
    res = section_approches(b)
    section_bord(b)
    section_stade(b)
    section_robustesse(b)
    section_conclusion(res)
    R.enregistrer()
    if markdown is not None:
        print(f"Résumé écrit dans {markdown.relative_to(ROOT).as_posix()}, "
              f"figures dans {R.figures.relative_to(ROOT).as_posix()}/")
    print(f"\nTerminé en {time.perf_counter() - debut:.0f} s. Détail : notebooks/00_synthese.ipynb à 04.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

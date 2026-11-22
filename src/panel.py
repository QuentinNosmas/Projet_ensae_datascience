"""Panel « maire sortant x élection municipale » (2020 et 2026).

Rapproche, pour chaque commune, le maire sortant de la liste qu'il conduit
(ou non) à l'élection suivante, et de celle qu'il conduisait à l'élection
précédente. C'est la table sur laquelle viendront se greffer les procédures
judiciaires : la clé d'un élu est (code_insee, nom_norm, prenom_norm).

Sources (data/processed, produites par src/load.py) :
- maire sortant 2026 : liste officielle des maires en fonction au 27/02/2026 ;
- maire sortant 2020 : pas de liste officielle publiée. Approximation : tête
  de la liste arrivée en tête (le plus de sièges) en 2014. Disponible
  uniquement pour les communes au scrutin de liste en 2014 (≥ 1000 hab.), et
  fausse si le maire a changé en cours de mandat ;
- résultats par liste 2014, 2020, 2026 ; têtes de liste 2026 tirées du
  fichier des candidatures (le fichier de résultats du 1er tour 2026 ne donne
  pas le nom de la tête de liste) ;
- RNE (maires en fonction en août 2026) : réélection effective en 2026 ;
- COG : communes dont le périmètre a changé (fusions...) entre deux scrutins.

Champ : communes au scrutin de liste, hors Paris / Lyon / Marseille et hors
sections électorales de Polynésie (résultats par secteur, non comparables).

Usage :
    python src/panel.py      # écrit data/processed/panel_sortants.parquet
"""

from __future__ import annotations

import logging
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

log = logging.getLogger("panel")

DATES_PREMIER_TOUR = {2014: "2014-03-23", 2020: "2020-03-15", 2026: "2026-03-15"}
PLM = {"75056", "13055", "69123"}  # Paris, Marseille, Lyon : scrutin par secteur

# Codes COG des événements qui changent le périmètre d'une commune :
# 21 rétablissement, 31 fusion simple, 32 commune nouvelle, 33 fusion-association,
# 34 fusion-association devenant fusion simple.
MOD_PERIMETRE = {"21", "31", "32", "33", "34"}

SEUIL_SIMILARITE = 0.85  # appariement approché des noms (difflib), voir apparier()

# Nuances de liste (sans le « L » initial) -> bloc politique, pour comparer
# 2014, 2020 et 2026 malgré les changements de grille du ministère.
# NC (non classé) : nuance non attribuée dans les communes de 1000 à 3500 hab.
BLOCS = {
    "EXG": "extreme_gauche",
    "FG": "gauche", "PG": "gauche", "COM": "gauche", "FI": "gauche", "SOC": "gauche",
    "UG": "gauche", "DVG": "gauche", "RDG": "gauche", "VEC": "gauche", "ECO": "gauche",
    "MDM": "centre", "UDI": "centre", "UC": "centre", "REM": "centre", "REN": "centre",
    "ENS": "centre", "HOR": "centre", "DVC": "centre",
    "UMP": "droite", "LR": "droite", "UD": "droite", "DVD": "droite", "DLF": "droite",
    # UDR (Ciotti) et UXD : alliés du RN, classés avec l'extrême droite.
    "FN": "extreme_droite", "RN": "extreme_droite", "EXD": "extreme_droite",
    "REC": "extreme_droite", "UXD": "extreme_droite", "UDR": "extreme_droite",
    "DIV": "divers", "REG": "divers", "GJ": "divers", "DSV": "divers",
}


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def read_processed(name: str) -> pd.DataFrame:
    """Lit data/processed/<name>.parquet, en le régénérant depuis data/raw s'il manque."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        load.write_processed(name)
    return pd.read_parquet(path)


def normalize_name(s: pd.Series) -> pd.Series:
    """« Jean-Pierre », « JEAN PIERRE », « Jéan  pierre » -> « JEAN PIERRE »."""
    def norm(v: object) -> object:
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        v = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().upper()
        v = "".join(ch if ch.isalpha() else " " for ch in v)
        return " ".join(v.split()) or pd.NA
    return s.map(norm).astype("string")


def add_name_keys(df: pd.DataFrame, nom: str, prenom: str) -> pd.DataFrame:
    """Ajoute nom_norm, prenom_norm, prenom1 (premier prénom) et nom1 (premier mot du nom)."""
    df = df.copy()
    df["nom_norm"] = normalize_name(df[nom])
    df["prenom_norm"] = normalize_name(df[prenom])
    df["prenom1"] = df["prenom_norm"].str.split(" ").str[0]
    df["nom1"] = df["nom_norm"].str.split(" ").str[0]
    return df


def bloc_politique(nuance: pd.Series) -> pd.Series:
    code = nuance.astype("string").str.upper().str.removeprefix("L")
    return code.map(BLOCS).astype("string")


def apparier(gauche: pd.DataFrame, droite: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rattache à chaque ligne de ``gauche`` (un élu par commune) la ligne de ``droite``
    (une liste) dont la tête porte le même nom, par critères de moins en moins stricts.

    Étapes : nom + prénom complets, puis nom + premier prénom, puis premier mot du
    nom + premier prénom, enfin premier prénom + au moins un mot du nom en commun
    (noms d'usage : « MAGNON MICOLAS » / « MICOLAS », « LEFRANC CARBONNEL » /
    « CARBONNEL LEFRANC »), enfin nom et prénom chacun très proches (similarité
    ≥ 0,85 : fautes de frappe du type FRANCK / FRANK, BANBRY / BANDRY) et même
    sexe quand il est connu des deux côtés (MICHEL / MICHELE). À chaque étape on
    n'accepte que les correspondances uniques dans la commune.
    ``methode_appariement`` garde la trace de l'étape qui a réussi (NA si aucune).
    Limite connue : un changement de nom complet (mariage) n'est pas rattrapé.
    """
    etapes = [("nom_prenom", ["nom_norm", "prenom_norm"]),
              ("nom_prenom1", ["nom_norm", "prenom1"]),
              ("nom1_prenom1", ["nom1", "prenom1"])]
    out = gauche.reset_index(drop=True)
    out["methode_appariement"] = pd.Series(pd.NA, index=out.index, dtype="string")
    for c in cols:
        out[c] = pd.NA
    for methode, cles in etapes:
        cles = ["code_insee"] + cles
        cand = droite.dropna(subset=cles).drop_duplicates(subset=cles, keep=False)[cles + cols]
        a_faire = out["methode_appariement"].isna()
        m = out.loc[a_faire, cles].rename_axis("ligne").reset_index().merge(cand, on=cles, how="inner").set_index("ligne")
        out.loc[m.index, cols] = m[cols]
        out.loc[m.index, "methode_appariement"] = methode

    cles = ["code_insee", "prenom1"]
    a_faire = out["methode_appariement"].isna()
    m = (out.loc[a_faire, cles + ["nom_norm"]].rename_axis("ligne").reset_index()
         .merge(droite.dropna(subset=cles + ["nom_norm"])[cles + ["nom_norm"] + cols],
                on=cles, suffixes=("", "_d")))
    commun = np.array([bool(set(a.split()) & set(b.split())) for a, b in zip(m["nom_norm"], m["nom_norm_d"])],
                      dtype=bool)
    m = m[commun].drop_duplicates("ligne", keep=False).set_index("ligne")
    out.loc[m.index, cols] = m[cols]
    out.loc[m.index, "methode_appariement"] = "nom_partiel_prenom1"

    a_faire = out["methode_appariement"].isna()
    avec_sexe = "sexe" in out.columns and "sexe" in droite.columns
    cles = ["code_insee", "nom_norm", "prenom_norm"] + (["sexe"] if avec_sexe else [])
    m = (out.loc[a_faire, cles].rename_axis("ligne").reset_index()
         .merge(droite.dropna(subset=["nom_norm", "prenom_norm"])[cles + cols],
                on="code_insee", suffixes=("", "_d")))
    m = m.dropna(subset=["nom_norm", "prenom_norm"])

    def proches(a: pd.Series, b: pd.Series) -> np.ndarray:
        return np.array([SequenceMatcher(None, x, y).ratio() >= SEUIL_SIMILARITE for x, y in zip(a, b)],
                        dtype=bool)

    proche = proches(m["nom_norm"], m["nom_norm_d"]) & proches(m["prenom_norm"], m["prenom_norm_d"])
    if avec_sexe:
        sexe_different = (m["sexe"].notna() & m["sexe_d"].notna() & (m["sexe"] != m["sexe_d"])).to_numpy(dtype=bool)
        proche &= ~sexe_different
    m = m[proche].drop_duplicates("ligne", keep=False).set_index("ligne")
    out.loc[m.index, cols] = m[cols]
    out.loc[m.index, "methode_appariement"] = "similarite"
    return out


# --------------------------------------------------------------------------
# Résultats par liste
# --------------------------------------------------------------------------
def resultats_listes(annee: int) -> pd.DataFrame:
    """Une ligne par commune x tour x liste, avec la tête de liste et son rang.

    Garde les communes au scrutin de liste, hors PLM et sections de Polynésie.
    """
    m = read_processed(f"municipales_{annee}")
    m = m[(m["mode_scrutin"] == "liste") & m["secteur"].isna() & m["code_insee"].notna()
          & ~m["code_insee"].isin(PLM)]
    cols = ["code_insee", "tour", "libelle_commune", "inscrits", "votants", "exprimes", "num_panneau",
            "nuance", "nom", "prenom", "liste", "voix", "pct_voix_exp", "sieges_cm"]
    if "num_panneau" not in m.columns:  # 2014 : numéro d'ordre dans le fichier
        m["num_panneau"] = (m.groupby(["code_insee", "tour"]).cumcount() + 1).astype("string")
    m = m[cols].copy()
    if annee == 2026:
        # Le nom de la tête de liste vient du fichier des candidatures.
        c = read_processed("candidatures_2026")
        tetes = (c[c["tete_de_liste"]]
                 .rename(columns={"numero_de_panneau": "num_panneau",
                                  "nom_sur_le_bulletin_de_vote": "nom_tete",
                                  "prenom_sur_le_bulletin_de_vote": "prenom_tete"})
                 [["code_insee", "tour", "num_panneau", "nom_tete", "prenom_tete"]]
                 .drop_duplicates(subset=["code_insee", "tour", "num_panneau"]))
        m = m.merge(tetes, on=["code_insee", "tour", "num_panneau"], how="left")
        m["nom"] = m["nom"].fillna(m.pop("nom_tete"))
        m["prenom"] = m["prenom"].fillna(m.pop("prenom_tete"))
    m = add_name_keys(m, "nom", "prenom")
    m["bloc"] = bloc_politique(m["nuance"])
    m["rang"] = m.groupby(["code_insee", "tour"])["voix"].rank(method="min", ascending=False).astype("Int64")
    return m


def candidats_t1(annee: int) -> pd.DataFrame:
    """Tous les candidats du 1er tour (pas seulement les têtes de liste), avec leur
    liste (numéro de panneau) et leur rang sur la liste. 2020 : communes ≥ 1000 hab."""
    if annee == 2020:
        c = read_processed("candidatures_2020").rename(columns={
            "n_panneau_liste": "num_panneau", "n_candidat": "ordre",
            "nom_candidat": "nom", "prenom_candidat": "prenom", "sexe_candidat": "sexe"})
        c = c[c["secteur"].isna()]
    elif annee == 2026:
        c = read_processed("candidatures_2026").rename(columns={
            "numero_de_panneau": "num_panneau", "nom_sur_le_bulletin_de_vote": "nom",
            "prenom_sur_le_bulletin_de_vote": "prenom"})
        c = c[c["tour"] == 1]
    else:
        raise ValueError(f"pas de fichier de candidatures pour {annee}")
    c = c[["code_insee", "num_panneau", "ordre", "nom", "prenom", "sexe"]].copy()
    c["ordre"] = pd.to_numeric(c["ordre"], errors="coerce").astype("Int64")
    return add_name_keys(c, "nom", "prenom")


def listes_gagnantes(listes: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par commune : la liste qui obtient le plus de sièges au dernier tour
    (prime majoritaire), départagée par les voix.

    Les numéros de panneau changent entre les deux tours : la liste gagnante est
    donc repérée par le nom de sa tête de liste, pas par son panneau.
    """
    dernier = listes["tour"] == listes.groupby("code_insee")["tour"].transform("max")
    return (listes[dernier].sort_values(["sieges_cm", "voix"], ascending=False)
            .drop_duplicates("code_insee"))


def bilan_communes(listes: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par commune : participation au 1er tour, nombre de listes, tête de la liste gagnante."""
    t1 = listes[listes["tour"] == 1]
    out = t1.groupby("code_insee").agg(
        libelle_commune=("libelle_commune", "first"), inscrits=("inscrits", "first"),
        votants=("votants", "first"), n_listes=("liste", "size"))
    out["participation"] = out["votants"] / out["inscrits"]
    g = listes_gagnantes(listes).set_index("code_insee")
    out["second_tour"] = g["tour"].eq(2)
    out["nom1_gagnant"] = g["nom1"]
    out["prenom1_gagnant"] = g["prenom1"]
    return out.reset_index()


# --------------------------------------------------------------------------
# Maires sortants
# --------------------------------------------------------------------------
def sortants_2026() -> pd.DataFrame:
    s = read_processed("maires_sortants_2026")
    s = s.rename(columns={"date_de_naissance": "date_naissance",
                          "date_de_debut_de_la_fonction": "date_debut_fonction"})
    s = s[["code_insee", "nom", "prenom", "sexe", "date_naissance", "date_debut_fonction"]]
    s = s[~s["code_insee"].isin(PLM)].drop_duplicates("code_insee")
    return add_name_keys(s, "nom", "prenom").assign(source_sortant="maires_sortants_2026")


def sortants_2020(listes_2014: pd.DataFrame) -> pd.DataFrame:
    """Approximation : tête de la liste gagnante en 2014 (voir docstring du module)."""
    g = listes_gagnantes(listes_2014)
    s = g[["code_insee", "nom", "prenom", "nom_norm", "prenom_norm", "prenom1", "nom1"]].copy()
    s["sexe"] = pd.NA
    s["date_naissance"] = pd.NaT
    s["date_debut_fonction"] = pd.NaT
    return s.assign(source_sortant="tete_liste_gagnante_2014")


def communes_perimetre_modifie(debut: str, fin: str) -> set[str]:
    """Codes des communes touchées par une fusion ou un rétablissement entre deux dates."""
    mv = read_processed("cog_mouvements")
    mv = mv[mv["mod"].isin(MOD_PERIMETRE) & (mv["date_eff"] > debut) & (mv["date_eff"] <= fin)]
    return set(mv["com_av"].dropna()) | set(mv["com_ap"].dropna())


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
COLS_LISTE = ["num_panneau", "nuance", "bloc", "voix", "pct_voix_exp", "rang", "sieges_cm"]


def panel_election(annee: int, annee_prec: int, sortants: pd.DataFrame,
                   listes: pd.DataFrame, listes_prec: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par commune ayant voté au scrutin de liste en ``annee``.

    Variables du sortant : se représente-t-il (sur une liste au 1er tour, à
    n'importe quelle place : dans les petites communes le maire n'est pas
    toujours tête de liste) ? Est-il tête de liste ? Score, rang, nuance de sa
    liste ; sa liste l'emporte-t-elle ? Pour l'élection précédente, on ne
    connaît que les têtes de liste (pas de fichier de candidatures en 2014) :
    on y cherche la liste que le sortant conduisait, d'où les écarts ``delta_*``.
    """
    bilan = bilan_communes(listes)
    bilan_prec = bilan_communes(listes_prec)
    t1 = listes[listes["tour"] == 1]
    t1_prec = listes_prec[listes_prec["tour"] == 1]

    # Univers : communes votant au scrutin de liste en `annee`, ayant un sortant connu.
    p = sortants.merge(bilan, on="code_insee", how="inner")
    p.insert(0, "annee", annee)

    # Liste sur laquelle figure le sortant à cette élection.
    p = apparier(p, candidats_t1(annee), ["num_panneau", "ordre"])
    p["sortant_candidat"] = p["methode_appariement"].notna()
    p["sortant_tete_liste"] = (pd.to_numeric(p.pop("ordre")) == 1).where(p["sortant_candidat"])
    resultats = (t1[["code_insee"] + COLS_LISTE + ["nom1", "prenom1"]]
                 .rename(columns={"nom1": "nom1_tete", "prenom1": "prenom1_tete"})
                 .drop_duplicates(subset=["code_insee", "num_panneau"], keep=False))
    p = p.merge(resultats, on=["code_insee", "num_panneau"], how="left")
    gagne = (p["nom1_tete"] == p["nom1_gagnant"]) & (p["prenom1_tete"] == p["prenom1_gagnant"])
    p["liste_sortant_gagnante"] = gagne.where(p["sortant_candidat"] & p["nom1_tete"].notna())
    p = p.drop(columns=["nom1_gagnant", "prenom1_gagnant", "nom1_tete", "prenom1_tete"])

    # Liste conduite par le sortant à l'élection précédente.
    prec = apparier(sortants[["code_insee", "nom_norm", "prenom_norm", "prenom1", "nom1"]], t1_prec,
                    COLS_LISTE)
    prec = prec.drop(columns=["nom_norm", "prenom_norm", "prenom1", "nom1"])
    prec["tete_liste_prec"] = prec.pop("methode_appariement").notna()
    p = p.merge(prec.add_suffix("_prec").rename(columns={"code_insee_prec": "code_insee",
                                                         "tete_liste_prec_prec": "tete_liste_prec"}),
                on="code_insee", how="left")
    p = p.merge(bilan_prec[["code_insee", "participation", "n_listes"]].add_suffix("_prec")
                .rename(columns={"code_insee_prec": "code_insee"}), on="code_insee", how="left")
    p["tete_liste_prec"] = p["tete_liste_prec"].fillna(False)

    for c in ("pct_voix_exp", "pct_voix_exp_prec", "voix", "voix_prec", "rang", "rang_prec"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["delta_pct_voix_exp"] = p["pct_voix_exp"] - p["pct_voix_exp_prec"]
    p["delta_participation"] = p["participation"] - p["participation_prec"]

    modifiees = communes_perimetre_modifie(DATES_PREMIER_TOUR[annee_prec], DATES_PREMIER_TOUR[annee])
    p["perimetre_modifie"] = p["code_insee"].isin(modifiees)
    return p


def reelus_2026(panel: pd.DataFrame) -> np.ndarray:
    """Le sortant est-il maire en août 2026 (RNE) ?

    Le RNE compte un maire par commune : on compare le sortant à ce maire. Même
    personne si (même date de naissance et un mot du nom ou le premier prénom en
    commun) ou (même nom et même premier prénom) : la seconde règle tolère les
    erreurs de saisie de date, la première les changements de nom d'usage.
    """
    rne = add_name_keys(read_processed("rne_maires"), "nom", "prenom")
    rne = rne.rename(columns={"date_de_naissance": "date_naissance"}).drop_duplicates("code_insee")
    m = panel[["code_insee", "nom_norm", "prenom1", "date_naissance"]].merge(
        rne[["code_insee", "nom_norm", "prenom1", "date_naissance"]], on="code_insee", how="left",
        suffixes=("", "_rne"))
    nom_commun = np.array([isinstance(a, str) and isinstance(b, str) and bool(set(a.split()) & set(b.split()))
                           for a, b in zip(m["nom_norm"], m["nom_norm_rne"])], dtype=bool)
    meme_prenom = (m["prenom1"] == m["prenom1_rne"]).fillna(False).to_numpy(dtype=bool)
    meme_ddn = (m["date_naissance"] == m["date_naissance_rne"]).fillna(False).to_numpy(dtype=bool)
    meme_nom = (m["nom_norm"] == m["nom_norm_rne"]).fillna(False).to_numpy(dtype=bool)
    return (meme_ddn & (nom_commun | meme_prenom)) | (meme_nom & meme_prenom)


def build_panel() -> pd.DataFrame:
    listes = {a: resultats_listes(a) for a in (2014, 2020, 2026)}
    p20 = panel_election(2020, 2014, sortants_2020(listes[2014]), listes[2020], listes[2014])
    p26 = panel_election(2026, 2020, sortants_2026(), listes[2026], listes[2020])
    p26["reelu_maire"] = pd.array(reelus_2026(p26), dtype="boolean")
    p20["reelu_maire"] = pd.array([pd.NA] * len(p20), dtype="boolean")  # pas de RNE de 2020
    # Colonnes vides en 2020 (sexe, date de naissance... du sortant, inconnus) :
    # on leur donne le type de 2026 pour que la concaténation garde ce type.
    for c in p20.columns[p20.isna().all()]:
        if c in p26:
            p20[c] = p20[c].astype(p26[c].dtype)
    panel = pd.concat([p20, p26], ignore_index=True)
    for c in ("sortant_candidat", "sortant_tete_liste", "liste_sortant_gagnante", "tete_liste_prec",
              "perimetre_modifie", "second_tour"):
        panel[c] = panel[c].astype("boolean")
    for c in ("num_panneau", "nuance", "bloc", "num_panneau_prec", "nuance_prec", "bloc_prec"):
        panel[c] = panel[c].astype("string")
    for c in ("sieges_cm", "sieges_cm_prec", "n_listes", "n_listes_prec"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("Int64")
    return panel


def resume(panel: pd.DataFrame) -> pd.DataFrame:
    """Taux clés par élection, pour contrôler la qualité des appariements."""
    return panel.groupby("annee").agg(
        communes=("code_insee", "size"),
        sortant_candidat=("sortant_candidat", "mean"),
        sortant_tete_liste=("sortant_tete_liste", "mean"),
        liste_gagnante_si_candidat=("liste_sortant_gagnante", "mean"),
        tete_liste_prec=("tete_liste_prec", "mean"),
        delta_pct_disponible=("delta_pct_voix_exp", lambda s: s.notna().mean()),
        perimetre_modifie=("perimetre_modifie", "mean"),
    ).round(3)


PANEL_PATH = PROCESSED_DIR / "panel_sortants.parquet"


def write_panel() -> pd.DataFrame:
    panel = build_panel()
    panel.to_parquet(PANEL_PATH, index=False)
    log.info("panel : %d lignes x %d colonnes -> %s", len(panel), panel.shape[1], PANEL_PATH.relative_to(ROOT))
    return panel


def load_panel() -> pd.DataFrame:
    """Le panel enregistré, construit (et enregistré) s'il n'existe pas encore."""
    return pd.read_parquet(PANEL_PATH) if PANEL_PATH.exists() else write_panel()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    panel = write_panel()
    log.info("résumé :\n%s", resume(panel).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

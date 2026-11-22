"""Affaires judiciaires des maires sortants : rattachement au panel.

Il n'existe pas d'open data officiel nominatif sur les procédures visant des
élus. On combine deux bases citoyennes (voir scripts/download_data.py) :

- Casier Politique : condamnations seulement, extraites de Wikipédia ;
- Poligraph : affaires à tous les stades (enquête, mise en examen, procès,
  condamnation, relaxe), avec date de début de la procédure et du verdict.

Étapes :
1. harmoniser les deux sources en une table « une ligne par affaire » ;
2. retrouver chaque personnalité parmi les maires sortants du panel
   (src/panel.py) : même nom et premier prénom, PUIS confirmation par la date
   de naissance ou par le nom de la commune cité dans sa fiche (homonymes) ;
3. pour chaque ligne du panel (sortant x élection), dater les affaires par
   rapport au 1er tour : connue avant le scrutin, révélée pendant le mandat
   sortant, ou seulement après (ces dernières servent de groupe placebo).

Date « publique » d'une affaire : date de début de la procédure si elle est
connue, sinon date du verdict (Poligraph) ; date de condamnation (Casier
Politique). C'est une borne haute : l'affaire a souvent été révélée par la
presse plus tôt. Une date au 1er janvier est en général une année seule ; si
c'est l'année du scrutin, on ne sait pas si elle précède le vote (« ambiguë »).

Usage :
    python src/affaires.py   # écrit data/processed/panel_affaires.parquet
                             # et data/processed/affaires_sortants.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402

log = logging.getLogger("affaires")

# Atteintes à la probité au sens du code pénal (art. 432-10 à 432-16) : corruption,
# trafic d'influence, prise illégale d'intérêts, favoritisme, détournement de fonds
# publics (dont emplois fictifs). Même périmètre que la catégorie de Casier Politique.
PROBITE_CASIER = {"Atteinte à la probité"}
PROBITE_POLIGRAPH = {
    "CORRUPTION", "CORRUPTION_PASSIVE", "TRAFIC_INFLUENCE", "PRISE_ILLEGALE_INTERETS",
    "FAVORITISME", "DETOURNEMENT_FONDS_PUBLICS", "EMPLOI_FICTIF",
}

# Statuts Poligraph regroupés en trois issues.
STATUTS_CONDAMNATION = {
    "CONDAMNATION_PREMIERE_INSTANCE", "APPEL_EN_COURS", "CONDAMNATION_EN_APPEL",
    "POURVOI_EN_CASSATION", "CONDAMNATION_DEFINITIVE",
}
STATUTS_ABANDON = {
    "RELAXE", "ACQUITTEMENT", "NON_LIEU", "CLASSEMENT_SANS_SUITE", "PRESCRIPTION",
    "INSTRUCTION_CLOTUREE_SANS_MISE_EN_EXAMEN",
}  # tout le reste : procédure en cours


def _issue(statut_code: pd.Series) -> pd.Series:
    issue = pd.Series("en_cours", index=statut_code.index, dtype="string")
    issue[statut_code.isin(STATUTS_CONDAMNATION)] = "condamnation"
    issue[statut_code.isin(STATUTS_ABANDON)] = "abandon"
    return issue


def _date_precise(d: pd.Series) -> pd.Series:
    return d.notna() & ~((d.dt.month == 1) & (d.dt.day == 1))


def _texte(*cols: pd.Series) -> pd.Series:
    """Concatène des colonnes de texte, normalisées comme les noms (majuscules, sans accents)."""
    joint = pd.concat([c.astype("string").fillna("") for c in cols], axis=1).agg(" ".join, axis=1)
    return P.normalize_name(joint).fillna("")


# --------------------------------------------------------------------------
# 1. Harmonisation
# --------------------------------------------------------------------------
def affaires_casier() -> pd.DataFrame:
    c = P.read_processed("casier_politique")
    return pd.DataFrame({
        "source": "casier_politique",
        "personne_id": c["personne_id"],
        "intitule": c["affaire"],
        "categorie": c["categories"],
        "probite": c["categories"].str.split("; ").map(lambda cs: bool(set(cs) & PROBITE_CASIER)),
        "issue": "condamnation",
        "date_publique": c["date_condamnation"],
        "date_condamnation": c["date_condamnation"],
        "liee_au_mandat": pd.NA,
        "texte": _texte(c["affaire"], c["description"]),
    })


def affaires_poligraph() -> pd.DataFrame:
    a = P.read_processed("poligraph_affaires")
    a = a[a["implication_code"] == "DIRECT"]  # mis en cause (ni plaignant, ni victime, ni témoin)
    issue = _issue(a["statut_code"])
    return pd.DataFrame({
        "source": "poligraph",
        "personne_id": a["personne_id"],
        "intitule": a["titre"],
        "categorie": a["categorie"],
        "probite": a["categorie_code"].isin(PROBITE_POLIGRAPH),
        "issue": issue,
        "date_publique": a["date_debut"].fillna(a["date_verdict"]),
        "date_condamnation": a["date_verdict"].where(issue == "condamnation"),
        "liee_au_mandat": a["liee_au_mandat"].map({"oui": True, "non": False}),
        "texte": _texte(a["titre"], a["description"]),
    })


def personnes() -> pd.DataFrame:
    """Une ligne par personnalité et par source : nom, date de naissance, texte de sa fiche."""
    c = P.read_processed("casier_politique").drop_duplicates("personne_id")
    m = P.read_processed("casier_politique_mandats")
    lieux = m.assign(t=m["label"].fillna("") + " " + m["lieu"].fillna("")).groupby("personne_id")["t"].agg(" ".join)
    casier = pd.DataFrame({
        "source": "casier_politique", "personne_id": c["personne_id"], "nom_complet": c["nom_complet"],
        "date_naissance": c["date_naissance"],
        "texte": _texte(c["personne_id"].map(lieux)),
    })
    p = P.read_processed("poligraph_politiques")
    poligraph = pd.DataFrame({
        "source": "poligraph", "personne_id": p["personne_id"],
        "nom_complet": p["prenom"] + " " + p["nom"], "date_naissance": p["date_de_naissance"],
        "texte": _texte(p["titre_du_mandat"], p["circonscription"]),
    })
    return pd.concat([casier, poligraph], ignore_index=True)


def toutes_affaires() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(personnes, affaires) ; le texte des affaires est ajouté à celui de la personne."""
    aff = pd.concat([affaires_casier(), affaires_poligraph()], ignore_index=True)
    aff["date_precise"] = _date_precise(aff["date_publique"])
    pers = personnes()
    texte_aff = aff.groupby(["source", "personne_id"])["texte"].agg(" ".join)
    cle = pd.MultiIndex.from_frame(pers[["source", "personne_id"]])
    pers["texte"] = pers["texte"] + " " + texte_aff.reindex(cle).fillna("").to_numpy()
    return pers, aff.drop(columns="texte")


# --------------------------------------------------------------------------
# 2. Rattachement aux maires sortants
# --------------------------------------------------------------------------
def libelles_communes(panel: pd.DataFrame) -> pd.Series:
    """code_insee -> ensemble des libellés normalisés : COG (noms actuels et anciens)
    et fichiers de résultats (seuls à couvrir la Nouvelle-Calédonie et la Polynésie)."""
    cog = P.read_processed("cog_historique")
    cog = cog[cog["typecom"] == "COM"]
    codes = pd.concat([cog["code_insee"], panel["code_insee"]], ignore_index=True)
    libs = P.normalize_name(pd.concat([cog["libelle"], panel["libelle_commune"]], ignore_index=True))
    return pd.Series(libs.to_numpy(), index=codes.to_numpy()).dropna().groupby(level=0).agg(set)


def apparier_sortants(panel: pd.DataFrame, pers: pd.DataFrame) -> pd.DataFrame:
    """Couples (personnalité, sortant) confirmés. Une ligne par (source, personne, annee, commune).

    Candidat : nom complet de la personnalité = premier prénom + nom du sortant
    (ou prénoms complets + nom). Confirmé si même date de naissance, ou si le nom
    de la commune figure dans la fiche de la personnalité (mandats, affaires).
    Un couple (nom, commune) confirmé par une source ou une année l'est pour
    toutes : le sortant 2020 n'a pas de date de naissance connue, et une même
    personne figure souvent dans les deux sources.

    Enfin, on écarte les « faux sortants » : le sortant 2020 est approché par la
    tête de la liste gagnante de 2014 (src/panel.py), or beaucoup d'élus
    nationaux ont quitté leur mairie en cours de mandat (loi de 2014 sur le
    cumul, appliquée en 2017). Les mandats de maire datés par Casier Politique
    permettent de le repérer (``encore_maire`` = False).
    """
    pers = pers.assign(full=P.normalize_name(pers["nom_complet"]))
    cles = pd.concat([
        panel.assign(full=panel["prenom1"] + " " + panel["nom_norm"]),
        panel.assign(full=panel["prenom_norm"] + " " + panel["nom_norm"]),
    ]).drop_duplicates(["annee", "code_insee", "full"])
    cand = pers.merge(cles[["annee", "code_insee", "full", "date_naissance"]], on="full",
                      suffixes=("", "_sortant"))

    libelles = libelles_communes(panel)
    # ±1 jour : certaines dates de Poligraph sont décalées d'un jour (conversion UTC).
    ecart = (cand["date_naissance"] - cand["date_naissance_sortant"]).abs()
    cand["meme_naissance"] = (ecart <= pd.Timedelta(days=1)).fillna(False)
    cand["commune_citee"] = [
        any(f" {lib} " in f" {texte} " for lib in libelles.get(code, ()))
        for code, texte in zip(cand["code_insee"], cand["texte"])
    ]
    ok = cand.loc[cand["meme_naissance"] | cand["commune_citee"], ["full", "code_insee"]]
    identifie = pd.MultiIndex.from_frame(cand[["full", "code_insee"]]).isin(pd.MultiIndex.from_frame(ok))
    cand["encore_maire"] = encore_maire(cand, libelles)
    cand["confirme"] = identifie & cand["encore_maire"].fillna(True).to_numpy(dtype=bool)  # NA = inconnu : gardé
    cand["motif_rejet"] = pd.Series(pd.NA, index=cand.index, dtype="string")
    cand.loc[~identifie, "motif_rejet"] = "homonyme probable"
    cand.loc[identifie & ~cand["confirme"], "motif_rejet"] = "plus maire au scrutin"
    log.info("candidats : %d couples, %d confirmés ; rejetés : %s", len(cand), int(cand["confirme"].sum()),
             cand["motif_rejet"].value_counts().to_dict())
    return cand.drop(columns=["full", "texte", "date_naissance_sortant"])


MARGE_FIN_MANDAT = pd.Timedelta(days=180)  # fins de mandat datées approximativement


def encore_maire(cand: pd.DataFrame, libelles: pd.Series) -> pd.Series:
    """Le candidat était-il maire de la commune au 1er tour ? (booléen, NA si pas de mandat daté).

    Mandats « maire » de Casier Politique dont le lieu cite la commune ; maire au
    scrutin si l'un d'eux a commencé avant le vote et n'a pas fini plus de
    6 mois avant.
    """
    m = P.read_processed("casier_politique_mandats")
    m = m[m["type"] == "maire"].assign(full=lambda d: P.normalize_name(d["nom_complet"]),
                                       lieu_n=lambda d: _texte(d["label"], d["lieu"]))
    par_nom = dict(tuple(m.groupby("full")))
    out = []
    for full, code, annee in zip(cand["full"], cand["code_insee"], cand["annee"]):
        mandats = par_nom.get(full)
        if mandats is None:
            out.append(pd.NA)
            continue
        ici = mandats[[any(f" {lib} " in f" {t} " for lib in libelles.get(code, ())) for t in mandats["lieu_n"]]]
        if ici.empty:
            out.append(pd.NA)
            continue
        vote = pd.Timestamp(P.DATES_PREMIER_TOUR[annee])
        couvre = (ici["debut"].isna() | (ici["debut"] <= vote)) & \
                 (ici["fin"].isna() | (ici["fin"] >= vote - MARGE_FIN_MANDAT))
        out.append(bool(couvre.any()))
    return pd.Series(out, index=cand.index, dtype="boolean")


# --------------------------------------------------------------------------
# 3. Variables de traitement
# --------------------------------------------------------------------------
def dater(aff: pd.DataFrame) -> pd.DataFrame:
    """Position de chaque affaire par rapport au 1er tour de l'élection (colonne ``annee``)."""
    d = aff["date_publique"]
    vote = pd.to_datetime(aff["annee"].map(P.DATES_PREMIER_TOUR))
    prec = pd.to_datetime((aff["annee"] - 6).map(P.DATES_PREMIER_TOUR))
    ambigue = ~aff["date_precise"] & (d.dt.year == vote.dt.year)
    aff = aff.copy()
    aff["moment"] = pd.Series(pd.NA, index=aff.index, dtype="string")
    aff.loc[d.notna() & (d < vote) & (d < prec), "moment"] = "avant_mandat"
    aff.loc[d.notna() & (d < vote) & (d >= prec), "moment"] = "pendant_mandat"
    aff.loc[d.notna() & (d >= vote), "moment"] = "apres_scrutin"
    aff.loc[ambigue, "moment"] = "ambigue"
    aff.loc[d.isna(), "moment"] = "sans_date"
    cond = aff["date_condamnation"]
    aff["condamne_avant"] = cond.notna() & (cond < vote) & ~(~_date_precise(cond) & (cond.dt.year == vote.dt.year))
    return aff


def variables_traitement(aff_sortants: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (annee, code_insee) ayant au moins une affaire rattachée."""
    a = aff_sortants
    connue = a["moment"].isin(["avant_mandat", "pendant_mandat"])
    flags = pd.DataFrame({
        "annee": a["annee"], "code_insee": a["code_insee"],
        "affaire_connue": connue,
        "affaire_mandat": a["moment"] == "pendant_mandat",
        "affaire_probite_connue": connue & a["probite"],
        "affaire_probite_mandat": (a["moment"] == "pendant_mandat") & a["probite"],
        "condamne_avant": a["condamne_avant"],
        "affaire_apres": a["moment"] == "apres_scrutin",
        "affaire_ambigue": a["moment"].isin(["ambigue", "sans_date"]),
    })
    out = flags.groupby(["annee", "code_insee"]).any()
    out["n_affaires_connues"] = connue.groupby([a["annee"], a["code_insee"]]).sum()
    out["premiere_affaire"] = a["date_publique"].where(connue).groupby([a["annee"], a["code_insee"]]).min()
    return out.reset_index()


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = P.load_panel()
    pers, aff = toutes_affaires()
    couples = apparier_sortants(panel, pers)
    ok = couples[couples["confirme"]]
    aff_sortants = dater(ok[["source", "personne_id", "annee", "code_insee"]].merge(aff, on=["source", "personne_id"]))

    traite = variables_traitement(aff_sortants)
    p = panel.merge(traite, on=["annee", "code_insee"], how="left")
    bool_cols = [c for c in traite.columns if c not in ("annee", "code_insee", "n_affaires_connues", "premiere_affaire")]
    for c in bool_cols:
        p[c] = p[c].astype("boolean").fillna(False)
    p["n_affaires_connues"] = p["n_affaires_connues"].fillna(0).astype("Int64")
    p["personnalite_affaires"] = pd.MultiIndex.from_frame(p[["annee", "code_insee"]]).isin(
        pd.MultiIndex.from_frame(ok[["annee", "code_insee"]]))
    return p, aff_sortants


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    p, aff_sortants = build()
    p.to_parquet(P.PROCESSED_DIR / "panel_affaires.parquet", index=False)
    aff_sortants.to_parquet(P.PROCESSED_DIR / "affaires_sortants.parquet", index=False)
    cols = ["personnalite_affaires", "affaire_connue", "affaire_mandat", "affaire_probite_mandat",
            "condamne_avant", "affaire_apres", "affaire_ambigue"]
    log.info("sortants concernés par élection :\n%s", p.groupby("annee")[cols].sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

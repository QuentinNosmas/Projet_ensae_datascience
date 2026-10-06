"""Chargement et nettoyage des sources brutes (data/raw) vers data/processed.

Chaque fonction ``load_*`` lit un fichier brut avec l'encodage et le séparateur
consignés dans data/MANIFEST.json, force les codes INSEE en chaînes de
5 caractères (zéros de tête, 2A/2B, outre-mer) et renvoie un DataFrame aux
colonnes en snake_case. Aucune jointure entre sources n'est faite ici.

Usage :
    python src/load.py              # écrit un parquet par source dans data/processed/
    python src/load.py --only municipales_2014 cog_communes
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "data" / "MANIFEST.json"

log = logging.getLogger("load")

# Codes « département » utilisés par le ministère de l'Intérieur pour l'outre-mer,
# et préfixe à 3 chiffres du code INSEE correspondant. Le code commune du
# ministère n'est pas stable (Mayotte « 501 » en 2014 et « 601 » en 2020,
# Polynésie « 012 » en 2014 et « 712 » en 2020) : seuls ses 2 derniers
# chiffres sont fiables, d'où préfixe + 2 derniers chiffres.
OUTRE_MER_PREFIX = {
    "ZA": "971",  # Guadeloupe
    "ZB": "972",  # Martinique
    "ZC": "973",  # Guyane
    "ZD": "974",  # La Réunion
    "ZS": "975",  # Saint-Pierre-et-Miquelon
    "ZM": "976",  # Mayotte
    "ZW": "986",  # Wallis-et-Futuna
    "ZP": "987",  # Polynésie française
    "ZN": "988",  # Nouvelle-Calédonie
}
# Saint-Barthélemy (97701) et Saint-Martin (97801) : code commune « 701 » / « 801 ».
OUTRE_MER_97_PLUS_COM = {"ZX"}
# Français établis hors de France : pas de commune.
HORS_COMMUNE = {"ZZ"}


# --------------------------------------------------------------------------
# Utilitaires génériques
# --------------------------------------------------------------------------
def snake_case(name: str) -> str:
    """« % Voix/Exp » -> « pct_voix_exp », « Libellé de la commune » -> « libelle_de_la_commune »."""
    s = name.replace("%", " pct ").replace("°", " ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "col"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case + suffixe numérique pour les noms en double."""
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        base = snake_case(str(c))
        n = seen.get(base, 0)
        cols.append(base if n == 0 else f"{base}_{n}")
        seen[base] = n + 1
    return df.set_axis(cols, axis=1)


def manifest_entry(source: str, member: str | None = None) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if source not in manifest:
        raise KeyError(f"{source} absent de MANIFEST.json : lancer scripts/download_data.py --only {source}")
    entry = manifest[source]
    if member is None:
        return entry
    for e in entry.get("extracted", []):
        if e["path"].endswith(member):
            return e
    raise KeyError(f"{member} introuvable dans l'archive {source}")


def raw_file(source: str, member: str | None = None) -> tuple[Path, str | None, str | None]:
    """(chemin, encodage, séparateur) d'après le manifeste."""
    e = manifest_entry(source, member)
    return ROOT / e["path"], e.get("encoding"), e.get("separator")


def to_num(s: pd.Series) -> pd.Series:
    """Nombres « à la française » (virgule décimale, zéros de tête, « 18,86% », vide) -> float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    s = (s.astype("string").str.strip().str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False).str.rstrip("%").str.strip())
    return pd.to_numeric(s.replace("", pd.NA), errors="coerce").astype("float64")


def to_int(s: pd.Series) -> pd.Series:
    return to_num(s).round().astype("Int64")


def clean_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s == "")


def _code_part(value: object) -> str:
    """Normalise une cellule de code : « 1.0 » (xls) -> « 1 », espaces retirés."""
    v = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    if re.fullmatch(r"\d+\.0", v):
        v = v[:-2]
    return v


def code_insee(dep: object, com: object) -> str | None:
    """Code INSEE à 5 caractères à partir des codes département / commune du ministère.

    Gère : codes déjà complets (2026, RNE), zéros de tête perdus (« 1 » / « 4 »
    -> « 01004 »), Corse (2A/2B), outre-mer (ZA..ZX), et les suffixes de
    secteur / section (« 056SR01 », « 013SN01 ») dont on ne garde que la commune.
    """
    d, c = _code_part(dep), _code_part(com)
    if not c or d in HORS_COMMUNE:
        return None
    if len(c) == 5 and (c[:2].isdigit() or c[:2] in ("2A", "2B")):
        return c  # déjà un code INSEE complet
    c = c[:3] if len(c) > 3 else c  # retire le suffixe de secteur / section
    c = c.zfill(3)
    if d in OUTRE_MER_PREFIX:
        return OUTRE_MER_PREFIX[d] + c[-2:]
    if d in OUTRE_MER_97_PLUS_COM:
        return "97" + c
    if d.isdigit():
        d = d.zfill(2)
        if len(d) == 3:  # 971..976 : le code commune porte déjà le 3e chiffre
            return d[:2] + c
        return d + c
    if d in ("2A", "2B"):
        return d + c
    return None


def secteur_from_code(com: object) -> str | None:
    """Suffixe de secteur (PLM : « SR01 ») ou de section électorale (Polynésie : « SN01 »)."""
    c = _code_part(com)
    return c[3:] if len(c) > 5 else None


def add_code_insee(df: pd.DataFrame, dep_col: str, com_col: str, with_secteur: bool = False) -> pd.DataFrame:
    """Ajoute ``code_insee`` (et, si demandé, ``secteur`` : PLM « SR01 », Polynésie « SN01 »)."""
    df = df.copy()
    df.insert(0, "code_insee", pd.Series(
        [code_insee(d, c) for d, c in zip(df[dep_col], df[com_col])], index=df.index, dtype="string"))
    if with_secteur:
        df.insert(1, "secteur", pd.Series([secteur_from_code(c) for c in df[com_col]],
                                          index=df.index, dtype="string"))
    return df


def read_csv_raw(source: str, member: str | None = None, **kwargs) -> pd.DataFrame:
    """pandas.read_csv tout en str, avec encodage et séparateur du manifeste."""
    path, enc, sep = raw_file(source, member)
    return pd.read_csv(path, sep=sep, encoding=enc, dtype=str, keep_default_na=False, **kwargs)


def iter_rows(source: str, skip: int = 0) -> tuple[list[str], Iterable[list[str]]]:
    """(en-tête, lignes) d'un fichier texte à lignes de longueur variable."""
    path, enc, sep = raw_file(source)
    fh = path.open(encoding=enc, newline="")
    reader = csv.reader(fh, delimiter=sep)
    for _ in range(skip):
        next(reader)
    header = next(reader)

    def rows() -> Iterable[list[str]]:
        with fh:
            yield from reader

    return header, rows()


NUMERIC_RE = re.compile(r"\s*-?\d+(?:[.,]\d+)?\s*%?\s*")
YES_NO = {"OUI", "NON", "ELU", "ÉLU", "NON ELU", "NON ÉLU"}


def wide_to_long(header: Sequence[str], rows: Iterable[Sequence[object]], n_prefix: int,
                 block_size: int, label_col: str | None = None) -> pd.DataFrame:
    """Passe un fichier « large » au format long.

    Les fichiers de résultats du ministère juxtaposent, sur une même ligne,
    les colonnes de la commune (``n_prefix`` premières colonnes) puis un bloc
    de ``block_size`` colonnes par liste (ou par candidat dans les communes au
    scrutin plurinominal). L'en-tête ne décrit que le premier bloc et finit
    souvent par une colonne vide (séparateur terminal). Pour chaque ligne :
    - on découpe la partie après le préfixe en tranches de ``block_size`` ;
    - on ignore les tranches entièrement vides (séparateurs de fin de ligne) ;
    - une tranche incomplète mais non vide est signalée (ligne mal formée).
    Résultat : une ligne par commune x liste, colonnes = préfixe + bloc.

    Réparation (``label_col``) : quelques libellés de liste contiennent le
    caractère séparateur (une tabulation en 2020), ce qui décale tout le reste
    de la ligne. Si le champ qui suit le libellé n'est pas numérique alors
    qu'il devrait l'être (sièges ou voix), on le recolle au libellé. À
    n'utiliser que lorsque la colonne suivante est toujours numérique.
    """
    prefix_cols = [h.strip() for h in header[:n_prefix]]
    block_cols = [re.sub(r"\s+1$", "", h.strip()) for h in header[n_prefix:n_prefix + block_size]]
    label_idx = block_cols.index(label_col) if label_col else None
    out: list[list[object]] = []
    n_bad = n_repaired = 0
    for row in rows:
        row = list(row)
        if not any(str(x).strip() for x in row):
            continue
        prefix, rest = row[:n_prefix], row[n_prefix:]
        i = 0
        while i < len(rest):
            if label_idx is not None:
                j = i + label_idx
                # Libellé vide ou « Oui » / « Non » à la suite : ligne au scrutin
                # plurinominal (sections de Polynésie), rien à réparer.
                while (j + 1 < len(rest) and str(rest[j]).strip() and str(rest[j + 1]).strip()
                       and not NUMERIC_RE.fullmatch(str(rest[j + 1]))
                       and str(rest[j + 1]).strip().upper() not in YES_NO):
                    rest[j] = f"{rest[j]} {rest.pop(j + 1)}".strip()
                    n_repaired += 1
            block = rest[i:i + block_size]
            i += block_size
            if not any(str(x).strip() for x in block):
                continue
            if len(block) < block_size:
                n_bad += 1
                block = block + [""] * (block_size - len(block))
            out.append(prefix + block)
    if n_repaired:
        log.warning("wide_to_long : %d libellé(s) contenant le séparateur recollé(s)", n_repaired)
    if n_bad:
        log.warning("wide_to_long : %d bloc(s) incomplet(s) complété(s) par du vide", n_bad)
    return pd.DataFrame(out, columns=prefix_cols + block_cols, dtype=object)


# --------------------------------------------------------------------------
# Municipales
# --------------------------------------------------------------------------
# Correspondance des noms (après snake_case) vers un schéma commun aux 3 scrutins.
MUNI_RENAME = {
    "code_du_departement": "code_dep", "code_departement": "code_dep", "coddpt": "code_dep",
    "libelle_du_departement": "libelle_dep", "libelle_departement": "libelle_dep", "libdpt": "libelle_dep",
    "code_de_la_commune": "code_commune_source", "code_commune": "code_commune_source",
    "codsubcom": "code_commune_source",
    "libelle_de_la_commune": "libelle_commune", "libsubcom": "libelle_commune",
    "popsubcom": "population",
    "nbrins": "inscrits", "nbrabs": "abstentions", "nbrvot": "votants", "nbrblanul": "blancs_et_nuls",
    "nbrexp": "exprimes", "exprimes": "exprimes",
    "n_pan": "num_panneau", "numero_de_panneau": "num_panneau",
    "code_nuance": "nuance", "nuance_liste": "nuance",
    "sexpsn": "sexe", "sexe_candidat": "sexe",
    "nompsnext": "nom", "nom_candidat": "nom", "prepsn": "prenom", "prenom_candidat": "prenom",
    "liste": "liste", "libelle_de_liste": "liste", "libelle_abrege_de_liste": "liste_abrege",
    "sieges_elu": "sieges_elu", "elu": "elu_brut",
    "sieges_au_cm": "sieges_cm", "sieges_secteur": "sieges_secteur", "sieges_cc": "sieges_cc",
    "sieges_au_cc": "sieges_cc",
    "nbrvoix": "voix", "pctvoixins": "pct_voix_ins", "pctvoixexp": "pct_voix_exp",
    "pct_voix_inscrits": "pct_voix_ins", "pct_voix_exprimes": "pct_voix_exp",
}
MUNI_COUNTS = ["inscrits", "abstentions", "votants", "blancs", "nuls", "blancs_et_nuls", "exprimes",
               "voix", "sieges_cm", "sieges_secteur", "sieges_cc", "population"]
MUNI_PCTS = ["pct_voix_ins", "pct_voix_exp"]
MUNI_ORDER = ["annee", "tour", "code_insee", "secteur", "code_dep", "libelle_dep", "code_commune_source",
              "libelle_commune", "mode_scrutin", "inscrits", "abstentions", "votants", "blancs", "nuls",
              "blancs_et_nuls", "exprimes", "num_panneau", "nuance", "sexe", "nom", "prenom", "liste",
              "liste_abrege", "voix", "pct_voix_ins", "pct_voix_exp", "elu", "sieges_cm", "sieges_secteur",
              "sieges_cc", "source"]
ELU_TRUE = {"OUI", "ELU", "O", "ÉLU"}


def _finalize_muni(df: pd.DataFrame, annee: int, tour: int, mode: str, source: str) -> pd.DataFrame:
    df = normalize_columns(df).rename(columns=MUNI_RENAME)
    # Pourcentages par commune (abstention, votants...) : redondants, recalculables.
    df = df.drop(columns=[c for c in df.columns if c.startswith("pct_") and c not in MUNI_PCTS]
                 + [c for c in ("date_de_l_export", "type_de_scrutin") if c in df.columns])
    df = add_code_insee(df, "code_dep", "code_commune_source", with_secteur=True)
    # « Sièges / Elu » : nombre de sièges (scrutin de liste) ou « Oui » (plurinominal).
    if "sieges_elu" in df.columns:
        raw = clean_str(df.pop("sieges_elu"))
        if mode == "liste":
            # Dans les fichiers « ≥ 1000 hab. », les sections électorales de Polynésie
            # sont au scrutin plurinominal : la colonne vaut « Oui » au lieu d'un nombre.
            pluri = raw.str.upper().isin(YES_NO).fillna(False).astype(bool)
            df["sieges_cm"] = raw.mask(pluri)
            df["elu_brut"] = raw.where(pluri)
            df["mode_scrutin"] = np.where(pluri, "plurinominal", "liste")
        else:
            df["elu_brut"] = raw
    for c in MUNI_COUNTS:
        if c in df.columns:
            df[c] = to_int(df[c])
    for c in MUNI_PCTS:
        if c in df.columns:
            df[c] = to_num(df[c])
    elu = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if "elu_brut" in df.columns:
        elu = clean_str(df.pop("elu_brut")).str.upper().isin(ELU_TRUE).astype("boolean")
    if "sieges_cm" in df.columns:
        has_sieges = df["sieges_cm"].notna()
        elu = elu.mask(has_sieges, df["sieges_cm"] > 0).astype("boolean")
    df["elu"] = elu
    if "mode_scrutin" not in df.columns:
        df["mode_scrutin"] = mode
    df = df.assign(annee=annee, tour=tour, source=source)
    for c in ("code_dep", "libelle_dep", "code_commune_source", "libelle_commune", "num_panneau", "nuance",
              "sexe", "nom", "prenom", "liste", "liste_abrege"):
        if c in df.columns:
            df[c] = clean_str(df[c])
    n_missing = df["code_insee"].isna().sum()
    if n_missing:
        log.warning("%s : %d ligne(s) sans code INSEE (Français de l'étranger ou code inconnu)", source, n_missing)
    return df[[c for c in MUNI_ORDER if c in df.columns]]


def _muni_wide(source: str, annee: int, tour: int, mode: str, first_block_col: str,
               block_size: int, label_col: str | None = None) -> pd.DataFrame:
    header, rows = iter_rows(source)
    n_prefix = [h.strip() for h in header].index(first_block_col)
    # Réparation des libellés seulement pour le scrutin de liste (colonne suivante = sièges).
    df = wide_to_long(header, rows, n_prefix, block_size, label_col if mode == "liste" else None)
    return _finalize_muni(df, annee, tour, mode, source)


def load_municipales_2014() -> pd.DataFrame:
    """Municipales 2014, une ligne par commune (x secteur) x liste ou candidat, T1 et T2.

    - ≥ 1000 hab. (T1, T2) et < 1000 hab. T2 : format large, bloc de 11 colonnes
      (« Code Nuance » ... « % Voix/Exp ») répété par liste / candidat.
    - < 1000 hab. T1 : déjà au format long (une ligne par candidat), et ne
      contient que les candidats élus au 1er tour.
    """
    parts = [
        _muni_wide("muni2014_t1_1000plus", 2014, 1, "liste", "Code Nuance", 11, "Liste"),
        _muni_wide("muni2014_t2_1000plus", 2014, 2, "liste", "Code Nuance", 11, "Liste"),
        _muni_wide("muni2014_t2_moins1000", 2014, 2, "plurinominal", "Code Nuance", 11),
        _finalize_muni(read_csv_raw("muni2014_t1_moins1000"), 2014, 1, "plurinominal", "muni2014_t1_moins1000"),
    ]
    return pd.concat(parts, ignore_index=True)


def load_municipales_2020() -> pd.DataFrame:
    """Municipales 2020, même logique : bloc de 12 colonnes (« N.Pan. » ... « % Voix/Exp »)."""
    parts = [
        _muni_wide("muni2020_t1_1000plus", 2020, 1, "liste", "N.Pan.", 12, "Liste"),
        _muni_wide("muni2020_t2_1000plus", 2020, 2, "liste", "N.Pan.", 12, "Liste"),
        _muni_wide("muni2020_t1_moins1000", 2020, 1, "plurinominal", "N.Pan.", 12),
        _muni_wide("muni2020_t2_moins1000", 2020, 2, "plurinominal", "N.Pan.", 12),
    ]
    return pd.concat(parts, ignore_index=True)


def load_municipales_2026() -> pd.DataFrame:
    """Municipales 2026 : scrutin de liste dans toutes les communes (loi de 2025).

    Colonnes numérotées (« Voix 1 », « Voix 2 »...) : bloc de 13 colonnes
    (« Numéro de panneau i » ... « Sièges au CC i »).
    """
    parts = [
        _muni_wide("muni2026_t1_communes", 2026, 1, "liste", "Numéro de panneau 1", 13, "Libellé de liste"),
        _muni_wide("muni2026_t2_communes", 2026, 2, "liste", "Numéro de panneau 1", 13, "Libellé de liste"),
    ]
    return pd.concat(parts, ignore_index=True)


def communes_from_muni(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (annee, tour, code_insee, secteur) : variables de participation."""
    keys = ["annee", "tour", "code_insee", "secteur"]
    cols = [c for c in ("code_dep", "libelle_commune", "mode_scrutin", "inscrits", "abstentions", "votants",
                        "blancs", "nuls", "blancs_et_nuls", "exprimes") if c in df.columns]
    g = df.groupby(keys, dropna=False, sort=True)
    out = g[cols].first()
    out["n_listes"] = g.size()
    return out.reset_index()


# --------------------------------------------------------------------------
# Élus et candidats
# --------------------------------------------------------------------------
DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y")  # %y : maires sortants 2026 (« 04/03/51 »)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes « date* » : essaie les formats connus, garde le premier qui parse tout."""
    for c in df.columns:
        if not c.startswith("date"):
            continue
        filled = df[c].notna() & (df[c].astype("string").str.strip() != "")
        best = None
        for fmt in DATE_FORMATS:
            parsed = pd.to_datetime(df[c], format=fmt, errors="coerce")
            if fmt.endswith("%y"):
                # Python lit « 51 » comme 2051 : une date future recule d'un siècle.
                future = parsed > pd.Timestamp.today()
                parsed = parsed.mask(future, parsed - pd.DateOffset(years=100))
            if best is None or parsed[filled].notna().sum() > best[filled].notna().sum():
                best = parsed
        n_bad = int((best.isna() & filled).sum())
        if n_bad:
            log.warning("%s : %d date(s) non reconnue(s)", c, n_bad)
        df[c] = best
    return df


def _load_rne_like(source: str) -> pd.DataFrame:
    df = normalize_columns(read_csv_raw(source))
    df = add_code_insee(df, "code_du_departement", "code_de_la_commune")
    df = df.rename(columns={"nom_de_l_elu": "nom", "prenom_de_l_elu": "prenom", "code_sexe": "sexe"})
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = clean_str(df[c])
    return _parse_dates(df)


def load_rne_maires() -> pd.DataFrame:
    """Répertoire national des élus : maires en fonction (instantané à la date de MAJ)."""
    return _load_rne_like("rne_maires")


def load_maires_sortants_2026() -> pd.DataFrame:
    """Maires en fonction au 27/02/2026, juste avant le scrutin."""
    return _load_rne_like("muni2026_maires_sortants")


def load_candidatures_2020() -> pd.DataFrame:
    """Candidatures T1 2020, communes ≥ 1000 hab. (listes). 2 lignes de préambule avant l'en-tête."""
    path, enc, sep = raw_file("muni2020_t1_listes_candidats")
    with path.open(encoding=enc) as fh:
        skip = next(i for i, line in enumerate(fh) if line.startswith("Code du d"))
    df = normalize_columns(read_csv_raw("muni2020_t1_listes_candidats", skiprows=skip))
    df = df.loc[:, [c for c in df.columns if not c.startswith("unnamed")]]
    df = add_code_insee(df, "code_du_departement", "code_commune", with_secteur=True)
    df["tete_de_liste"] = to_int(df["n_candidat"]).eq(1).astype("boolean")
    keep = {"tete_de_liste", "code_insee", "secteur"}
    return df.apply(lambda s: s if s.name in keep else clean_str(s))


def load_candidats_plurinominal_2020() -> pd.DataFrame:
    """Candidatures T1 2020, communes < 1000 hab. (scrutin plurinominal)."""
    df = normalize_columns(read_csv_raw("muni2020_t1_candidats_plurinominal"))
    df = df.loc[:, [c for c in df.columns if not c.startswith("unnamed")]]
    return add_code_insee(df, "code_du_departement", "code_commune", with_secteur=True)


def load_elus_2020() -> pd.DataFrame:
    """Conseillers municipaux élus en 2020 (T1 + T2)."""
    df = normalize_columns(read_csv_raw("muni2020_elus_t1t2"))
    df = add_code_insee(df, "code_departement", "code_commune", with_secteur=True)
    return _parse_dates(df)


def load_candidatures_2026() -> pd.DataFrame:
    """Candidatures 2026 T1 et T2 ; « Code circonscription » = code commune INSEE."""
    parts = []
    for tour in (1, 2):
        df = normalize_columns(read_csv_raw(f"muni2026_t{tour}_candidatures"))
        df = add_code_insee(df, "code_departement", "code_circonscription")
        df["tete_de_liste"] = df["tete_de_liste"].str.upper().eq("OUI").astype("boolean")
        df.insert(1, "tour", tour)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_candidats_elus_2026() -> pd.DataFrame:
    parts = []
    for tour in (1, 2):
        df = normalize_columns(read_csv_raw(f"muni2026_t{tour}_candidats_elus"))
        df = add_code_insee(df, "coddpt", "codcom")
        df.insert(1, "tour", tour)
        parts.append(df.rename(columns={"datnaipsn": "date_naissance"}))
    out = pd.concat(parts, ignore_index=True)
    out["date_naissance"] = pd.to_datetime(out["date_naissance"], format="%Y-%m-%d", errors="coerce")
    return out


# --------------------------------------------------------------------------
# COG et contours
# --------------------------------------------------------------------------
def load_cog_communes() -> pd.DataFrame:
    """COG 2026 : communes, arrondissements municipaux, communes déléguées / associées."""
    df = normalize_columns(read_csv_raw("cog2026_communes"))
    return df.rename(columns={"com": "code_insee"}).apply(clean_str)


def load_cog_mouvements() -> pd.DataFrame:
    """Événements sur les communes depuis 1943 (fusions...). Table de passage NON reconstruite."""
    df = normalize_columns(read_csv_raw("cog2026_mouvements_communes")).apply(clean_str)
    df["date_eff"] = pd.to_datetime(df["date_eff"], errors="coerce")
    return df


def load_cog_historique() -> pd.DataFrame:
    df = normalize_columns(read_csv_raw("cog2026_communes_depuis_1943")).apply(clean_str)
    for c in ("date_debut", "date_fin"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.rename(columns={"com": "code_insee"})


def load_contours():
    """Contours des communes 2025 (data.gouv.fr, simplification 50 m) en GeoDataFrame."""
    import geopandas as gpd

    path = ROOT / manifest_entry("contours_communes_2025_50m")["path"]
    with gzip.open(path, "rb") as fh:
        gdf = gpd.read_file(fh)
    gdf = normalize_columns(gdf).rename(columns={"code": "code_insee"})
    gdf["code_insee"] = gdf["code_insee"].astype("string")
    return gdf


# --------------------------------------------------------------------------
# INSEE (fichiers Melodi, format long : une ligne par GEO x dimensions x mesure)
# --------------------------------------------------------------------------
def _melodi_communes(source: str, member: str) -> pd.DataFrame:
    """Filtre GEO_OBJECT == "COM" AVANT tout traitement, puis nettoie.

    Lecture en flux avec pyarrow (le fichier population fait ~800 Mo) : seules
    les lignes communales sont conservées en mémoire. Les dimensions (sexe,
    âge, PCS...) sont stockées en catégories.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    path, enc, sep = raw_file(source, member)
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(encoding=enc, block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter=sep),
        convert_options=pacsv.ConvertOptions(column_types={"GEO": pa.string(), "OBS_VALUE": pa.float64()}),
    )
    batches = [b.filter(pc.equal(b.column("GEO_OBJECT"), "COM")) for b in reader]
    df = pa.Table.from_batches(batches).to_pandas()
    df = normalize_columns(df).rename(columns={"geo": "code_insee"}).drop(columns=["geo_object"])
    df["code_insee"] = df["code_insee"].astype("string").str.zfill(5)
    df["time_period"] = df["time_period"].astype("Int64")
    for c in df.columns.difference(["code_insee", "time_period", "obs_value"]):
        df[c] = df[c].astype("string").astype("category")
    return df


def load_filosofi() -> pd.DataFrame:
    """Filosofi 2023 : une ligne par commune, une colonne par indicateur (médiane, déciles...).

    Les valeurs absentes correspondent au secret statistique (petites communes).
    Au niveau communal, seuls la médiane du niveau de vie (med_sl) et le taux
    de pauvreté (pr_md60) sont diffusés : les indicateurs entièrement vides
    sont retirés, mais toutes les communes sont conservées.
    """
    df = _melodi_communes("insee_filosofi_2023", "DS_FILOSOFI_CC_2023_data.csv")
    wide = (df.set_index(["code_insee", "time_period", "filosofi_measure"])["obs_value"]
            .unstack("filosofi_measure"))
    empty = wide.columns[wide.isna().all()]
    log.info("filosofi : %d indicateur(s) entièrement soumis au secret retiré(s)", len(empty))
    wide = wide.drop(columns=empty)
    wide.columns = [snake_case(str(c)) for c in wide.columns]
    return wide.reset_index().rename(columns={"time_period": "annee"})


def load_diplomes() -> pd.DataFrame:
    """Diplômes (RP 2012, 2017, 2023), population de 15 ans ou plus, format long."""
    df = _melodi_communes("insee_diplomes_rp2023", "DS_RP_DIPLOMES_PRINC_2023_data.csv")
    return df.rename(columns={"time_period": "annee", "obs_value": "valeur"})


def load_population() -> pd.DataFrame:
    """Évolution et structure de la population (RP), format long."""
    df = _melodi_communes("insee_population_rp2023", "DS_RP_POPULATION_COMP_2023_data.csv")
    return df.rename(columns={"time_period": "annee", "obs_value": "valeur"})


# --------------------------------------------------------------------------
# Présidentielles (une ligne par commune x candidat)
# --------------------------------------------------------------------------
PRES_RENAME = {"n_panneau": "num_panneau", "code_du_departement": "code_dep",
               "libelle_du_departement": "libelle_dep", "code_de_la_commune": "code_commune_source",
               "libelle_de_la_commune": "libelle_commune"}


def _finalize_pres(df: pd.DataFrame, annee: int, tour: int) -> pd.DataFrame:
    df = normalize_columns(df).rename(columns=PRES_RENAME)
    df = df.drop(columns=[c for c in df.columns if c.startswith("pct_") and c not in MUNI_PCTS])
    df = add_code_insee(df, "code_dep", "code_commune_source")
    for c in ("inscrits", "abstentions", "votants", "blancs", "nuls", "exprimes", "voix", "num_panneau"):
        df[c] = to_int(df[c])
    for c in MUNI_PCTS:
        df[c] = to_num(df[c])
    for c in ("code_dep", "code_commune_source"):
        df[c] = df[c].map(_code_part).astype("string")
    df = df[df["code_insee"].notna()]  # retire les Français de l'étranger (ZZ)
    return df.assign(annee=annee, tour=tour)


def load_pres2017() -> pd.DataFrame:
    """Présidentielle 2017 (xls lu avec xlrd). En-tête en ligne 4, blocs de 7 colonnes par candidat."""
    import xlrd

    parts = []
    for tour in (1, 2):
        path = ROOT / manifest_entry(f"pres2017_t{tour}_communes")["path"]
        sheet = xlrd.open_workbook(path, on_demand=True).sheet_by_index(0)
        header_row = next(i for i in range(10) if str(sheet.cell_value(i, 0)).startswith("Code du d"))
        header = [str(v) for v in sheet.row_values(header_row)]
        # xlrd renvoie des float (« 1.0 ») et parfois des nombres pour des libellés : tout en str.
        rows = ([_code_part(v) for v in sheet.row_values(i)] for i in range(header_row + 1, sheet.nrows))
        n_prefix = header.index("N°Panneau")
        parts.append(_finalize_pres(wide_to_long(header, rows, n_prefix, 7), 2017, tour))
    return pd.concat(parts, ignore_index=True)


def load_pres2022() -> pd.DataFrame:
    """Présidentielle 2022 (txt), colonne « Etat saisie » en plus, blocs de 7 colonnes."""
    parts = []
    for tour in (1, 2):
        header, rows = iter_rows(f"pres2022_t{tour}_communes")
        n_prefix = [h.strip() for h in header].index("N°Panneau")
        parts.append(_finalize_pres(wide_to_long(header, rows, n_prefix, 7), 2022, tour))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Écriture
# --------------------------------------------------------------------------
LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "municipales_2014": load_municipales_2014,
    "municipales_2020": load_municipales_2020,
    "municipales_2026": load_municipales_2026,
    "rne_maires": load_rne_maires,
    "maires_sortants_2026": load_maires_sortants_2026,
    "candidatures_2020": load_candidatures_2020,
    "candidats_plurinominal_2020": load_candidats_plurinominal_2020,
    "elus_2020": load_elus_2020,
    "candidatures_2026": load_candidatures_2026,
    "candidats_elus_2026": load_candidats_elus_2026,
    "cog_communes": load_cog_communes,
    "cog_mouvements": load_cog_mouvements,
    "cog_historique": load_cog_historique,
    "contours_communes": load_contours,
    "filosofi": load_filosofi,
    "diplomes": load_diplomes,
    "population": load_population,
    "presidentielle_2017": load_pres2017,
    "presidentielle_2022": load_pres2022,
}


def write_processed(name: str) -> Path:
    df = LOADERS[name]()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    log.info("%-28s %9d lignes x %3d colonnes -> %s", name, len(df), df.shape[1], out.relative_to(ROOT))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Nettoie les sources brutes vers data/processed/*.parquet")
    p.add_argument("--only", nargs="+", choices=sorted(LOADERS), metavar="NOM")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    failures = []
    for name in args.only or LOADERS:
        try:
            write_processed(name)
        except Exception as exc:  # noqa: BLE001 : on veut le diagnostic de toutes les sources
            log.exception("%s : ÉCHEC : %s", name, exc)
            failures.append(name)
    if failures:
        log.error("Échec : %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

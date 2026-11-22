"""Caractéristiques des communes, pour comparer des sortants « toutes choses égales ».

Pour chaque élection municipale, on prend la source la plus proche qui lui est
antérieure ou contemporaine :

| élection | recensement (RP) | présidentielle | revenus (Filosofi) |
|----------|------------------|----------------|--------------------|
| 2020     | 2017             | 2017           | 2023 (seul millésime disponible) |
| 2026     | 2023             | 2022           | 2023               |

Variables (en % de la population de 15 ans ou plus, sauf mention) :
diplômés du supérieur, sans diplôme, cadres, ouvriers, retraités, 55 ans ou plus ;
revenu médian (euros par unité de consommation), taux de pauvreté ; vote
d'extrême droite et vote Macron au 1er tour de la présidentielle (% des exprimés),
participation à ce 1er tour.

Usage :
    python src/covariables.py   # écrit data/processed/covariables_communes.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P  # noqa: E402

log = logging.getLogger("covariables")

RP_PAR_ELECTION = {2020: 2017, 2026: 2023}
PRES_PAR_ELECTION = {2020: 2017, 2026: 2022}
EXTREME_DROITE = {2017: {"LE PEN"}, 2022: {"LE PEN", "ZEMMOUR"}}

# Codes du recensement (Melodi). Diplômes : population non scolarisée de 15 ans ou plus.
DIPLOME_SUP = "500T702_RP"      # bac+2 et plus
SANS_DIPLOME = "001T100_RP"     # aucun diplôme ou certificat d'études
PCS = {"3": "cadres", "6": "ouvriers", "7": "retraites"}


def _part(df: pd.DataFrame, dim: str, codes: dict[str, str], total: str = "_T") -> pd.DataFrame:
    """Part (en %) de chaque modalité de ``dim`` dans le total, par commune et année."""
    w = df.pivot_table(index=["code_insee", "annee"], columns=dim, values="valeur",
                       aggfunc="sum", observed=True)
    out = pd.DataFrame({nom: 100 * w[code] / w[total] for code, nom in codes.items()})
    return out.reset_index()


def recensement() -> pd.DataFrame:
    dip = P.read_processed("diplomes")
    dip = dip[dip["sex"] == "_T"]
    dip = _part(dip, "educ", {DIPLOME_SUP: "pct_diplome_sup", SANS_DIPLOME: "pct_sans_diplome"})

    pop = P.read_processed("population")
    pop = pop[pop["sex"] == "_T"]
    pcs = _part(pop[pop["age"] == "Y_GE15"], "pcs", {k: f"pct_{v}" for k, v in PCS.items()})
    age = pop[pop["pcs"] == "_T"]
    age = _part(age, "age", {"Y_GE55": "pct_55_plus"}, total="Y_GE15")
    age = age.merge(pop[(pop["pcs"] == "_T") & (pop["age"] == "Y_GE15")]
                    .groupby(["code_insee", "annee"], observed=True)["valeur"].sum()
                    .rename("population_15_plus").reset_index(), on=["code_insee", "annee"])
    rp = dip.merge(pcs, on=["code_insee", "annee"], how="outer").merge(age, on=["code_insee", "annee"], how="outer")
    return rp.rename(columns={"annee": "annee_rp"})


def presidentielle(annee: int) -> pd.DataFrame:
    df = P.read_processed(f"presidentielle_{annee}")
    t1 = df[df["tour"] == 1]
    commune = t1.groupby("code_insee").agg(inscrits=("inscrits", "first"), votants=("votants", "first"),
                                           exprimes=("exprimes", "first"))
    voix = t1.assign(ed=t1["nom"].isin(EXTREME_DROITE[annee]), macron=t1["nom"] == "MACRON")
    out = pd.DataFrame({
        "pct_extreme_droite_pres": 100 * voix[voix.ed].groupby("code_insee")["voix"].sum() / commune["exprimes"],
        "pct_macron_pres": 100 * voix[voix.macron].groupby("code_insee")["voix"].sum() / commune["exprimes"],
        "participation_pres": 100 * commune["votants"] / commune["inscrits"],
    })
    return out.reset_index().assign(annee_pres=annee)


def build() -> pd.DataFrame:
    rp = recensement()
    filo = P.read_processed("filosofi").drop(columns="annee").rename(
        columns={"med_sl": "revenu_median", "pr_md60": "taux_pauvrete"})
    parts = []
    for election in (2020, 2026):
        c = rp[rp["annee_rp"] == RP_PAR_ELECTION[election]]
        c = c.merge(presidentielle(PRES_PAR_ELECTION[election]), on="code_insee", how="outer")
        c = c.merge(filo, on="code_insee", how="left")
        parts.append(c.assign(annee=election))
    return pd.concat(parts, ignore_index=True)


COV_PATH = P.PROCESSED_DIR / "covariables_communes.parquet"


def load_covariables() -> pd.DataFrame:
    """Covariables enregistrées, construites (et enregistrées) si besoin."""
    if not COV_PATH.exists():
        build().to_parquet(COV_PATH, index=False)
    return pd.read_parquet(COV_PATH)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    cov = build()
    cov.to_parquet(COV_PATH, index=False)
    log.info("covariables : %d lignes x %d colonnes", *cov.shape)
    log.info("valeurs manquantes (%%) :\n%s", (100 * cov.isna().mean()).round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Téléchargement reproductible des données brutes du projet.

Question de recherche : les électeurs sanctionnent-ils les maires visés par une
procédure pour atteinte à la probité ?

Les fichiers sont écrits dans data/raw/ (jamais versionné) ; un petit manifeste
data/MANIFEST.json (versionné) garde la trace de ce qui a été téléchargé.

Usage :
    python scripts/download_data.py              # tout télécharger
    python scripts/download_data.py --list       # lister les sources
    python scripts/download_data.py --only rne_maires
    python scripts/download_data.py --force      # retélécharger même si présent

Toutes les URL ont été retrouvées via l'API data.gouv.fr
(https://www.data.gouv.fr/api/1/datasets/...) ou, pour Filosofi, via le
catalogue officiel INSEE Melodi (https://api.insee.fr/melodi/catalog/all).
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "MANIFEST.json"

USER_AGENT = "Projet-ENSAE-probite-maires/1.0 (projet etudiant ENSAE; python-requests)"
TIMEOUT = (15, 120)  # (connexion, lecture) en secondes
MAX_RETRIES = 4
PAUSE_BETWEEN_FILES = 1.5  # secondes, pour ménager les serveurs
CHUNK_SIZE = 1 << 20  # 1 Mo
MIN_FREE_BYTES = 3 * 1024**3  # avertissement si moins de 3 Go libres

TEXT_FORMATS = {"csv", "txt"}
CANDIDATE_ENCODINGS = ("utf-8", "cp1252", "latin-1")
CANDIDATE_SEPARATORS = ";,\t|"

MI = "Ministère de l'Intérieur"
DGF = "https://static.data.gouv.fr/resources"

# --------------------------------------------------------------------------
# Sources. format : csv | txt | xls | zip | geojson.gz
# --------------------------------------------------------------------------
SOURCES: dict[str, dict[str, str]] = {
    # 1. Résultats des municipales par commune --------------------------------
    # Producteur : Ministère de l'Intérieur, data.gouv.fr. Licence : Licence Ouverte.
    "muni2014_t1_moins1000": {
        "url": "https://www.data.gouv.fr/storage/f/2014-03-25T16-08-50/muni-2014-resultats-com-moins-1000-t1.txt",
        "filename": "muni2014_t1_moins1000.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2014_t1_1000plus": {
        "url": "https://www.data.gouv.fr/storage/f/2014-03-25T16-06-23/muni-2014-resultats-com-1000-et-plus-t1.txt",
        "filename": "muni2014_t1_1000plus.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2014_t2_moins1000": {
        "url": "https://www.data.gouv.fr/storage/f/2014-03-31T09-51-08/muni-2014-resultats-com-moins-1000-t2.txt",
        "filename": "muni2014_t2_moins1000.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2014_t2_1000plus": {
        "url": "https://www.data.gouv.fr/storage/f/2014-03-31T09-49-28/muni-2014-resultats-com-1000-et-plus-t2.txt",
        "filename": "muni2014_t2_1000plus.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2020_t1_moins1000": {
        "url": f"{DGF}/elections-municipales-2020-resultats/20200525-133805/2020-05-18-resultats-communes-de-moins-de-1000.txt",
        "filename": "muni2020_t1_moins1000.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2020_t1_1000plus": {
        "url": f"{DGF}/elections-municipales-2020-resultats/20200525-133704/2020-05-18-resultats-communes-de-1000-et-plus.txt",
        "filename": "muni2020_t1_1000plus.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2020_t2_moins1000": {
        "url": f"{DGF}/municipales-2020-resultats-2nd-tour/20200629-192436/2020-06-29-resultats-t2-communes-de-moins-de-1000-hab.txt",
        "filename": "muni2020_t2_moins1000.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2020_t2_1000plus": {
        "url": f"{DGF}/municipales-2020-resultats-2nd-tour/20200629-192435/2020-06-29-resultats-t2-communes-de-1000-hab-et-plus.txt",
        "filename": "muni2020_t2_1000plus.txt",
        "format": "txt",
        "producer": MI,
    },
    # Licence Ouverte v2.0
    "muni2026_t1_communes": {
        "url": f"{DGF}/elections-municipales-2026-resultats-du-premier-tour/20260320-164339/municipales-2026-resultats-communes-2026-03-20.csv",
        "filename": "muni2026_t1_communes.csv",
        "format": "csv",
        "producer": MI,
    },
    # NB : la faute « scond » figure dans l'URL officielle, ne pas la corriger.
    "muni2026_t2_communes": {
        "url": f"{DGF}/elections-municipales-2026-resultats-du-scond-tour/20260323-180124/municipales-2026-resultats-communes-2026-03-23-16h14.csv",
        "filename": "muni2026_t2_communes.csv",
        "format": "csv",
        "producer": MI,
    },
    # 2. Maires ----------------------------------------------------------------
    # Répertoire national des élus. Licence Ouverte v2.0. Instantané à la date de MAJ.
    "rne_maires": {
        "url": f"{DGF}/repertoire-national-des-elus-1/20260811-155100/elus-maire-mai.csv",
        "filename": "rne_maires.csv",
        "format": "csv",
        "producer": MI,
    },
    # Maires sortants avant le scrutin de 2026. Licence Ouverte v2.0.
    "muni2026_maires_sortants": {
        "url": f"{DGF}/elections-municipales-2026-maires-et-conseillers-municipaux-sortants/20260302-101708/mun2026-maires-sortants-20260227.csv",
        "filename": "muni2026_maires_sortants.csv",
        "format": "csv",
        "producer": MI,
    },
    # 3. Candidats et têtes de liste -------------------------------------------
    # TODO : candidatures 2014. Pas de fichier France entière sur data.gouv.fr,
    #   seulement environ 24 .xls régionaux par jeu (ids 53699396a3a729239d20422d
    #   et 53699396a3a729239d20422c). Les têtes de liste 2014 sont reprises
    #   depuis les fichiers de résultats muni2014_*.
    "muni2020_t1_listes_candidats": {
        "url": f"{DGF}/elections-municipales-2020-candidatures-au-1er-tour/20200304-105123/livre-des-listes-et-candidats.txt",
        "filename": "muni2020_t1_listes_candidats.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2020_t1_candidats_plurinominal": {
        "url": f"{DGF}/elections-municipales-2020-candidatures-au-1er-tour/20200304-105140/livre-des-candidats-scrutin-plurinominal.txt",
        "filename": "muni2020_t1_candidats_plurinominal.txt",
        "format": "txt",
        "producer": MI,
    },
    "muni2020_elus_t1t2": {
        "url": f"{DGF}/municipales-2020-resultats-2nd-tour/20200630-191733/mn20-elus-t1t2-vf.txt",
        "filename": "muni2020_elus_t1t2.txt",
        "format": "txt",
        "producer": MI,
    },
    # Licence Ouverte v2.0. Fichier volumineux (~145 Mo).
    "muni2026_t1_candidatures": {
        "url": f"{DGF}/elections-municipales-2026-listes-candidates-au-premier-tour/20260313-152615/municipales-2026-candidatures-france-entiere-tour-1-2026-03-13.csv",
        "filename": "muni2026_t1_candidatures.csv",
        "format": "csv",
        "producer": MI,
    },
    "muni2026_t2_candidatures": {
        "url": f"{DGF}/elections-municipales-2026-listes-candidates-au-second-tour/20260320-141955/municipales-2026-candidatures-france-entiere-tour-2-2026-03-20.csv",
        "filename": "muni2026_t2_candidatures.csv",
        "format": "csv",
        "producer": MI,
    },
    "muni2026_t1_candidats_elus": {
        "url": f"{DGF}/elections-municipales-2026-resultats-du-premier-tour/20260320-164100/municipales-2026-candidats-elus-france-entiere-tour-1-2026-03-20.csv",
        "filename": "muni2026_t1_candidats_elus.csv",
        "format": "csv",
        "producer": MI,
    },
    "muni2026_t2_candidats_elus": {
        "url": f"{DGF}/elections-municipales-2026-resultats-du-scond-tour/20260323-180122/municipales-2026-candidats-elus-france-entiere-tour-2-2026-03-23.csv",
        "filename": "muni2026_t2_candidats_elus.csv",
        "format": "csv",
        "producer": MI,
    },
    # 4. Code officiel géographique ---------------------------------------------
    # Producteur : INSEE, millésime 2026. Licence Ouverte.
    "cog2026_communes": {
        "url": "https://www.insee.fr/fr/statistiques/fichier/8740222/v_commune_2026.csv",
        "filename": "cog2026_communes.csv",
        "format": "csv",
        "producer": "INSEE",
    },
    # Événements survenus aux communes depuis 1943 (fusions, communes nouvelles).
    # Sert de table de passage ; elle n'est pas reconstruite ici.
    "cog2026_mouvements_communes": {
        "url": "https://www.insee.fr/fr/statistiques/fichier/8740222/v_mvt_commune_2026.csv",
        "filename": "cog2026_mouvements_communes.csv",
        "format": "csv",
        "producer": "INSEE",
    },
    "cog2026_communes_depuis_1943": {
        "url": "https://www.insee.fr/fr/statistiques/fichier/8740222/v_commune_depuis_1943.csv",
        "filename": "cog2026_communes_depuis_1943.csv",
        "format": "csv",
        "producer": "INSEE",
    },
    # 5. Contours des communes ---------------------------------------------------
    # TODO : IGN ADMIN EXPRESS COG. data.gouv.fr (id 5808de39c751df1e0679df72) ne
    #   donne qu'une page web et des flux WMS/WFS, pas de lien de téléchargement direct.
    # Alternative : « Contours administratifs », data.gouv.fr (Etalab), licence ODbL,
    # millésime 2025, simplification 50 m. Conservé compressé (.gz).
    "contours_communes_2025_50m": {
        "url": "https://object.data.gouv.fr/contours-administratifs/2025/geojson/communes-50m.geojson.gz",
        "filename": "contours_communes_2025_50m.geojson.gz",
        "format": "geojson.gz",
        "producer": "data.gouv.fr (Etalab), ODbL",
    },
    # 6. INSEE au niveau communal (archives zip via Melodi). Licence Ouverte v2.0.
    "insee_population_rp2023": {
        "url": "https://api.insee.fr/melodi/file/DS_RP_POPULATION_COMP/DS_RP_POPULATION_COMP_2023_CSV_FR",
        "filename": "insee_population_rp2023.zip",
        "format": "zip",
        "producer": "INSEE",
    },
    "insee_diplomes_rp2023": {
        "url": "https://api.insee.fr/melodi/file/DS_RP_DIPLOMES_PRINC/DS_RP_DIPLOMES_PRINC_2023_CSV_FR",
        "filename": "insee_diplomes_rp2023.zip",
        "format": "zip",
        "producer": "INSEE",
    },
    # URL tirée du catalogue Melodi (api.insee.fr/melodi/catalog/all), pas de data.gouv.fr.
    "insee_filosofi_2023": {
        "url": "https://api.insee.fr/melodi/file/DS_FILOSOFI_CC/DS_FILOSOFI_CC_2023_CSV_FR",
        "filename": "insee_filosofi_2023.zip",
        "format": "zip",
        "producer": "INSEE",
    },
    # 7. Présidentielles par commune (résultats définitifs). Licence Ouverte.
    "pres2017_t1_communes": {
        "url": f"{DGF}/election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-1er-tour-par-communes/20170427-100544/Presidentielle_2017_Resultats_Communes_Tour_1_c.xls",
        "filename": "pres2017_t1_communes.xls",
        "format": "xls",
        "producer": MI,
    },
    "pres2017_t2_communes": {
        "url": f"{DGF}/election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-2nd-tour-par-communes/20170511-093054/Presidentielle_2017_Resultats_Communes_Tour_2_c.xls",
        "filename": "pres2017_t2_communes.xls",
        "format": "xls",
        "producer": MI,
    },
    "pres2022_t1_communes": {
        "url": f"{DGF}/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/20220414-152459/resultats-par-niveau-subcom-t1-france-entiere.txt",
        "filename": "pres2022_t1_communes.txt",
        "format": "txt",
        "producer": MI,
    },
    "pres2022_t2_communes": {
        "url": f"{DGF}/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-2nd-tour/20220428-142333/resultats-par-niveau-subcom-t2-france-entiere.txt",
        "filename": "pres2022_t2_communes.txt",
        "format": "txt",
        "producer": MI,
    },
}

log = logging.getLogger("download_data")


@dataclass
class TextInfo:
    encoding: str
    separator: str | None
    n_lines: int
    n_columns: int | None
    header: list[str]
    error: str | None = None


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_encoding(path: Path) -> str:
    """Essaie utf-8 sur tout le fichier, puis cp1252 ; latin-1 en dernier recours."""
    for enc in CANDIDATE_ENCODINGS[:-1]:
        decoder = codecs.getincrementaldecoder(enc)()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            return "utf-8-sig" if enc == "utf-8" and _has_bom(path) else enc
        except UnicodeDecodeError:
            continue
    return CANDIDATE_ENCODINGS[-1]  # latin-1 décode n'importe quel octet


def _has_bom(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(3) == codecs.BOM_UTF8


def count_lines(path: Path) -> int:
    n, last = 0, b""
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            n += chunk.count(b"\n")
            last = chunk
    if last and not last.endswith(b"\n"):
        n += 1  # dernière ligne sans saut de ligne final
    return n


def inspect_text(path: Path) -> TextInfo:
    """Encodage, séparateur, nb de lignes et en-tête. Ne lève jamais d'exception."""
    try:
        encoding = detect_encoding(path)
        n_lines = count_lines(path)
        with path.open("r", encoding=encoding, newline="") as f:
            sample = f.read(64 * 1024)
        try:
            separator: str | None = csv.Sniffer().sniff(sample, delimiters=CANDIDATE_SEPARATORS).delimiter
        except csv.Error:
            first = sample.splitlines()[0] if sample else ""
            counts = {s: first.count(s) for s in CANDIDATE_SEPARATORS}
            best = max(counts, key=counts.get)
            separator = best if counts[best] > 0 else None
        first_line = sample.splitlines()[0] if sample else ""
        header = next(csv.reader([first_line], delimiter=separator)) if separator else [first_line]
        header = [h.strip() for h in header]
        info = TextInfo(encoding, separator, n_lines, len(header), header[:50])
        if n_lines < 2 or not any(header):
            info.error = "en-tête vide ou moins de 2 lignes"
        return info
    except Exception as exc:  # noqa: BLE001 : on veut un diagnostic, pas un plantage
        return TextInfo("inconnu", None, 0, None, [], error=f"{type(exc).__name__}: {exc}")


def check_disk_space(path: Path) -> None:
    free = shutil.disk_usage(path).free
    log.info("Espace disque libre : %.1f Go", free / 1024**3)
    if free < MIN_FREE_BYTES:
        log.warning("Moins de 3 Go libres : le téléchargement complet (~0,7 Go + zips extraits) risque d'échouer.")


# --------------------------------------------------------------------------
# Téléchargement
# --------------------------------------------------------------------------
def _progress(done: int, total: int | None, name: str) -> None:
    if not sys.stderr.isatty():
        return
    if total:
        pct = done / total
        bar = "#" * int(30 * pct)
        sys.stderr.write(f"\r  {name}: [{bar:<30}] {pct:6.1%} {done / 1e6:7.1f}/{total / 1e6:.1f} Mo")
    else:
        sys.stderr.write(f"\r  {name}: {done / 1e6:7.1f} Mo")
    sys.stderr.flush()


def download(url: str, dest: Path, session: requests.Session, name: str) -> None:
    """Téléchargement en streaming vers un .part, renommé à la fin (atomique)."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0) or None
                done = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        f.write(chunk)
                        done += len(chunk)
                        _progress(done, total, name)
                if sys.stderr.isatty():
                    sys.stderr.write("\n")
                if total is not None and done != total:
                    raise OSError(f"taille reçue {done} != content-length {total}")
            os.replace(tmp, dest)
            return
        except (requests.RequestException, OSError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                raise
            wait = 2**attempt
            log.warning("%s : tentative %d/%d échouée (%s), nouvel essai dans %ds",
                        name, attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)


def extract_zip(archive: Path, target: Path) -> list[Path]:
    """Dézippe dans target/ en refusant les chemins sortant du dossier (zip slip)."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            out = (target / member.filename).resolve()
            if not out.is_relative_to(root):
                raise ValueError(f"chemin suspect dans l'archive : {member.filename}")
        zf.extractall(target)
    return sorted(p for p in target.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# Manifeste
# --------------------------------------------------------------------------
def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("MANIFEST.json illisible, il sera régénéré")
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    tmp = MANIFEST_PATH.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)


def describe_file(path: Path, fmt: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256sum(path),
    }
    if fmt in TEXT_FORMATS or path.suffix.lower() in {".csv", ".txt"}:
        info = inspect_text(path)
        entry.update(encoding=info.encoding, separator=info.separator, n_lines=info.n_lines,
                     n_columns=info.n_columns, header=info.header)
        if info.error:
            entry["check_error"] = info.error
            log.warning("%s : contrôle texte : %s", path.name, info.error)
    return entry


def is_valid(name: str, src: dict[str, str], manifest: dict[str, Any]) -> bool:
    """Présent, non vide, et taille identique au manifeste s'il en existe une entrée."""
    dest = RAW_DIR / src["filename"]
    if not dest.is_file() or dest.stat().st_size == 0:
        return False
    previous = manifest.get(name, {})
    if previous.get("url") not in (None, src["url"]):
        return False  # l'URL a changé depuis le dernier téléchargement
    if "size_bytes" in previous and previous["size_bytes"] != dest.stat().st_size:
        return False
    if src["format"] == "zip" and not (RAW_DIR / name).is_dir():
        return False
    return True


def process(name: str, src: dict[str, str], session: requests.Session,
            manifest: dict[str, Any], force: bool) -> bool:
    dest = RAW_DIR / src["filename"]
    if not force and is_valid(name, src, manifest):
        log.info("%s : déjà présent et valide, ignoré (--force pour retélécharger)", name)
        return False

    log.info("%s : téléchargement de %s", name, src["url"])
    download(src["url"], dest, session, name)
    if dest.stat().st_size == 0:
        raise ValueError(f"{name} : fichier téléchargé vide")

    entry: dict[str, Any] = {
        "source": name,
        "producer": src["producer"],
        "url": src["url"],
        "format": src["format"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **describe_file(dest, src["format"]),
    }
    if src["format"] == "zip":
        files = extract_zip(dest, RAW_DIR / name)
        entry["extracted"] = [describe_file(p, p.suffix.lstrip(".").lower()) for p in files]
        log.info("%s : %d fichier(s) extrait(s) dans %s", name, len(files), (RAW_DIR / name).relative_to(ROOT))

    manifest[name] = entry
    save_manifest(manifest)
    log.info("%s : OK (%.1f Mo, %s lignes, sha256 %s…)", name, entry["size_bytes"] / 1e6,
             entry.get("n_lines", "n/a"), entry["sha256"][:12])
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", metavar="NOM", help="ne traiter que ces sources")
    p.add_argument("--list", action="store_true", help="lister les sources et quitter")
    p.add_argument("--force", action="store_true", help="retélécharger même si déjà présent")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def list_sources(manifest: dict[str, Any]) -> None:
    width = max(map(len, SOURCES))
    for name, src in SOURCES.items():
        status = "présent" if (RAW_DIR / src["filename"]).is_file() else "absent"
        dl = manifest.get(name, {}).get("downloaded_at", "")
        print(f"{name:<{width}}  {src['format']:<10} {status:<8} {dl:<25} {src['url']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    manifest = load_manifest()

    if args.list:
        list_sources(manifest)
        return 0

    names = args.only or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        log.error("Source(s) inconnue(s) : %s (voir --list)", ", ".join(unknown))
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    check_disk_space(RAW_DIR)

    failures: list[str] = []
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for i, name in enumerate(names):
            try:
                downloaded = process(name, SOURCES[name], session, manifest, args.force)
            except Exception as exc:  # noqa: BLE001 : on continue avec les autres sources
                log.error("%s : ÉCHEC : %s", name, exc)
                failures.append(name)
                continue
            if downloaded and i < len(names) - 1:
                time.sleep(PAUSE_BETWEEN_FILES)

    if failures:
        log.error("%d source(s) en échec : %s", len(failures), ", ".join(failures))
        return 1
    log.info("Terminé : %d source(s) traitée(s).", len(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())

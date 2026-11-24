"""Exporte les figures du README depuis les notebooks exécutés.

Chaque figure est la première image produite par la cellule dont le code
contient le motif indiqué. Les notebooks doivent avoir été exécutés au
préalable (voir README).

Usage :
    python scripts/figures.py   # écrit docs/figures/*.png
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SORTIE = ROOT / "docs" / "figures"

FIGURES = {
    "carte.png": ("00_synthese", "G.carte_affaires("),
    "retour_moyenne.png": ("00_synthese", "G.retour_moyenne("),
    "estimations.png": ("00_synthese", "G.estimations("),
    "victoire_par_groupe.png": ("00_synthese", "G.victoire_par_groupe("),
    "stade_affaire.png": ("04_modelisation", "Effet de l'affaire selon son stade"),
}


def image(notebook: str, motif: str) -> bytes:
    nb = json.loads((ROOT / "notebooks" / f"{notebook}.ipynb").read_text(encoding="utf-8"))
    for cellule in nb["cells"]:
        if cellule["cell_type"] == "code" and motif in "".join(cellule["source"]):
            for sortie in cellule.get("outputs", []):
                if "image/png" in sortie.get("data", {}):
                    return base64.b64decode(sortie["data"]["image/png"])
            raise ValueError(f"{notebook} : la cellule « {motif} » n'a pas d'image (notebook non exécuté ?)")
    raise ValueError(f"{notebook} : aucune cellule ne contient « {motif} »")


def main() -> int:
    SORTIE.mkdir(parents=True, exist_ok=True)
    for nom, (notebook, motif) in FIGURES.items():
        (SORTIE / nom).write_bytes(image(notebook, motif))
        print(f"{SORTIE.relative_to(ROOT).as_posix()}/{nom}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

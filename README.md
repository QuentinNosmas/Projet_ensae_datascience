# Les électeurs sanctionnent-ils vraiment ?

Projet de data science (ENSAE 2A, cours « Python pour la data science »).

**Question.** Un maire visé par une affaire judiciaire (enquête, mise en examen,
condamnation) est-il sanctionné par les électeurs aux élections municipales
suivantes ? On compare les maires sortants des municipales de 2020 et de 2026 selon
qu'une affaire était connue ou non au moment du vote.

## Reproduire les résultats

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12 conseillé
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/03_affaires.ipynb
```

Les données ne sont pas versionnées. Au premier lancement, chaque source manquante
est téléchargée depuis son URL publique (environ 400 Mo, quelques minutes), puis
nettoyée dans `data/processed/`. On peut aussi tout préparer à l'avance :

```bash
python scripts/download_data.py   # data/raw/ ; --list pour voir les sources
python src/load.py                # data/processed/*.parquet
python src/panel.py               # panel maire sortant x élection
python src/affaires.py            # affaires rattachées au panel
```

`data/MANIFEST.json` (versionné) garde l'URL, la date de téléchargement et
l'empreinte SHA-256 de chaque fichier utilisé.

## Organisation

| Fichier | Rôle |
|---|---|
| `scripts/download_data.py` | téléchargement reproductible des sources (reprise, contrôles) |
| `src/load.py` | lecture et nettoyage de chaque source, codes INSEE harmonisés |
| `src/panel.py` | panel « maire sortant x élection » : candidature, score, victoire |
| `src/affaires.py` | rattachement des affaires judiciaires aux sortants, variables de traitement |
| `src/analyse.py` | comparaisons à taille de commune égale, intervalles bootstrap |
| `notebooks/01_exploration.ipynb` | diagnostic des sources |
| `notebooks/02_panel_sortants.ipynb` | comportement d'un sortant « ordinaire » |
| `notebooks/03_affaires.ipynb` | effet d'une affaire judiciaire |

## Sources

| Source | Producteur | Licence |
|---|---|---|
| Résultats et candidatures des municipales 2014, 2020, 2026 | Ministère de l'Intérieur (data.gouv.fr) | Licence Ouverte |
| Maires sortants 2026, Répertoire national des élus | Ministère de l'Intérieur (data.gouv.fr) | Licence Ouverte |
| Code officiel géographique 2026 | INSEE | Licence Ouverte |
| Affaires judiciaires : condamnations | [Casier Politique](https://casier-politique.fr/about) | CC BY-NC-SA 4.0 |
| Affaires judiciaires : tous stades | [Poligraph](https://poligraph.fr) | données publiques, code AGPL-3.0 |

Il n'existe pas de base officielle nominative des procédures visant des élus. Les
deux bases d'affaires sont des initiatives citoyennes, mises à jour en continu, qui
ne couvrent que les élus ayant une page Wikipédia ou une fiche Poligraph : surtout
des maires de grandes villes et des élus nationaux. Les limites qui en découlent
sont discutées à la fin du notebook 03.

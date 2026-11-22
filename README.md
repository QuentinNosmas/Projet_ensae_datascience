# Les électeurs sanctionnent-ils vraiment ?

Projet de data science (ENSAE 2A, cours « Python pour la data science »).

**Question.** Un maire visé par une affaire judiciaire (enquête, mise en examen,
condamnation) est-il sanctionné par les électeurs aux élections municipales
suivantes ? On compare les maires sortants des municipales de 2020 et de 2026 selon
qu'une affaire était connue ou non au moment du vote.

**Premiers résultats.** Aucune sanction électorale robuste n'est détectable : les sortants visés
par une affaire conservent leur score et gagnent aussi souvent que des sortants
comparables. Les comparaisons naïves sont trompées par le retour vers la moyenne et
par la prime électorale des élus notoires. À notoriété égale, une affaire révélée
pendant le mandat réduirait la probabilité de victoire d'environ 30 points, un
résultat à la limite de la significativité (une trentaine de cas).

**Projet en cours.** Synthèse d'étape : [`notebooks/00_synthese.ipynb`](notebooks/00_synthese.ipynb).

## Reproduire les résultats

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12 conseillé
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/00_synthese.ipynb
```

Les données ne sont pas versionnées. Au premier lancement, chaque source manquante
est téléchargée depuis son URL publique (environ 400 Mo, quelques minutes), puis
nettoyée dans `data/processed/`. On peut aussi tout préparer à l'avance :

```bash
python scripts/download_data.py   # data/raw/ ; --list pour voir les sources
python src/load.py                # data/processed/*.parquet
python src/panel.py               # panel maire sortant x élection
python src/affaires.py            # affaires rattachées au panel
python src/covariables.py         # contexte socio-économique et politique des communes
```

`data/MANIFEST.json` (versionné) garde l'URL, la date de téléchargement et
l'empreinte SHA-256 de chaque fichier utilisé. Les deux bases d'affaires étant mises
à jour en continu, on en versionne un instantané daté (`data/snapshots/`, 1,8 Mo) ;
`python scripts/download_data.py --live` récupère la version du jour.

## Organisation

| Fichier | Rôle |
|---|---|
| `scripts/download_data.py` | téléchargement reproductible des sources (reprise, contrôles) |
| `src/load.py` | lecture et nettoyage de chaque source, codes INSEE harmonisés |
| `src/panel.py` | panel « maire sortant x élection » : candidature, score, victoire |
| `src/affaires.py` | rattachement des affaires judiciaires aux sortants, variables de traitement |
| `src/covariables.py` | recensement, revenus, présidentielles : contexte de chaque commune |
| `src/analyse.py` | comparaisons à taille de commune égale, intervalles bootstrap |
| `src/modeles.py` | régressions (MCO, logit), appariement sur score de propension |
| `src/graphiques.py` | charte graphique, carte, graphique des estimations |
| `notebooks/00_synthese.ipynb` | synthèse d'étape : la démarche et les premiers résultats |
| `notebooks/01_exploration.ipynb` | diagnostic des sources |
| `notebooks/02_panel_sortants.ipynb` | comportement d'un sortant « ordinaire » |
| `notebooks/03_affaires.ipynb` | affaires judiciaires : rattachement, description, carte, comparaisons à taille égale |
| `notebooks/04_modelisation.ipynb` | régressions, appariement, délai, participation, robustesse |

## Sources

| Source | Producteur | Licence |
|---|---|---|
| Résultats et candidatures des municipales 2014, 2020, 2026 | Ministère de l'Intérieur (data.gouv.fr) | Licence Ouverte |
| Maires sortants 2026, Répertoire national des élus | Ministère de l'Intérieur (data.gouv.fr) | Licence Ouverte |
| Code officiel géographique 2026 | INSEE | Licence Ouverte |
| Recensement 2017 et 2023, revenus Filosofi 2023 (API Melodi) | INSEE | Licence Ouverte |
| Présidentielles 2017 et 2022 | Ministère de l'Intérieur (data.gouv.fr) | Licence Ouverte |
| Contours des communes 2025 | Etalab (data.gouv.fr) | ODbL |
| Affaires judiciaires : condamnations | [Casier Politique](https://casier-politique.fr/about) | CC BY-NC-SA 4.0 |
| Affaires judiciaires : tous stades | [Poligraph](https://poligraph.fr) | données publiques, code AGPL-3.0 |

Il n'existe pas de base officielle nominative des procédures visant des élus. Les
deux bases d'affaires sont des initiatives citoyennes, mises à jour en continu, qui
ne couvrent que les élus ayant une page Wikipédia ou une fiche Poligraph : surtout
des maires de grandes villes et des élus nationaux. Les limites qui en découlent
sont discutées dans la synthèse (notebook 00, section 7).

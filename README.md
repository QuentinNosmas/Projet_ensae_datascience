# Les électeurs sanctionnent-ils vraiment ?

Projet personnel de data science.

**Question.** Un maire visé par une affaire judiciaire (enquête, mise en examen,
condamnation) est-il sanctionné par les électeurs aux élections municipales
suivantes ? On compare les maires sortants des municipales de 2020 et de 2026 selon
qu'une affaire était connue ou non au moment du vote.

**Démarche.** On rapproche 741 affaires judiciaires recensées par deux bases
citoyennes (Casier Politique, Poligraph) des maires sortants de près de 44 000
communes x élection. Après élimination des homonymes et des maires partis en cours
de mandat, 75 sortants avaient une affaire connue au moment du vote. On les compare
à des sortants comparables par quatre approches : à taille de commune égale,
régressions avec contrôles (score précédent, taille, compétition, bord politique,
contexte socio-économique et présidentiel), appariement sur score de propension, et
comparaison à « notoriété égale » avec les élus dont l'affaire n'éclatera qu'après
le vote.

## Résultats

1. **Pas de sanction électorale robuste.** Les sortants visés par une affaire se
   représentent, conservent leur score et gagnent aussi souvent que des sortants
   comparables. Aucune approche ne trouve de perte de voix significative.
2. **Deux pièges expliquent les comparaisons naïves.** Le retour vers la moyenne :
   ces sortants avaient été moins bien élus la fois précédente (52 % contre 62 %),
   donc avaient moins à perdre ; la perte apparente de 5 points en 2026 disparaît
   une fois le score précédent contrôlé. La prime de notoriété : les élus recensés
   (grandes villes, figures nationales) gagnent davantage que les autres.
3. **Un seul indice de sanction, fragile.** À notoriété égale, une affaire révélée
   pendant le mandat réduit la probabilité de victoire d'environ 29 points
   (IC 95 % de -57 à -1, p = 0,04), sur une trentaine de cas et surtout en 2020.
4. **Ni le bord politique ni le stade de l'affaire ne changent ce constat.**
   La sanction ne paraît pas plus forte à gauche qu'à droite (écart non
   significatif). Une condamnation prononcée avant le vote réduirait la probabilité
   de victoire d'environ 26 points (p = 0,08), sans écart significatif avec une
   affaire pas encore jugée, et les élus déjà blanchis ne sont pas sanctionnés de
   façon détectable.

**Portée.** Les bases d'affaires ne couvrent que les élus assez connus pour avoir
une page Wikipédia ou une fiche Poligraph : les résultats valent pour les maires de
villes moyennes et grandes, pas pour les petites communes. Avec quelques dizaines
de cas, seul un effet important serait détectable.

Le détail est dans [`notebooks/00_synthese.ipynb`](notebooks/00_synthese.ipynb).

## Reproduire les résultats

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12 conseillé
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Les données ne sont pas versionnées. Au premier lancement, chaque source manquante
est téléchargée depuis son URL publique (environ 550 Mo ; prévoir 2 Go
d'espace disque une fois les archives décompressées), puis
nettoyée dans `data/processed/`. L'API de l'INSEE coupe parfois les transferts
longs : le téléchargement reprend alors automatiquement là où il s'était arrêté, et
un nouveau lancement repart du fichier partiel. On peut aussi tout préparer à l'avance :

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

**Sous Windows.** Placer le projet dans un chemin court (par exemple `C:\projets\`) :
au-delà de 260 caractères, Windows refuse de charger certaines bibliothèques de
`geopandas` installées dans `.venv`, et la carte des communes échoue. Le fichier
`.gitattributes` garde les instantanés de données intacts malgré la conversion des
fins de ligne, pour que leurs empreintes correspondent au manifeste.

## Organisation

| Fichier | Rôle |
|---|---|
| `scripts/download_data.py` | téléchargement reproductible des sources (reprise, contrôles) |
| `src/load.py` | lecture et nettoyage de chaque source, codes INSEE harmonisés |
| `src/panel.py` | panel « maire sortant x élection » : candidature, score, victoire |
| `src/affaires.py` | rattachement des affaires judiciaires aux sortants, variables de traitement, stade de l'affaire au vote |
| `src/covariables.py` | recensement, revenus, présidentielles : contexte de chaque commune |
| `src/analyse.py` | comparaisons à taille de commune égale, intervalles bootstrap |
| `src/modeles.py` | régressions (MCO, logit), appariement sur score de propension |
| `src/graphiques.py` | charte graphique, carte, graphique des estimations |
| `notebooks/00_synthese.ipynb` | synthèse d'étape : la démarche et les premiers résultats |
| `notebooks/01_exploration.ipynb` | diagnostic des sources |
| `notebooks/02_panel_sortants.ipynb` | comportement d'un sortant « ordinaire » |
| `notebooks/03_affaires.ipynb` | affaires judiciaires : rattachement, description, carte, comparaisons à taille égale |
| `notebooks/04_modelisation.ipynb` | régressions, appariement, délai, participation, bord politique, stade de l'affaire, robustesse |

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

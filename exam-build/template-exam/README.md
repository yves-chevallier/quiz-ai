# TexSmith Exam Template

Cette template apporte au moteur TeXSmith une mise en page d'examen prête à l'emploi basée sur la classe `exam` et les personnalisations HEIG-VD (`hexam.sty`). Elle reprend les objectifs décrits dans `../README.md`: ancres par question pour l'IA, page de garde normalisée et zones de réponse structurées.

## Fonctionnalités

- **Surcharge des niveaux Markdown**: `#` devient un `\titlequestion`, `##` un `\part`, `###` un `\subpart`, `####` un `\subsubpart`.
- **Bloc `!!! solution`** converti en environnement `solution`, donc masqué automatiquement quand `\printanswers` est inactif.
- **Extension `--- n ---`** (et `--- grid n ---`) qui génère `\fillwithdottedlines` ou `\fillwithgrid` via un hook renderer.
- **Cover page configurable**: titre, sous-titre, auteur, département, école, session, seed, etc.
- **Gradetable optionnelle**, directives en liste et champs libres (`course`, `duration`, `room`).

## Installation

```bash
# Depuis la racine du dépôt
uv pip install ./exam-build/template-exam
```

La template et les hooks renderer seront ensuite disponibles via les entry points `exam` et `texsmith_exam`.

## Utilisation

```bash
cd exam-build
texsmith render exam-example.md \
  --template ../template-exam \
  --build
```

- La compilation utilise LuaLaTeX + `minted` ⇒ `--build` ajoute automatiquement `-shell-escape`.
- Les attributs se passent via le front-matter `press` (cf. exemple).
- La syntaxe `--- n ---` / `--- grid n ---` et les champs `{{réponse}}` sont pris en charge automatiquement par la template (pas besoin d'extension Markdown explicite).

## Métadonnées supportées

| Champ front-matter      | Effet                                                |
| ----------------------- | ----------------------------------------------------- |
| `press.title`           | `\title`                                             |
| `press.subtitle`        | Sous-titre sur la couverture                          |
| `press.author(s)`       | Auteur affiché                                        |
| `press.department`      | `\department{...}`                                   |
| `press.school`          | `\school{...}`                                       |
| `press.date`            | Date sur la page de garde                             |
| `press.language`        | Langue Babel + option de classe (`french` par défaut) |
| `press.course`          | Ligne *Cours* dans le tableau meta                    |
| `press.session`         | Ligne *Session*                                       |
| `press.duration`        | Ligne *Durée*                                         |
| `press.room`            | Ligne *Salle*                                         |
| `press.seed`            | Ligne *Seed*                                          |
| `press.version`         | Ligne *Version*                                       |
| `press.directives`      | Liste d'items sous "Consignes"                       |
| `press.cover_notice`    | Texte centré sous le titre                            |
| `press.show_gradetable` | Booléen, affiche ou masque la table des points        |

## Markdown spécifique

- `--- 4 ---` devient `\fillwithdottedlines{4\baselineskip}` et `--- grid 8 ---` devient `\fillwithgrid{8\baselineskip}`.
- `{{réponse attendue}}` insère `\fillin[réponse attendue]{nelem}` avec une largeur approximative proportionnelle au texte.
- Les listes de tâches Markdown (`- [ ]`, `- [x]`) se transforment en `\begin{checkboxes}\choice...\end{checkboxes}` avec `\CorrectChoice` pour les cases cochées.
- `!!! solution "Titre"` crée un environnement `solution` sans forcer le titre "Solution".
- Les blocs de code utilisent toujours `hexam.sty` (`tcolorbox` + `minted`).

Consultez `exam-example.md` pour un flux complet allant du front matter aux questions et solutions.

---
press:
  subtitle: "TE-01"
  author: "Prof. Yves Chevallier"
  date: 2024-06-10
  school: "HEIG-VD"
  department: "Département TIN"
  course: "INFO1"
  duration: "90 minutes"
  room: "Bâtiment E31"
  directives:
    - Écrire votre nom et votre prénom sur la première page.
    - Écrire lisiblement, au stylo ou au crayon à papier gras.
    - Répondre aux questions dans les espaces appropriés.
    - Relire toutes vos réponses avant de rendre votre travail.
    - Rendre toutes les feuilles de ce travail écrit.
    - Les réponses données sur les feuilles de brouillon ne sont pas acceptées.
    - Aucun moyen de communication autorisé.
    - Toutes les réponses concernent le langage C (standard C20).
---

# Travail écrit Info1-TIN-B

## Numération

Pour chacune des constantes littérales suivantes indiquez leur équivalent:

1. binaire,
2. octal,
3. décimal signé,
4. décimal non signé et
5. hexadécimal.

Considérez que les nombres sont stockés en mémoire sur **8 bits** et que la
représentation signée utilise le **complément à deux**.
Complétez le tableau ci-dessous en remplissant les cases vides.

/// latex
\vskip 2em
\begingroup
\renewcommand{\arraystretch}{1.8}
\setlength{\tabcolsep}{1.1em}
///

| bin            | octal    | int      | uint    | hex      |
|----------------|----------|----------|---------|----------|
| 0b01011010     | {{0132}} | {{+90}}  | {{90}}  | {{0x5a}} |
| {{0b11000111}} | {{0307}} | {{-57}}  | 199     | {{0xc7}} |
| {{0b10010000}} | {{0220}} | -112     | {{144}} | {{0x90}} |
| {{0b01111101}} | 0175     | {{+125}} | {{125}} | {{0x7d}} |
| {{0b11111010}} | {{0372}} | {{-6}}   | {{250}} | 0xfa     |

/// latex
\endgroup
///

!!! solution
    Pour retrouver la valeur signée d'un nombre négatif, appliquez la
    règle du complément à deux (inversion des bits puis ajout de 1).
    Le passage en base octale s'obtient en regroupant les bits par
    paquets de trois à partir de la droite, et l'hexadécimal en paquets de quatre.

/// latex
\clearpage
///

## Syntaxe et identificateurs

### Syntaxe

Pour chacun des identificateurs suivants, indiquez s'ils sont corrects selon le standard C. S'ils sont invalides, proposez un nom compatible en C (ASCII, lettres/chiffres et `_` uniquement).

1. `résumé`

   !!! solution {lines=1}

       **Incorrect** (accent). Proposition : `resume` ou `summary`.

2. `_AMountain_`

    !!! solution {lines=1}

        **Correct**.

3. `while`

    !!! solution {lines=1}

        **Incorrect** car mot-clé réservé. Proposition : `case_value`, `while_loop`, etc.

4. `std99`

    !!! solution {lines=1}

        **Correct**.

5. `data-stream`

    !!! solution {lines=1}

        **Incorrect** (tiret). Proposition : `data_stream`.

## Question à choix multiples

Quel est la capitale de la France ?

- [ ] Berlin
- [ ] Madrid
- [x] Paris
- [ ] Rome
- [ ] Lisbonne

## Question ouverte

Résoudre l'équation quadratique suivante: $3x^2 + 2x - 5C = 0$, où les coefficients sont définis comme suit:

Indiquer les solutions $x_1$ et $x_2$ ci-dessous:

!!! solution {lines=2}

    La solution de l'équation quadratique $Ax^2 + Bx + C = 0$ est donnée par la formule:

    $$ x = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A} $$

    En substituant les valeurs de $A = 3$, $B = 2$, et $C = -5$ dans la formule, nous obtenons:

    $$ x = \frac{-2 \pm \sqrt{2^2 - 4 \cdot 3 \cdot (-5)}}{2 \cdot 3} $$

    Calculons le discriminant:

    $$ D = B^2 - 4AC = 2^2 - 4 \cdot 3 \cdot (-5) = 4 + 60 = 64 $$

    Maintenant, nous pouvons calculer les racines:

    $$ x = \frac{-2 \pm \sqrt{64}}{6} = \frac{-2 \pm 8}{6} $$

    Cela nous donne deux solutions:

    En toutes lettres, on attend la {{cinquième}} racine.

    1. Pour la racine positive:

        $$ x_1 = \frac{-2 + 8}{6} = \frac{6}{6} = 1 $$

    2. Pour la racine négative:

        $$ x_2 = \frac{-2 - 8}{6} = \frac{-10}{6} = -\frac{5}{3} $$

    Donc les solutions sont $x_1 = 1 $ et $ x_2 = -\frac{5}{3}$.

## Fill in the blanks

Compléter la phrase suivante en remplissant les espaces vides:

La Terre est la {{ cinquième }} planète à partir du {{ Soleil }} dans le système solaire.

## Avec subparties

Considérez les questions suivantes:

### Quelle est la formule chimique de l'eau ?

!!! solution {lines=2}

    La formule chimique de l'eau est H₂O.

### Combien de continents y a-t-il sur Terre ?

!!! solution {lines=1}

    Il y a 7 continents sur Terre.

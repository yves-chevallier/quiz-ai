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

#### `résumé`

!!! solution {lines=1}

    **Incorrect** (accent). Proposition : \texttt{resume} ou \texttt{summary}.

#### `_AMountain_`

!!! solution {lines=1}

    **Correct**.

#### `while`

!!! solution {lines=1}

    **Incorrect** car mot-clé réservé. Proposition : \texttt{case\_value}, \texttt{while\_loop}, etc.

#### `std99`

!!! solution {lines=1}

    **Correct**.

#### `data-stream`

!!! solution {lines=1}

    **Incorrect** (tiret). Proposition : \texttt{data\_stream}.

### Constantes littérales

Pour chaque constante littérale suivante, indiquez si elle est **correcte** et,
si applicable, précisez le **type** associé.

#### `42uL`

!!! solution {lines=1}

    `42uL` : Correct, type `unsigned long int`.

#### `0758`

!!! solution {lines=1}

    `0758` : Incorrect (les littéraux octaux n'acceptent pas 8 ou 9).

#### `.128f`

!!! solution {lines=1}

    `.128f` : Correct, type `float`.

#### `'\\x41'`

!!! solution {lines=1}

    `'\\x41'` : Correct, type `char`.

#### `0b00001010`

!!! solution {lines=1}

    `0b00001010` : Correct à partir de C23 (`int`).

/// latex
\clearpage
///

/// latex
\lstDeleteShortInline|
///

## Entrées sorties

### Formatage avec `printf`

Pour les appels de fonction `printf` suivants, indiquez l'affichage exact
produit sur *stdout* ou, en cas d'erreur, la nature de cette dernière.
Représentez la sortie dans l'espace approprié, un caractère par case.
Utilisez la notation des caractères d'échappement du langage C pour
les caractères non imprimables (p.ex. `\n`).
Terminez chaque sortie par une croix sur toute la case.

Considérez les états des variables suivants:

```c
short s = -27;
unsigned char uc = 201;
char c = 'c'; // Valeur ASCII de 'c': 99
double x = 2.71828;
```

#### `printf("|0x%05x|\\n", uc);`

!!! solution {lines=2}

    \texttt{\textbar 0x000c9\textbar \textbackslash n}

#### `printf("%+7.2f\\n", s / 2.0);`

!!! solution {lines=2}

    \texttt{\ \ -13.50\textbackslash n}

#### `printf("%c%c%c%c %hhd", c, c - 2, 'a' + 2, 97, c);`

!!! solution {lines=1}
    \texttt{caca\ 99}

#### `printf(">%-06.1f<", x);`

!!! solution {lines=1}
    \texttt{>0002.7<}

#### `printf("%-*.*f%d\\n", 7, 3, 5.4321, 42);`

!!! solution {lines=1}
    \texttt{5.432\ \ 42}

### Formatage avec `scanf`

Soient les déclarations suivantes :

```c
int r = 0, n = 0, m = 0;
double y = 0.0;
char ch = '0';
```

Pour les appels `sscanf` ci-dessous, indiquez:

1. la valeur des variables affectées;
2. la valeur de retour de `r`.

#### `r = sscanf("  -15 0x1f", "%d %i", &n, &m);`

!!! solution {lines=1}

    n = -15, m = 31, r = 2

#### `r = sscanf("42kg", "%2d%c", &n, &ch);`

!!! solution {lines=1}

    n = 42, ch = 'k', r = 2

#### `r = sscanf("0.75,12", "%lf,%d", &y, &n);`

!!! solution {lines=1}

    y = 0.75, n = 12, r = 2

#### `r = sscanf("abc", "%d", &n);`

!!! solution {lines=1}

    n = 0, r = 0

#### `r = sscanf("9.82", "%d %d", &n, &m);`

!!! solution {lines=1}

    n = 9, m = 82, r = 2

/// latex
\lstMakeShortInline|
///

/// latex
\clearpage
///

## Boucles et contrôle de flux

Donnez les valeurs affichées sur `stdout` pour chaque fragment de code C suivant :

### Boucles simples

```c
int s = 1;
while (s < 20) {
    printf("%d ", s);
    s <<= 1;
}
```

!!! solution {lines=1}

    `1 2 4 8 16 `

### Boucles avec contrôle de flux

```c
for (int i = 5; i > 0; --i) {
    if (i % 2 == 0) continue;
    printf("%d", i);
}
```

!!! solution {lines=1}

    `531`

### Boucles imbriquées

```c
int total = 0;
for (int i = 1; i <= 3; ++i) {
    for (int j = i; j <= 3; ++j) {
        total += j;
    }
}
printf("%d", total);
```

!!! solution {lines=1}

    `18`

### Boucles avec conditions

```c
int i = 1, t = 0;
do {
    t += i;
    printf("%d;", t);
    i += 2;
} while (t < 15);
```

!!! solution {lines=1}

    `1;4;9;16;`

/// latex
\clearpage
///

## Programmation

Écrire un programme complet en C qui calcule la résistance équivalente d'un
réseau de `N` résistances montées en parallèle, selon :

$$
R_{\mathrm{eq}} = \left(\frac{1}{R_1} + \frac{1}{R_2} + \cdots + \frac{1}{R_N}\right)^{-1}
$$

- Les valeurs sont fournies via les arguments de la ligne de commande (`argv[1..N]`).
- Toute valeur manquante, nulle ou négative doit provoquer un message
    d'erreur et un code de retour non nul.
- Vous pouvez utiliser `sscanf` pour valider les entrées.

Exemples d'exécution :

```bash
$ ./parallel 120 220 330
54.6796
$ ./parallel 100 100 100 100 100
20.0
$ ./parallel 47 -10 10
Error: Invalid resistance values
```

/// latex
\clearpage
///

## Quiz

### Quel est la capitale de la France ?

- [ ] Berlin
- [ ] Madrid
- [x] Paris
- [ ] Rome
- [ ] Lisbonne

### Quel est le résultat de l'expression `5 + 3 * 2` en C ?

- [ ] 16
- [x] 21
- [ ] 13
- [ ] 10
- [ ] 11

### Quel est l'élément chimique dont le symbole est `Fe` ?

- [ ] Fluor
- [ ] Francium
- [ ] Ferium
- [ ] Fermium
- [ ] Fermate
- [x] Fer

/// latex
\clearpage
///

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

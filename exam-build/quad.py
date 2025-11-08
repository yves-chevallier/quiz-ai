import random
from math import gcd
import sys
import json

def generate_quadratic(
    seed: int,
    root_range: tuple[int, int] = (-10, 10),
    leading_range: tuple[int, int] = (1, 10),
    allow_equal_roots: bool = True,
    primitive: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """
    Génère (A, B, C) et (X1, X2) pour une équation du 2e degré à racines entières.

    Paramètres
    ----------
    seed : int
        Graine pour une génération déterministe.
    root_range : (min, max)
        Intervalle inclusif pour les racines entières (X1, X2).
    leading_range : (minA, maxA)
        Valeurs absolues possibles pour A (A peut être ± au hasard). A≠0.
    allow_equal_roots : bool
        Si False, force X1 != X2 (pas de racine double).
    primitive : bool
        Si True, divise (A, B, C) par leur PGCD et rend A > 0.

    Retour
    ------
    ((A, B, C), (X1, X2))
        Coefficients et racines (ordonnées X1 ≤ X2).
    """
    rng = random.Random(seed)

    # Choisir A ≠ 0
    a = 0
    while a == 0:
        k = rng.randint(leading_range[0], leading_range[1])
        a = k if rng.choice([True, False]) else -k

    # Choisir deux racines entières
    lo, hi = root_range
    if allow_equal_roots:
        r1 = rng.randint(lo, hi)
        r2 = rng.randint(lo, hi)
    else:
        # Rejeter tant que r1 == r2
        while True:
            r1 = rng.randint(lo, hi)
            r2 = rng.randint(lo, hi)
            if r1 != r2:
                break

    # Construire B et C à partir des racines
    b = -a * (r1 + r2)
    c = a * r1 * r2

    if primitive:
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g > 1:
            a //= g; b //= g; c //= g
        if a < 0:  # normaliser le signe
            a, b, c = -a, -b, -c

    x1, x2 = sorted((r1, r2))
    return (a, b, c), (x1, x2)


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 12345

    coeffs, roots = generate_quadratic(
        seed=seed,
        root_range=(-9, 9),
        leading_range=(1, 7),
        allow_equal_roots=False,
        primitive=True,
    )
    print(json.dumps({
        "A": coeffs[0],
        "B": coeffs[1],
        "C": coeffs[2],
        "X1": roots[0],
        "X2": roots[1],
    }))

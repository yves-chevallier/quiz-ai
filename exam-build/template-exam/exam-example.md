---
press:
  title: Travail écrit Info1-TIN-B
  subtitle: TE-Template
  author: Prof. Yves Chevallier
  department: Département TIN
  school: Haute École d'Ingénierie et de Gestion du Canton de Vaud
  date: 4 novembre 2025
  language: fr
  course: Programmation C
  session: Automne 2025
  duration: 90 minutes
  room: Aula B12
  seed: 42A
  version: V1
  directives:
    - Écrire votre nom et prénom sur la première page.
    - Répondre lisiblement dans les zones prévues.
    - Aucun appareil électronique n'est autorisé.
  show_gradetable: true
---

# Analyse de code

Ce quiz valide votre compréhension des entrées/sorties en C. Chaque sous-section correspond à une `\part` dans la classe `exam`.

## Questions rapides

### Conversion décimale

Expliquez comment convertir rapidement un entier non signé en hexadécimal mentalement.

--- 4 ---

!!! solution "Indice"
    Utiliser les puissances de 16 et regrouper les bits par paquets de 4.

### Pièges classiques

Citez au minimum deux erreurs fréquentes lors de la lecture d'un `scanf`.

- Utilisation d'un format incorrect.
- Adresse manquante pour les entiers (`&`).

!!! solution
    Mentionner par exemple: oubli du `&`, mauvais séparateur, laisser un `\n` traîner dans le buffer.

## Lecture de fichier

### Compréhension

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("data.txt", "r");
    if (!fp) {
        return 1;
    }

    int value = 0;
    while (fscanf(fp, "%d", &value) == 1) {
        printf("%d\n", value * value);
    }
    fclose(fp);
}
```

Expliquez ce que produit ce programme et quel est l'effet si `data.txt` contient une ligne vide au milieu.

--- grid 6 ---

!!! solution "Comportement"
    Le programme lit chaque entier, affiche son carré puis ignore les lignes vides car `fscanf` saute l'espace blanc.

### Validation

Donnez une entrée qui provoquerait une erreur lors de la lecture précédente et justifiez pourquoi.

--- 3 ---

!!! solution
    Tout caractère non numérique (`abc`) stoppe la boucle car `fscanf` retourne 0.

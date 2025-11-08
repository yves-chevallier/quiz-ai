#include <stdio.h>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s R1 R2 R3\n", argv[0]);
        return 1;
    }

    double r1, r2, r3;
    if (sscanf(argv[1], "%lf", &r1) != 1 ||
        sscanf(argv[2], "%lf", &r2) != 1 ||
        sscanf(argv[3], "%lf", &r3) != 1) {
        printf("Paramètres invalides\n");
        return 1;
    }

    if (r1 <= 0 || r2 <= 0 || r3 <= 0) {
        printf("Résistances positives requises\n");
        return 1;
    }

    double denom = 1.0 / r1 + 1.0 / r2 + 1.0 / r3;
    if (denom <= 0.0) {
        printf("Combinaison invalide\n");
        return 1;
    }

    double req = 1.0 / denom;
    printf("Résistance équivalente: %.4f ohms\n", req);

    return 0;
}

#include <stdio.h>

int main(void) {
    {
        int s = 1;
        while (s < 20) {
            printf("%d ", s);
            s <<= 1;
        }
    }
    printf("\n");

    {
        for (int i = 5; i > 0; --i) {
            if (i % 2 == 0) continue;
            printf("%d", i);
        }
    }
    printf("\n");

    {
        int total = 0;
        for (int i = 1; i <= 3; ++i) {
            for (int j = i; j <= 3; ++j) {
                total += j;
            }
        }
        printf("%d", total);
    }
    printf("\n");

    {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i == j) break;
                printf("%d%d ", i, j);
            }
        }
    }
    printf("\n");

    {
        int i = 1, t = 0;
        do {
            t += i;
            printf("%d;", t);
            i += 2;
        } while (t < 15);
    }
    printf("\n");

    return 0;
}

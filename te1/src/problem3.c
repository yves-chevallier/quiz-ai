#include <stdio.h>

int main(void) {
    short s = -27;
    unsigned char uc = 201;
    char c = 'd';
    double x = 2.71828;
    char text[] = "gadget";

    printf("|%#05x|\n", uc);
    printf("%+7.2f\n", s / 5.0);
    printf("%.4s_%d\n", text + 2, c - 'a');
    printf("%-6.1e|\n", x);
    printf("%6u\n", (unsigned)(s + uc));

    return 0;
}

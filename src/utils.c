#include "utils.h"

void print_progress(char *prefix, int current, int total, int width) {
    const char bar = '#';
    const char blank = ' ';

    float percentage = (float)current / total;

    printf("\r%s %d/%d [", prefix, current, total);
    for (int i=0; i<width; i++) {
        if (percentage > (float)i / width) {
            printf("%c", bar);
        } else {
            printf("%c", blank);
        }
    }
    printf("] %.2f%%", percentage * 100);
    fflush(stdout);
    
    if (current >= total) {
        printf("\n\n");
    }
}
#include <cblas.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    /* ARGPARSE */
    if (argc < 4) {
        printf("Usage: %s [flops] [dim_vec] [num_split] [num_iter]\n", argv[0]);
        exit(1);
    }

    unsigned long long flops = strtoull(argv[1], NULL, 10);
    unsigned long long K = strtoull(argv[2], NULL, 10);
    unsigned long long num_split = strtoull(argv[3], NULL, 10);
    unsigned long long num_iter = strtoull(argv[4], NULL, 10);
    
    openblas_set_num_threads(1);

    unsigned long long M = flops / 2 / K;

    float *mat = malloc(sizeof(float) * M * K / num_split);
    float *vec = malloc(sizeof(float) * K);
    float *out = malloc(sizeof(float) * M);

    if (!mat) { printf("size too big! split more.\n"); exit(1); }

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i=0; i<num_iter; i++) {
        for (int j=0; j<num_split; j++) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, M / num_split, K, 1.0, mat, K, vec, 1, 0.0, out, 1);

        }
    }

    gettimeofday(&end_time, NULL);

    float eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;
    printf("ETA: %f (msec)\n");

    unsigned long long result_flops = M * K * 2 * num_iter;
    printf("Total FLOPS: %llu\n", result_flops);


    return 0;
}
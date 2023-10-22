#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

int main (int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s [mem_size(MB)] [num_iter]\n", argv[0]);
        exit(1);
    }

    size_t mem_size_mb = atoi(argv[1]);
    size_t mem_size = mem_size_mb * 1024 * 1024;
    size_t num_iter = atoi(argv[2]);

    void *mem_area = malloc(sizeof(char) * mem_size);

    /** WARMUP **/

    for (int i = 0; i < num_iter; i++) {
        for (int j = 0; j < mem_size; j += 32 * 32) {
            _mm256_loadu_si256(mem_area + j);
            _mm256_loadu_si256(mem_area + j + 32);
            _mm256_loadu_si256(mem_area + j + 64);
            _mm256_loadu_si256(mem_area + j + 96);
            _mm256_loadu_si256(mem_area + j + 128);
            _mm256_loadu_si256(mem_area + j + 160);
            _mm256_loadu_si256(mem_area + j + 192);
            _mm256_loadu_si256(mem_area + j + 224);
            _mm256_loadu_si256(mem_area + j + 240);
            _mm256_loadu_si256(mem_area + j + 256);
            _mm256_loadu_si256(mem_area + j + 288);
            _mm256_loadu_si256(mem_area + j + 320);
            _mm256_loadu_si256(mem_area + j + 352);
            _mm256_loadu_si256(mem_area + j + 384);
            _mm256_loadu_si256(mem_area + j + 416);
            _mm256_loadu_si256(mem_area + j + 448);
            _mm256_loadu_si256(mem_area + j + 480);
            _mm256_loadu_si256(mem_area + j + 512);
            _mm256_loadu_si256(mem_area + j + 544);
            _mm256_loadu_si256(mem_area + j + 576);
            _mm256_loadu_si256(mem_area + j + 608);
            _mm256_loadu_si256(mem_area + j + 640);
            _mm256_loadu_si256(mem_area + j + 672);
            _mm256_loadu_si256(mem_area + j + 704);
            _mm256_loadu_si256(mem_area + j + 736);
            _mm256_loadu_si256(mem_area + j + 768);
            _mm256_loadu_si256(mem_area + j + 800);
            _mm256_loadu_si256(mem_area + j + 832);
            _mm256_loadu_si256(mem_area + j + 864);
            _mm256_loadu_si256(mem_area + j + 896);
            _mm256_loadu_si256(mem_area + j + 928);
            _mm256_loadu_si256(mem_area + j + 960);
        }
    }

    /** BENCH **/

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < num_iter; i++) {
        for (int j = 0; j < mem_size; j += 32 * 32) {
            _mm256_loadu_si256(mem_area + j);
            _mm256_loadu_si256(mem_area + j + 32);
            _mm256_loadu_si256(mem_area + j + 64);
            _mm256_loadu_si256(mem_area + j + 96);
            _mm256_loadu_si256(mem_area + j + 128);
            _mm256_loadu_si256(mem_area + j + 160);
            _mm256_loadu_si256(mem_area + j + 192);
            _mm256_loadu_si256(mem_area + j + 224);
            _mm256_loadu_si256(mem_area + j + 240);
            _mm256_loadu_si256(mem_area + j + 256);
            _mm256_loadu_si256(mem_area + j + 288);
            _mm256_loadu_si256(mem_area + j + 320);
            _mm256_loadu_si256(mem_area + j + 352);
            _mm256_loadu_si256(mem_area + j + 384);
            _mm256_loadu_si256(mem_area + j + 416);
            _mm256_loadu_si256(mem_area + j + 448);
            _mm256_loadu_si256(mem_area + j + 480);
            _mm256_loadu_si256(mem_area + j + 512);
            _mm256_loadu_si256(mem_area + j + 544);
            _mm256_loadu_si256(mem_area + j + 576);
            _mm256_loadu_si256(mem_area + j + 608);
            _mm256_loadu_si256(mem_area + j + 640);
            _mm256_loadu_si256(mem_area + j + 672);
            _mm256_loadu_si256(mem_area + j + 704);
            _mm256_loadu_si256(mem_area + j + 736);
            _mm256_loadu_si256(mem_area + j + 768);
            _mm256_loadu_si256(mem_area + j + 800);
            _mm256_loadu_si256(mem_area + j + 832);
            _mm256_loadu_si256(mem_area + j + 864);
            _mm256_loadu_si256(mem_area + j + 896);
            _mm256_loadu_si256(mem_area + j + 928);
            _mm256_loadu_si256(mem_area + j + 960);
        }
    }

    gettimeofday(&end_time, NULL);
    float total_eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e6;
    unsigned long long total_flops = 16 * (mem_size / (64*4)) * num_iter;

    printf("================== RESULT ==================\n");
    printf("+ Loaded         %10lu MBytes         +\n", mem_size_mb);
    printf("+ Executed fmla: %10llu Million        +\n", total_flops / 1000000);
    printf("+ repeated       %10lu times          +\n", num_iter);
    printf("+ Total ETA:     %10.2f sec            +\n", total_eta);
    printf("+ ---------------- SPECS. ---------------- +\n");
    printf("+ Memory bandwidth: %10.2f MBytes/sec  +\n", (float)(mem_size >> 20) * num_iter / total_eta);
    printf("+ FL32 calc.:       %10.2f MFLOPS      +\n", (float)total_flops / total_eta / 10000000);
    printf("============================================\n");

    return 0;
}
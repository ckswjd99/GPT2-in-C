#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s [mem_size(MB)] [num_iter]\n", argv[0]);
        exit(1);
    }

    size_t mem_size_mb = atoi(argv[1]);
    size_t mem_size = mem_size_mb * 1024 * 1024;
    size_t num_iter = atoi(argv[2]);

    void *mem_area = malloc(sizeof(char) * mem_size);
    unsigned long long *temp;

    /** WARMUP **/
    for (int i = 0; i < num_iter; i++) {
        for (int j = 0; j < mem_size; j += 16 * sizeof(unsigned long long)) {
            unsigned long long *temp_mem = mem_area + j;
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
        }
    }

    /** BENCH **/

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < num_iter; i++) {
        for (int j = 0; j < mem_size; j += 16 * sizeof(unsigned long long)) {
            unsigned long long *temp_mem = mem_area + j;
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            *temp = *(temp_mem++);
            
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
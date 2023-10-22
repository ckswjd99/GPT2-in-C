#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

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
        for (void *mem_temp = mem_area; mem_temp < mem_area + mem_size;) {
            __asm__ __volatile__ (
                "ld4 {v16.16b, v17.16b, v18.16b, v19.16b}, [%0], #64\n"
                "ld4 {v20.16b, v21.16b, v22.16b, v23.16b}, [%0], #64\n"
                "ld4 {v24.16b, v25.16b, v26.16b, v27.16b}, [%0], #64\n"
                "ld4 {v28.16b, v29.16b, v30.16b, v31.16b}, [%0], #64\n"
                : "=r" (mem_temp)
                : "0" (mem_temp)
                : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );


        }
    }

    /** BENCH **/

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    for (int i = 0; i < num_iter; i++) {
        for (void *mem_temp = mem_area; mem_temp < mem_area + mem_size;) {
            __asm__ __volatile__ (
                "ld4 {v16.16b, v17.16b, v18.16b, v19.16b}, [%0], #64\n"
                "ld4 {v20.16b, v21.16b, v22.16b, v23.16b}, [%0], #64\n"
                "ld4 {v24.16b, v25.16b, v26.16b, v27.16b}, [%0], #64\n"
                "ld4 {v28.16b, v29.16b, v30.16b, v31.16b}, [%0], #64\n"

                "fmla v16.4s, v16.4s, v17.4s\n"
                "fmla v18.4s, v18.4s, v19.4s\n"
                "fmla v20.4s, v20.4s, v21.4s\n"
                "fmla v22.4s, v22.4s, v23.4s\n"
                "fmla v24.4s, v24.4s, v25.4s\n"
                "fmla v26.4s, v26.4s, v27.4s\n"
                "fmla v28.4s, v28.4s, v29.4s\n"
                "fmla v30.4s, v30.4s, v31.4s\n"
                
                "fmla v16.4s, v16.4s, v17.4s\n"
                "fmla v18.4s, v18.4s, v19.4s\n"
                "fmla v20.4s, v20.4s, v21.4s\n"
                "fmla v22.4s, v22.4s, v23.4s\n"
                "fmla v24.4s, v24.4s, v25.4s\n"
                "fmla v26.4s, v26.4s, v27.4s\n"
                "fmla v28.4s, v28.4s, v29.4s\n"
                "fmla v30.4s, v30.4s, v31.4s\n"
                
                : "=r" (mem_temp)
                : "0" (mem_temp)
                : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );
        }
    }

    gettimeofday(&end_time, NULL);
    float total_eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e6;
    unsigned long long total_flops = 16 * (mem_size / (64*4)) * num_iter;

    printf("================== RESULT ==================\n");
    printf("+ Loaded         %10d MBytes         +\n", mem_size_mb);
    printf("+ Executed fmla: %10u Million        +\n", total_flops / 1000000);
    printf("+ repeated       %10d times          +\n", num_iter);
    printf("+ Total ETA:     %10.2f sec            +\n", total_eta);
    printf("+ ---------------- SPECS. ---------------- +\n");
    printf("+ Memory bandwidth: %10.2f MBytes/sec  +\n", (float)(mem_size >> 20) * num_iter / total_eta);
    printf("+ FL32 calc.:       %10.2f MFLOPS      +\n", (float)total_flops / total_eta / 10000000);
    printf("============================================\n");

    return 0;
}
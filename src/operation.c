#include "operation.h"

static void fast_sgemv_neon(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out);

void fast_sgemv(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out) {
    if (N % 64 == 0) {
        fast_sgemv_neon(M, N, alpha, mat, vec, beta, out);
    } else {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, mat, N, vec, 1, beta, out, 1);
    }
}

void fast_sgemv_neon(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out) {

    __asm__ __volatile__ (
        "mov x0, %0\n"  // M
        "mov x1, %1\n"  // N
        "mov x2, x1\n"  // N'
        "mov x3, %2\n"  // mat
        "mov x4, %3\n"  // vec
        "mov x5, x4\n"  // vec'
        "mov x6, %4\n"  // out
        
        "1:\n"
        "mov x5, x4\n"
        "mov x2, x1\n"
        
        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        
        "2:\n"
        // load vector
        "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x5], #64\n"
        "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x5], #64\n"

        // load matrix row (0)
        "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x3], #64\n"
        "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x3]\n"
        "sub x3, x3, #64\n"
        "add x3, x3, x1\n"
        "add x3, x3, x1\n"
        "add x3, x3, x1\n"
        "add x3, x3, x1\n"
        
        "fmla v0.4s, v8.4s, v24.4s\n"
        "fmla v0.4s, v9.4s, v25.4s\n"
        "fmla v0.4s, v10.4s, v26.4s\n"
        "fmla v0.4s, v11.4s, v27.4s\n"
        
        "fmla v0.4s, v12.4s, v28.4s\n"
        "fmla v0.4s, v13.4s, v29.4s\n"
        "fmla v0.4s, v14.4s, v30.4s\n"
        "fmla v0.4s, v15.4s, v31.4s\n"
        
        // load matrix row (1)
        "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x3], #64\n"
        "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x3], #64\n"
        "sub x3, x3, x1\n"
        "sub x3, x3, x1\n"
        "sub x3, x3, x1\n"
        "sub x3, x3, x1\n"
        
        "fmla v1.4s, v16.4s, v24.4s\n"
        "fmla v1.4s, v17.4s, v25.4s\n"
        "fmla v1.4s, v18.4s, v26.4s\n"
        "fmla v1.4s, v19.4s, v27.4s\n"
        
        "fmla v1.4s, v20.4s, v28.4s\n"
        "fmla v1.4s, v21.4s, v29.4s\n"
        "fmla v1.4s, v22.4s, v30.4s\n"
        "fmla v1.4s, v23.4s, v31.4s\n"
        
        "subs x2, x2, #32\n"
        "bgt 2b\n"
        
        "faddp v0.4s, v0.4s, v0.4s\n"
        "faddp s0, v0.2s\n"

        "faddp v1.4s, v1.4s, v1.4s\n"
        "faddp s1, v1.2s\n"

        // x3을 N * (2 - 1) * 4 만큼 증가
        "add x3, x3, x1\n"
        "add x3, x3, x1\n"
        "add x3, x3, x1\n"
        "add x3, x3, x1\n"

        "str s0, [x6], #4\n"
        "str s1, [x6], #4\n"
        
        "subs x0, x0, #2\n"
        
        "bgt 1b\n"

    : "+r" (M), "+r" (N), "+r" (mat), "+r" (vec), "+r" (out)
    ::
    "x0", "x1", "x2", "x3", "x4", "x5", "x6",
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22","v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );

}
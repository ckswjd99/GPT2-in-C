#include "operation.h"

void sgemv_custom(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out) {
    assert(N % 64 == 0);

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
        "fmov s16, wzr\n"  
        
        "2:\n"
        "ld4 {v8.4s, v9.4s, v10.4s, v11.4s}, [x3], #64\n"
        "ld4 {v12.4s, v13.4s, v14.4s, v15.4s}, [x3], #64\n"
        "ld4 {v20.4s, v21.4s, v22.4s, v23.4s}, [x5], #64\n"
        "ld4 {v24.4s, v25.4s, v26.4s, v27.4s}, [x5], #64\n"
        
        "fmla v0.4s, v8.4s, v20.4s\n"
        "fmla v1.4s, v9.4s, v21.4s\n"
        "fmla v2.4s, v10.4s, v22.4s\n"
        "fmla v3.4s, v11.4s, v23.4s\n"
        
        "fmla v0.4s, v12.4s, v24.4s\n"
        "fmla v1.4s, v13.4s, v25.4s\n"
        "fmla v2.4s, v14.4s, v26.4s\n"
        "fmla v3.4s, v15.4s, v27.4s\n"

        
        "ld4 {v16.4s, v17.4s, v18.4s, v19.4s}, [x3], #64\n"
        "ld4 {v20.4s, v21.4s, v22.4s, v23.4s}, [x3], #64\n"
        "ld4 {v28.4s, v29.4s, v30.4s, v31.4s}, [x5], #64\n"
        "ld4 {v8.4s, v9.4s, v10.4s, v11.4s}, [x5], #64\n"
        
        "fmla v0.4s, v16.4s, v28.4s\n"
        "fmla v1.4s, v17.4s, v29.4s\n"
        "fmla v2.4s, v18.4s, v30.4s\n"
        "fmla v3.4s, v19.4s, v31.4s\n"
        
        "fmla v0.4s, v20.4s, v8.4s\n"
        "fmla v1.4s, v21.4s, v9.4s\n"
        "fmla v2.4s, v22.4s, v10.4s\n"
        "fmla v3.4s, v23.4s, v11.4s\n"
        
        "subs x2, x2, #64\n"
        
        "bgt 2b\n"
        
        "faddp v0.4s, v0.4s, v1.4s\n"
        "faddp v2.4s, v2.4s, v3.4s\n"
        "faddp v0.4s, v0.4s, v2.4s\n"
        "faddp v0.4s, v0.4s, v0.4s\n"
        "faddp s16, v0.2s\n"
        "str s16, [x6], #4\n"
        
        "subs x0, x0, #1\n"
        
        "bgt 1b\n"

    : "+r" (M), "+r" (N), "+r" (mat), "+r" (vec), "+r" (out)
    ::
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "s16",
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
    "v9", "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22","v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );

}
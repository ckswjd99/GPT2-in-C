#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <assert.h>

void sgemv_custom(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out);

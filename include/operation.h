#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <string.h>
#include <cblas.h>
#include <math.h>

#ifndef M_PI
#define M_PI                ((float)3.14159265358979323846)
#endif

void layer_normalize(int N, float *vector, float *W, float *B, float *buf_sizeN, float *ones);
void layer_linear(int M, int N, float *input, float *W, float *B, float *output);
void layer_softmax(int N, float *vector);
void layer_GeLU(int N, float *vector);

int vector_argmax(int m, float *x, int incx);
void vector_onehot(float* dest, int n, int idx);

void fast_sgemv(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out);

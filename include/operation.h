#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cblas.h>
#include <math.h>

#ifndef M_PI
#define M_PI                ((float)3.14159265358979323846)
#endif

#define DTYPE_INT8          0
#define DTYPE_FP16          1
#define DTYPE_FP32          2
#define DTYPE_FP64          3

#define TENSOR_MAX_DIM      16

struct tensor_t {
    int dtype;
    unsigned int dtype_bytesize;

    unsigned int dims;
    unsigned int shape[TENSOR_MAX_DIM];

    float *data;
    unsigned int tensor_bytesize;
};

unsigned int dtype_to_bytesize(int dtype);
struct tensor_t *new_tensor(int dtype, unsigned int dims, ...);
void free_tensor(struct tensor_t *tensor);

void layer_normalize(int N, float *vector, float *W, float *B, float *buf_sizeN, float *ones);
void layer_linear(int M, int N, float *input, float *W, float *B, float *output);
void layer_softmax(int N, float *vector);
void layer_GeLU(int N, float *vector);

int vector_argmax(int m, float *x, int incx);
void vector_onehot(float* dest, int n, int idx);

void fast_sgemv(unsigned int M, unsigned int N, float alpha, float *mat, float *vec, float beta, float *out);

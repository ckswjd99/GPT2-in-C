#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <limits.h>

#ifdef CBLAS_ATLAS
#include <cblas-atlas.h>
#else
#include <cblas.h>
#endif

#ifndef M_PI
#define M_PI                ((float)3.14159265358979323846)
#endif

#ifndef INT32_MIN
#define INT32_MIN INT_MIN
#endif

#include "configs.h"

#define DECODER_NUM_TOKEN_INIT  256

#define LOAD_BUFFER_SIZE    256

typedef struct decoder_t decoder_t;
typedef struct GPT2Model_t GPT2Model_t;


struct decoder_t {
    /* CONFIGS */
    int d_hidden;
    int d_head;
    int d_ffn;

    /* PARAMS */
    // Utils
    float *ones;        // [768], filled with 1

    // Layer Normalization 1
    float *W_ln1;       // [768]
    float *B_ln1;       // [768]

    // QKV
    float *W_Q;         // [768, 768]
    float *B_Q;         // [768]
    float *W_K;         // [768, 768]
    float *B_K;         // [768]
    float *W_V;         // [768, 768]
    float *B_V;         // [768]

    // MHA
    float *W_O;         // [768, 768]
    float *B_O;         // [768]

    // Layer Normalization 2
    float *W_ln2;       // [768]
    float *B_ln2;       // [768]

    // FFN
    float *W_ffn1;      // [768]
    float *B_ffn1;      // [3072]
    float *W_ffn2;      // [768]
    float *B_ffn2;      // [3072]

    /* FEATURES */
    float *Q;
    float *K;
    float *V;

    /* BUFFERS */
    int _num_inferenced_token;
    float *_mem_start;
    float *_buf_embedded;
    float *_buf_ln1;
    float *_buf_ln1_temp;
    float *_buf_q;
    float *_buf_sha;
    float *_buf_o;
    float *_buf_attn;
    float *_buf_ln2;
    float *_buf_ln2_temp;
    float *_buf_ffn1;
    float *_buf_ffn2;

    /* DEBUG */
    #ifdef DEBUG
    unsigned long long _debug_flops_total;
    unsigned long long _debug_flops_last;
    float _debug_eta_total;   // (msec)
    float _debug_eta_last;    // (msec)
    #endif
};

decoder_t *new_decoder(int d_hidden, int d_head, int d_ffn);
void free_decoder(decoder_t *decoder);
void decoder_pre_forward(decoder_t *decoder);
void decoder_forward(decoder_t *decoder, float *last_input, float *last_output);
void decoder_set_debug_weight(decoder_t *decoder);

struct GPT2Model_t {
    int num_decoders;
    int d_hidden;
    int d_head;
    int d_ffn;

    float *wte;
    float *wpe;
    float *W_ln_f;
    float *B_ln_f;

    decoder_t **decoders;

    int _num_inferenced_token;
    
    float *_buf_rawinput;
    float *_buf_position_onehot;
    float *_buf_input;
    float *_buf_ln_f_temp;
    float *_buf_ln_f;
    float *_buf_output;
    float *_buf_swap;

    /* DEBUG */
    #ifdef DEBUG
    float _debug_eta_total;
    float _debug_eta_last;
    #endif
};

GPT2Model_t *new_GPT2Model(int num_decoders, int d_hidden, int d_head, int d_ffn);
void free_GPT2Model(GPT2Model_t *model);
void GPT2Model_pre_forward(GPT2Model_t *model);
int GPT2Model_forward(GPT2Model_t *gpt2model, int input_idx, float *output);
void GPT2Model_load(GPT2Model_t *gpt2model, char *weight_path);

int vector_argmax(int m, float *x, int incx);
void vector_onehot(float* dest, int n, int idx);
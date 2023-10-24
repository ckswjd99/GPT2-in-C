#include "gpt2.h"

int main(int argc, char *argv[]) {
    /* ARGPARSE */
    if (argc < 2) {
        printf("Usage: %s [length]\n", argv[0]);
        exit(1);
    }

    int gen_length = atoi(argv[1]);
    if (gen_length < 1) gen_length = 1;
    if (gen_length > 512) gen_length = 512;

    /* DEBUG */
    #ifdef DEBUG
    printf("RUNNING IN DEBUG MODE\n");
    #endif

    openblas_set_num_threads(1);

    int argmax;
    float output[GPT2_D_TOKENS];

    argmax = 29193;

    struct timeval start_time, end_time;

    GPT2Model_t *gpt2_model = new_GPT2Model(GPT2_NUM_DECODERS, GPT2_D_HIDDEN, GPT2_D_HEAD, GPT2_D_FFN);

    decoder_set_debug_weight(gpt2_model->decoders[0]);

    GPT2Model_load(gpt2_model, "./model/GPT2-124M.mymodel");

    gettimeofday(&start_time, NULL);
    for (int i=0; i<gen_length; i++) {
        GPT2Model_forward(gpt2_model, argmax, output);
        argmax = vector_argmax(GPT2_D_HIDDEN, output, 1);
        printf("next token: %d\n", argmax);
    }
    gettimeofday(&end_time, NULL);

    printf("Inferenced with GPT2Model\n");
    printf("Total ETA: %f (ms)\n", ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3);
    #ifdef DEBUG
    unsigned long long total_mac = 0;
    for (int i=0; i<GPT2_NUM_DECODERS; i++) {
        total_mac += gpt2_model->decoders[i]->_debug_flops_total;
    }
    printf("Total FLOPS: %llu\n", total_mac);
    #endif

    return 0;
}

void test_values() {
    
}

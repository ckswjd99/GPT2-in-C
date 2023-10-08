#include "gpt2.h"

int main() {
    float input[GPT2_D_TOKENS];
    float output[GPT2_D_TOKENS];

    bzero(input, sizeof(float) * GPT2_D_TOKENS);

    input[29193] = 1;
    vector_onehot(input, GPT2_D_TOKENS, 29193);

    /*
    decoder_t **model = (decoder_t **)malloc(sizeof(decoder_t *) * 12);
    for (int i=0; i<12; i++) {
        model[i] = new_decoder(768, 12, 768 * 4);
        decoder_set_debug_weight(model[i]);
        decoder_pre_forward(model[i]);
    }

    struct timeval stamps[256 + 1];

    gettimeofday(&stamps[0], NULL);

    for (int i=0; i<256; i++) {
        for (int j=0; j<12; j++) {
            decoder_forward(model[j], input, output);
            memcpy(input, output, 768);
        }

        gettimeofday(stamps + i + 1, NULL);
    }

    for (int i=0; i<256; i++) {
        printf("%d, %.6f\n", i, ((stamps[i+1].tv_sec * 1e6 + stamps[i+1].tv_usec) - (stamps[i].tv_sec * 1e6 + stamps[i].tv_usec)) / 1e3);
    }

    printf("Total ETA: %f (ms)\n", ((stamps[256].tv_sec * 1e6 + stamps[256].tv_usec) - (stamps[0].tv_sec * 1e6 + stamps[0].tv_usec)) / 1e3);

    */

    struct timeval start_time, end_time;

    GPT2Model_t *gpt2_model = new_GPT2Model(GPT2_NUM_DECODERS, GPT2_D_HIDDEN, GPT2_D_HEAD, GPT2_D_FFN);

    decoder_set_debug_weight(gpt2_model->decoders[0]);

    GPT2Model_load(gpt2_model, "./model/GPT2-124M.mymodel");

    gettimeofday(&start_time, NULL);
    for (int i=0; i<64; i++) {
        GPT2Model_forward(gpt2_model, input, output);
        int argmax = vector_argmax(GPT2_D_HIDDEN, output, 1);
        printf("next token: %d\n", argmax);
        vector_onehot(input, GPT2_D_TOKENS, argmax);
    }
    gettimeofday(&end_time, NULL);

    printf("Inferenced with GPT2Model\n");
    printf("Total ETA: %f (ms)\n", ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3);

    return 0;
}

void test_values() {
    
}

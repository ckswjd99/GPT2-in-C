#include "gpt2.h"

decoder_t *new_decoder(int d_hidden, int d_head, int d_ffn) {
    decoder_t *decoder = (decoder_t *)malloc(sizeof(decoder_t));
    
    /* ALLOC MEMS */
    float *memories = (float *)malloc(sizeof(float) * (
        // UTILS
        d_hidden * d_hidden

        // WEIGHTS
        + d_hidden * 2
        + d_hidden * d_hidden * 4 + d_hidden * 4
        + d_hidden * 2
        + d_hidden * d_ffn * 2 + d_ffn + d_hidden

        // BUFFERS
        + d_hidden * 8 + DECODER_NUM_TOKEN_INIT + d_ffn

        // FEATURES
        + d_hidden + d_hidden * DECODER_NUM_TOKEN_INIT * 2  
    ));
    float *mem_last = memories;

    decoder->_mem_start = memories;
    decoder->_num_inferenced_token = 0;

    decoder->d_hidden = d_hidden;
    decoder->d_head = d_head;
    decoder->d_ffn = d_ffn;

    /* DIST MEMES */
    // UTILS
    decoder->ones = mem_last;               mem_last += d_hidden * d_hidden;

    // WEIGHTS
    decoder->W_ln1 = mem_last;              mem_last += d_hidden;
    decoder->B_ln1 = mem_last;              mem_last += d_hidden;
    decoder->W_Q = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->W_K = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->W_V = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->W_O = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->W_ln2 = mem_last;              mem_last += d_hidden;
    decoder->W_ffn1 = mem_last;             mem_last += d_hidden * d_ffn;
    decoder->W_ffn2 = mem_last;             mem_last += d_ffn * d_hidden;
    
    decoder->B_Q = mem_last;                mem_last += d_hidden;
    decoder->B_K = mem_last;                mem_last += d_hidden;
    decoder->B_V = mem_last;                mem_last += d_hidden;
    decoder->B_O = mem_last;                mem_last += d_hidden;
    decoder->B_ln2 = mem_last;              mem_last += d_hidden;
    decoder->B_ffn1 = mem_last;             mem_last += d_ffn;
    decoder->B_ffn2 = mem_last;             mem_last += d_hidden;

    // BUFFERS
    decoder->_buf_embedded = mem_last;      mem_last += d_hidden;
    decoder->_buf_ln1 = mem_last;           mem_last += d_hidden;
    decoder->_buf_ln1_temp = mem_last;      mem_last += d_hidden;
    decoder->_buf_q = mem_last;             mem_last += d_hidden;
    decoder->_buf_attn = mem_last;          mem_last += DECODER_NUM_TOKEN_INIT;
    decoder->_buf_sha = mem_last;           mem_last += d_hidden;
    decoder->_buf_o = mem_last;             mem_last += d_hidden;
    decoder->_buf_ln2 = mem_last;           mem_last += d_hidden;
    decoder->_buf_ln2_temp = mem_last;       mem_last += d_hidden;
    decoder->_buf_ffn1 = mem_last;          mem_last += d_ffn;
    decoder->_buf_ffn2 = mem_last;          mem_last += d_hidden;

    // FEATURES
    decoder->Q = mem_last;                  mem_last += d_hidden;
    decoder->K = mem_last;                  mem_last += d_hidden * DECODER_NUM_TOKEN_INIT;
    decoder->V = mem_last;                  mem_last += d_hidden * DECODER_NUM_TOKEN_INIT;

    /* INIT MEMS */
    for (int i=0; i<decoder->d_hidden*decoder->d_hidden; i++) {
        decoder->ones[i] = 1.0;
    }

    /* INIT DEBUG */
    #ifdef DEBUG
    decoder->_debug_flops_total = 0;
    decoder->_debug_flops_last = 0;
    decoder->_debug_eta_total = 0;
    decoder->_debug_eta_last = 0;
    #endif

    return decoder;
}

void free_decoder(decoder_t *decoder) {
    free(decoder->_mem_start);
    free(decoder);
}

void decoder_forward(decoder_t *decoder, float *last_input, float *last_output) {

    /* DEBUG START */
    #ifdef DEBUG
    struct timeval start_time, end_time;
    float eta;
    unsigned long long flops;
    gettimeofday(&start_time, NULL);
    #endif

    // For convenience
    int d_hidden = decoder->d_hidden;
    int d_head = decoder->d_head;
    int d_ffn = decoder->d_ffn;
    int d_hid_per_head = d_hidden / d_head;

    int num_inferenced = decoder->_num_inferenced_token;

    float *W_Q = decoder->W_Q, *W_K = decoder->W_K, *W_V = decoder->W_V, *W_O = decoder->W_O;
    float *B_Q = decoder->B_Q, *B_K = decoder->B_K, *B_V = decoder->B_V, *B_O = decoder->B_O;

    float *W_ffn1 = decoder->W_ffn1, *W_ffn2 = decoder->W_ffn2;
    float *B_ffn1 = decoder->B_ffn1, *B_ffn2 = decoder->B_ffn2;
    
    float *W_ln1 = decoder->W_ln1, *W_ln2 = decoder->W_ln2;
    float *B_ln1 = decoder->B_ln1, *B_ln2 = decoder->B_ln2;

    float *Q = decoder->Q, *K = decoder->K, *V = decoder->V;

    // Residual Connection - Fanout
    memcpy(decoder->_buf_embedded, last_input, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->_buf_ln1, last_input, sizeof(float) * decoder->d_hidden);

    // Layer Normalization
    layer_normalize(d_hidden, decoder->_buf_ln1, W_ln1, B_ln1, decoder->_buf_ln1_temp, decoder->ones);
    
    // Compute QKV
    layer_linear(d_hidden, d_hidden, decoder->_buf_ln1, W_Q, B_Q, Q);
    layer_linear(d_hidden, d_hidden, decoder->_buf_ln1, W_K, B_K, K + d_hidden * num_inferenced);
    layer_linear(d_hidden, d_hidden, decoder->_buf_ln1, W_V, B_V, V + d_hidden * num_inferenced);

    // Compute MHA
    for (int i=0; i<d_head; i++) {
        // Attention
        cblas_sgemv(CblasColMajor, CblasTrans, d_hid_per_head, num_inferenced+1, 1.0/sqrtf(d_hid_per_head), K + i * d_hid_per_head, d_hidden, Q + i * d_hid_per_head, 1, 0.0, decoder->_buf_attn, 1);
        
        // Softmax
        layer_softmax(num_inferenced+1, decoder->_buf_attn);

        // SHA
        cblas_sgemv(CblasColMajor, CblasNoTrans, d_hid_per_head, num_inferenced+1, 1.0, V + i * d_hid_per_head, d_hidden, decoder->_buf_attn, 1, 0.0, decoder->_buf_sha + i * d_hid_per_head, 1);
    }

    // MHA
    layer_linear(d_hidden, d_hidden, decoder->_buf_sha, W_O, B_O, decoder->_buf_o);

    // Residual Connection - Sum and Fanout
    cblas_saxpy(d_hidden, 1.0, decoder->_buf_o, 1, decoder->_buf_embedded, 1);
    memcpy(decoder->_buf_ln2, decoder->_buf_embedded, sizeof(float) * d_hidden);

    // Layer Norm
    layer_normalize(d_hidden, decoder->_buf_ln2, W_ln2, B_ln2, decoder->_buf_ln2_temp, decoder->ones);
    
    // FFN1
    layer_linear(d_ffn, d_hidden, decoder->_buf_ln2, W_ffn1, B_ffn1, decoder->_buf_ffn1);

    // Activation: GeLU
    layer_GeLU(d_ffn, decoder->_buf_ffn1);

    // FFN2
    layer_linear(d_hidden, d_ffn, decoder->_buf_ffn1, W_ffn2, B_ffn2, decoder->_buf_ffn2);

    // Residual connection - Sum
    cblas_saxpy(d_hidden, 1.0, decoder->_buf_ffn2, 1, decoder->_buf_embedded, 1);

    // Copy output
    memcpy(last_output, decoder->_buf_embedded, sizeof(float) * d_hidden);

    // For next inference
    decoder->_num_inferenced_token++;

    /* DEBUG FINISH */
    #ifdef DEBUG
    gettimeofday(&end_time, NULL);
    eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;
    flops = (
        d_hidden * d_hidden * 4
        + d_hid_per_head * num_inferenced * 2 * d_head
        + d_hidden * d_hidden * 2
        + d_ffn * d_hidden * 2
    ) * 2;

    decoder->_debug_flops_total += flops;
    decoder->_debug_flops_last = flops;
    decoder->_debug_eta_total += eta;
    decoder->_debug_eta_last = eta;
    #endif
}

GPT2Model_t *new_GPT2Model(int num_decoders, int d_hidden, int d_head, int d_ffn) {
    GPT2Model_t *model = (GPT2Model_t *)malloc(sizeof(GPT2Model_t));

    model->num_decoders = num_decoders;
    model->d_hidden = d_hidden;
    model->d_head = d_head;
    model->d_ffn = d_ffn;

    model->wte = (float *)malloc(sizeof(float) * GPT2_D_VOCABS * GPT2_D_HIDDEN);
    model->wpe = (float *)malloc(sizeof(float) * GPT2_MAX_TOKEN * GPT2_D_HIDDEN);
    model->W_ln_f = (float *)malloc(sizeof(float) * GPT2_D_HIDDEN);
    model->B_ln_f = (float *)malloc(sizeof(float) * GPT2_D_HIDDEN);

    model->decoders = (decoder_t **)malloc(sizeof(decoder_t *) * model->num_decoders);
    for(int i=0; i<model->num_decoders; i++) {
        model->decoders[i] = new_decoder(model->d_hidden, model->d_head, model->d_ffn);
    }

    model->_num_inferenced_token = 0;

    model->_buf_rawinput = (float *)malloc(sizeof(float) * GPT2_D_VOCABS);

    model->_buf_input = (float *)malloc(sizeof(float) * model->d_hidden);
    model->_buf_ln_f = (float *)malloc(sizeof(float) * model->d_hidden);
    model->_buf_output = (float *)malloc(sizeof(float) * model->d_hidden);

    return model;
}

void free_GPT2Model(GPT2Model_t *model) {
    free(model->wte);
    free(model->wpe);
    free(model->W_ln_f);
    free(model->B_ln_f);

    for (int i=0; i<model->num_decoders; i++) {
        free_decoder(model->decoders[i]);
    }
    free(model->decoders);

    free(model->_buf_rawinput);

    free(model->_buf_input);
    free(model->_buf_ln_f);
    free(model->_buf_output);

    free(model);
}

void GPT2Model_sample(
    GPT2Model_t *model, 
    char *text, int length, int num_samples, int batch_size, 
    float temperature, int top_k, int num_beam
) {
    float *input_embed = (float *)malloc(sizeof(float) * model->d_hidden);
    float *output_embed = (float *)malloc(sizeof(float) * model->d_hidden);
    float *logits = (float *)malloc(sizeof(float) * GPT2_D_VOCABS);

    int argmax = 29193;

    for (int i=0; i<length; i++) {
        GPT2Model_encode(model, argmax, input_embed);
        GPT2Model_forward(model, input_embed, output_embed);
        GPT2Model_decode(model, output_embed, logits);
        argmax = vector_argmax(GPT2_D_VOCABS, logits, 1);
        printf("next token: %d\n", argmax);
    }
}

void GPT2Model_forward(GPT2Model_t *model, float *input_embed, float *output_embed) {
    // Input: int, index of previous token
    // Output: float[GPT2_D_TOKEN], logits of next token

    /* DEBUG START */
    #ifdef DEBUG
    struct timeval start_time, end_time;
    float eta;
    gettimeofday(&start_time, NULL);
    #endif

    int d_hidden = model->d_hidden;

    memcpy(model->_buf_input, input_embed, sizeof(float) * d_hidden);

    cblas_saxpy(d_hidden, 1.0, &model->wpe[d_hidden * model->_num_inferenced_token], 1, model->_buf_input, 1);

    for (int i=0; i<model->num_decoders; i++) {
        decoder_forward(model->decoders[i], model->_buf_input, model->_buf_output);
        model->_buf_swap = model->_buf_input;
        model->_buf_input = model->_buf_output;
        model->_buf_output = model->_buf_swap;
    }

    model->_buf_swap = model->_buf_input;
    model->_buf_input = model->_buf_output;
    model->_buf_output = model->_buf_swap;

    // Layer Normalization (final)
    layer_normalize(d_hidden, model->_buf_output, model->W_ln_f, model->B_ln_f, model->_buf_ln_f, model->decoders[0]->ones);

    // Output
    memcpy(output_embed, model->_buf_output, sizeof(float) * d_hidden);

    model->_num_inferenced_token++;

    /* DEBUG FINISH */
    #ifdef DEBUG
    gettimeofday(&end_time, NULL);
    eta = ((end_time.tv_sec * 1e6 + end_time.tv_usec) - (start_time.tv_sec * 1e6 + start_time.tv_usec)) / 1e3;

    model->_debug_eta_total += eta;
    model->_debug_eta_last = eta;
    #endif
}

void GPT2Model_encode(GPT2Model_t *model, int vocab_idx, float *embedded) {
    memcpy(embedded, &model->wte[model->d_hidden * vocab_idx], sizeof(float) * model->d_hidden);
}

void GPT2Model_decode(GPT2Model_t *model, float *embedded, float *logits) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, GPT2_D_VOCABS, model->d_hidden, 1.0, model->wte, model->d_hidden, embedded, 1, 0.0, logits, 1);
}

float *find_tensor_target_p(GPT2Model_t *model, char *tensor_name) {
    float *tensor_target_p;
    int dblock_idx;
    char dblock_subname[LOAD_BUFFER_SIZE] = {0,};

    if (strcmp(tensor_name, "wte") == 0) {
        tensor_target_p = model->wte;
    }
    else if (strcmp(tensor_name, "wpe") == 0) {
        tensor_target_p = model->wpe;
    }
    else if (strcmp(tensor_name, "ln_f_w") == 0) {
        tensor_target_p = model->W_ln_f;
    }
    else if (strcmp(tensor_name, "ln_f_b") == 0) {
        tensor_target_p = model->B_ln_f;
    }
    else if (strncmp(tensor_name, "dblock_", 7) == 0) {
        sscanf(
            tensor_name, "dblock_%d.%s\n", &dblock_idx, dblock_subname
        );
        
        if (strcmp(dblock_subname, "ln1_w") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_ln1;
        
        else if (strcmp(dblock_subname, "ln1_b") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_ln1;
        
        else if (strcmp(dblock_subname, "attn_wq") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_Q;
        
        else if (strcmp(dblock_subname, "attn_wk") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_K;
        
        else if (strcmp(dblock_subname, "attn_wv") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_V;
        
        else if (strcmp(dblock_subname, "attn_wo") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_O;
        
        else if (strcmp(dblock_subname, "attn_bq") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_Q;
        
        else if (strcmp(dblock_subname, "attn_bk") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_K;
        
        else if (strcmp(dblock_subname, "attn_bv") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_V;
        
        else if (strcmp(dblock_subname, "attn_bo") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_O;
        
        else if (strcmp(dblock_subname, "ln2_w") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_ln2;
        
        else if (strcmp(dblock_subname, "ln2_b") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_ln2;
        
        else if (strcmp(dblock_subname, "ffn1_w") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_ffn1;
        
        else if (strcmp(dblock_subname, "ffn1_b") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_ffn1;
        
        else if (strcmp(dblock_subname, "ffn2_w") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->W_ffn2;
        
        else if (strcmp(dblock_subname, "ffn2_b") == 0) 
            tensor_target_p = model->decoders[dblock_idx]->B_ffn2;
        
        else {
            fprintf(stderr, "Unknown tensor name!\n");
            exit(1);
        }
    }
    else {
        fprintf(stderr, "Unknown tensor name!\n");
        exit(1);
    }

    return tensor_target_p;
}

void GPT2Model_load(GPT2Model_t *model, char *weight_path) {
    fprintf(stdout, "Loading GPT2 weights from %s\n", weight_path);

    FILE *fp;
    char read_buffer[LOAD_BUFFER_SIZE] = {0,};
    char temp_buffer[LOAD_BUFFER_SIZE] = {0,};
    
    int num_tensor;
    char tensor_name[LOAD_BUFFER_SIZE] = {0,};
    int tensor_size;
    float *tensor_target_p;

    fp = fopen(weight_path, "r");
    if (!fp) {
        fprintf(stderr, "Weight file not exists!\n");
        exit(0);
    }

    fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
    sscanf(read_buffer, "NUM_TENSOR:%d\n", &num_tensor);
    printf("  Number of tensors: %d\n", num_tensor);

    for (int i=0; i<num_tensor; i++) {
        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "TENSOR:%s\n", tensor_name);
        
        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_SIZE:%d\n", &tensor_size);
        
        // printf("  Loading tensor %s(%d)\n", tensor_name, tensor_size);

        tensor_target_p = find_tensor_target_p(model, tensor_name);

        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_%s\n", temp_buffer);
        if (strncmp(temp_buffer, "START", 5) != 0) {
            fprintf(stderr, "  DATA_START field not exists!\n");
            fprintf(stderr, "    read input: %s\n", temp_buffer);
            exit(1);
        }

        fread((void *)tensor_target_p, tensor_size, 1, fp);

        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_%s\n", temp_buffer);
        if (strncmp(temp_buffer, "END", 3)) {
            fprintf(stderr, "  DATA_END field not exists!\n");
            fprintf(stderr, "    read input: %s\n", temp_buffer);
            exit(1);
        }

        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "TENSOR_%s\n", temp_buffer);
        if (strncmp(temp_buffer, "END", 3)) {
            fprintf(stderr, "  TENSOR_END field not exists!\n");
            fprintf(stderr, "    read input: %s\n", temp_buffer);
            exit(1);
        }

    }

    fprintf(stdout, "Finished loading weights!\n");

    fclose(fp);

    return;
}

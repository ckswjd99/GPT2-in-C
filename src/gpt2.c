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
    decoder->B_Q = mem_last;                mem_last += d_hidden;
    decoder->W_K = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->B_K = mem_last;                mem_last += d_hidden;
    decoder->W_V = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->B_V = mem_last;                mem_last += d_hidden;
    decoder->W_O = mem_last;                mem_last += d_hidden * d_hidden;
    decoder->B_O = mem_last;                mem_last += d_hidden;
    
    decoder->W_ln2 = mem_last;              mem_last += d_hidden;
    decoder->B_ln2 = mem_last;              mem_last += d_hidden;

    decoder->W_ffn1 = mem_last;             mem_last += d_hidden * d_ffn;
    decoder->B_ffn1 = mem_last;             mem_last += d_ffn;
    decoder->W_ffn2 = mem_last;             mem_last += d_ffn * d_hidden;
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

    decoder_pre_forward(decoder);

    return decoder;
}

void free_decoder(decoder_t *decoder) {
    free(decoder->_mem_start);
    free(decoder);
}

void decoder_pre_forward(decoder_t *decoder) {
    memcpy(decoder->_buf_ln1, decoder->B_ln1, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->Q, decoder->B_Q, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->K + decoder->_num_inferenced_token * decoder->d_hidden, decoder->B_K, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->V + decoder->_num_inferenced_token * decoder->d_hidden, decoder->B_V, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->_buf_o, decoder->B_O, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->_buf_ln2, decoder->B_ln2, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->_buf_ffn1, decoder->B_ffn1, sizeof(float) * decoder->d_ffn);
    memcpy(decoder->_buf_ffn2, decoder->B_ffn2, sizeof(float) * decoder->d_hidden);
}

void decoder_forward(decoder_t *decoder, float *last_input, float *last_output) {

    // For convenience
    int d_hidden = decoder->d_hidden;
    int d_head = decoder->d_head;
    int d_ffn = decoder->d_ffn;
    int d_hid_per_head = d_hidden / d_head;

    int num_inferenced = decoder->_num_inferenced_token;

    float *W_Q = decoder->W_Q;
    float *W_K = decoder->W_K;
    float *W_V = decoder->W_V;
    float *W_O = decoder->W_O;
    float *W_ffn1 = decoder->W_ffn1;
    float *W_ffn2 = decoder->W_ffn2;

    float *Q = decoder->Q;
    float *K = decoder->K;
    float *V = decoder->V;

    // Ready for computation
    memcpy(decoder->_buf_embedded, last_input, sizeof(float) * decoder->d_hidden);
    memcpy(decoder->_buf_ln1, last_input, sizeof(float) * decoder->d_hidden);

    // Layer Normalization
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_hidden, 1.0/768, decoder->ones, d_hidden, decoder->_buf_embedded, 1, 0.0, decoder->_buf_ln1_temp, 1);   // get average vector
    cblas_saxpy(d_hidden, -1.0, decoder->_buf_ln1_temp, 1, decoder->_buf_ln1, 1);                                                                           // x <- x-u
    float ln1_std = cblas_snrm2(d_hidden, decoder->_buf_ln1, 1) / sqrtf(decoder->d_hidden);                                                                                    // std <- s(x)
    memcpy(decoder->_buf_ln1_temp, decoder->B_ln1, sizeof(float) * d_hidden);
    cblas_ssbmv(CblasRowMajor, CblasUpper, d_hidden, 0, 1.0/ln1_std, decoder->W_ln1, 1, decoder->_buf_ln1, 1, 1.0, decoder->_buf_ln1_temp, 1);                  // ln1_result <- ln1 .* x / std
    memcpy(decoder->_buf_ln1, decoder->_buf_ln1_temp, sizeof(float) * d_hidden);
    
    // Compute Q
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_hidden, 1.0, W_Q, d_hidden, decoder->_buf_ln1, 1, 1.0, Q, 1);
    
    // Compute K, V
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_hidden, 1.0, W_K, d_hidden, decoder->_buf_ln1, 1, 1.0, K + d_hidden * num_inferenced, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_hidden, 1.0, W_V, d_hidden, decoder->_buf_ln1, 1, 1.0, V + d_hidden * num_inferenced, 1);

    // Compute MHA
    for (int i=0; i<d_head; i++) {
        // Attention
        cblas_sgemv(CblasColMajor, CblasTrans, d_hid_per_head, num_inferenced+1, 1.0/sqrtf(d_hid_per_head), K + i * d_hid_per_head, d_hidden, Q, 1, 0.0, decoder->_buf_attn, 1);
        
        // Softmax
        float sm_max = 0;
        for (int i=0; i<num_inferenced+1; i++) sm_max = (sm_max > decoder->_buf_attn[i] ? sm_max : decoder->_buf_attn[i]);
        float sm_sum = 0;
        for (int i=0; i<num_inferenced+1; i++) sm_sum += exp2f(decoder->_buf_attn[i] - sm_max);
        
        for (int i=0; i<num_inferenced+1; i++) decoder->_buf_attn[i] = exp2f(decoder->_buf_attn[i] - sm_max) / sm_sum;

        // TODO: SIMD this.

        // SHA
        cblas_sgemv(CblasColMajor, CblasNoTrans, d_hid_per_head, num_inferenced+1, 1.0, V + i * d_hid_per_head, d_hidden, decoder->_buf_attn, 1, 1.0, decoder->_buf_sha + i * d_hid_per_head, 1);
    }

    // MHA
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_hidden, 1.0, W_O, d_hidden, decoder->_buf_sha, 1, 1.0, decoder->_buf_o, 1);

    // Residual Connection - raw input
    cblas_saxpy(d_hidden, 1.0, decoder->_buf_o, 1, decoder->_buf_embedded, 1);
    memcpy(decoder->_buf_ln2, decoder->_buf_embedded, sizeof(float) * d_hidden);

    // Layer Norm
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_hidden, 1.0/768, decoder->ones, d_hidden, decoder->_buf_embedded, 1, 0.0, decoder->_buf_ln2_temp, 1);   // get average vector
    cblas_saxpy(d_hidden, -1.0, decoder->_buf_ln2_temp, 1, decoder->_buf_ln2, 1);                                                                           // x <- x-u
    float ln2_std = cblas_snrm2(d_hidden, decoder->_buf_ln2, 1) / sqrtf(d_hidden);           
    memcpy(decoder->_buf_ln2_temp, decoder->B_ln2, sizeof(float) * d_hidden);                                                                         // std <- s(x)
    cblas_ssbmv(CblasRowMajor, CblasUpper, d_hidden, 0, 1.0/ln2_std, decoder->W_ln2, 1, decoder->_buf_ln2, 1, 1.0, decoder->_buf_ln2_temp, 1);                  // ln1_result <- ln1 .* x / std
    memcpy(decoder->_buf_ln2, decoder->_buf_ln2_temp, sizeof(float) * d_hidden);

    // FFN1
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_ffn, d_hidden, 1.0, W_ffn1, d_hidden, decoder->_buf_ln2, 1, 1.0, decoder->_buf_ffn1, 1);

    // Activation: GeLU
    // TODO: SIMD this.
    for (int i=0; i<d_ffn; i++) decoder->_buf_ffn1[i] = 0.5 * decoder->_buf_ffn1[i] * (1 + tanh(sqrt(2.0 / M_PI) * (decoder->_buf_ffn1[i] + 0.044715 * powf(decoder->_buf_ffn1[i], 3))));

    // FFN2
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_hidden, d_ffn, 1.0, W_ffn2, d_ffn, decoder->_buf_ffn1, 1, 1.0, decoder->_buf_ffn2, 1);

    // Residual connection
    cblas_saxpby(d_hidden, 1.0, decoder->_buf_ffn2, 1, 1.0, decoder->_buf_embedded, 1);

    // Copy output
    memcpy(last_output, decoder->_buf_embedded, sizeof(float) * d_hidden);

    // For next inference
    decoder->_num_inferenced_token++;
    decoder_pre_forward(decoder);
}

void decoder_set_debug_weight(decoder_t *decoder) {
    int d_hidden = decoder->d_hidden;
    int d_head = decoder->d_head;
    int d_ffn = decoder->d_ffn;
    int d_hid_per_head = d_hidden / d_head;

    /* LN1 */
    decoder->W_ln1;
    for (int i=0; i<GPT2_D_HIDDEN; i++) {
        decoder->W_ln1[i] = 78.3836385;
        decoder->B_ln1[i] = 384.5 - i;
    }
    
    /* Attention */
    for (int i=0; i<GPT2_D_HIDDEN; i++) {       // select row
        for (int j=0; j<GPT2_D_HIDDEN; j++) {   // select col
            decoder->W_Q[i * GPT2_D_HIDDEN + j] = (float)i/GPT2_D_HIDDEN;
            decoder->W_K[i * GPT2_D_HIDDEN + j] = (float)i/GPT2_D_HIDDEN;
            decoder->W_V[i * GPT2_D_HIDDEN + j] = (float)i/GPT2_D_HIDDEN;
        }
    }
}

GPT2Model_t *new_GPT2Model(int num_decoders, int d_hidden, int d_head, int d_ffn) {
    GPT2Model_t *model = (GPT2Model_t *)malloc(sizeof(GPT2Model_t));

    model->num_decoders = num_decoders;
    model->d_hidden = d_hidden;
    model->d_head = d_head;
    model->d_ffn = d_ffn;

    model->wte = (float *)malloc(sizeof(float) * GPT2_D_TOKENS * GPT2_D_HIDDEN);
    model->wpe = (float *)malloc(sizeof(float) * GPT2_MAX_TOKEN * GPT2_D_HIDDEN);
    model->W_ln_f = (float *)malloc(sizeof(float) * GPT2_D_HIDDEN);
    model->B_ln_f = (float *)malloc(sizeof(float) * GPT2_D_HIDDEN);

    model->decoders = (decoder_t **)malloc(sizeof(decoder_t *) * model->num_decoders);
    for(int i=0; i<model->num_decoders; i++) {
        model->decoders[i] = new_decoder(model->d_hidden, model->d_head, model->d_ffn);
    }

    model->_num_inferenced_token = 0;

    model->_buf_rawinput = (float *)malloc(sizeof(float) * GPT2_D_TOKENS);
    model->_buf_position_onehot = (float *)malloc(sizeof(float) * GPT2_MAX_TOKEN);

    model->_buf_input = (float *)malloc(sizeof(float) * model->d_hidden);
    model->_buf_output = (float *)malloc(sizeof(float) * model->d_hidden);

    GPT2Model_pre_forward(model);

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
    free(model->_buf_position_onehot);

    free(model->_buf_input);
    free(model->_buf_output);

    free(model);
}

void GPT2Model_pre_forward(GPT2Model_t *model) {
    bzero(model->_buf_position_onehot, sizeof(float) * GPT2_MAX_TOKEN);

    // Decoders pre-forward
    for(int i=0; i<model->num_decoders; i++) {
        decoder_pre_forward(model->decoders[i]);
    }
}

int GPT2Model_forward(GPT2Model_t *model, float *input, float *output) {
    // Convert one-hot input to embedded token
    // TODO: take simpler approach
    //   wte and wpe are token-wisely storaged,
    //   so we don't need to get it through sgemv.
    memcpy(model->_buf_rawinput, input, sizeof(float) * GPT2_D_TOKENS);
    
    cblas_sgemv(
        CblasRowMajor, CblasTrans, 
        GPT2_D_TOKENS, model->d_hidden,
        1.0, model->wte, model->d_hidden,
        model->_buf_rawinput, 1,
        0.0, model->_buf_input, 1
    );

    model->_buf_position_onehot[model->_num_inferenced_token] = 1.0;

    cblas_sgemv(
        CblasRowMajor, CblasTrans,
        GPT2_MAX_TOKEN, model->d_hidden,
        1.0, model->wpe, model->d_hidden,
        model->_buf_position_onehot, 1,
        1.0, model->_buf_input, 1
    );

    for (int i=0; i<model->num_decoders; i++) {
        decoder_forward(model->decoders[i], model->_buf_input, model->_buf_output);
        model->_buf_swap = model->_buf_input;
        model->_buf_input = model->_buf_output;
        model->_buf_output = model->_buf_swap;
    }
    memcpy(output, model->_buf_input, sizeof(float) * model->d_hidden);

    // TODO: implement ln_f

    model->_num_inferenced_token++;
    GPT2Model_pre_forward(model);
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
        sscanf(read_buffer, "TENSOR:%s\n", &tensor_name);
        
        fgets(read_buffer, LOAD_BUFFER_SIZE, fp);
        sscanf(read_buffer, "DATA_SIZE:%d\n", &tensor_size);
        
        printf("  Loading tensor %s(%d)\n", tensor_name, tensor_size);

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

    GPT2Model_pre_forward(model);
    return;
}
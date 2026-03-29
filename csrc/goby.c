/*
 * goby.c — GobyLLM native inference engine
 *
 * Single-file C runtime for GobyLLM. No dependencies beyond libc and math.
 * Compiles on x86, ARM (RPi), Apple Silicon.
 *
 * Features:
 *   - mmap model loading (instant startup, OS manages memory paging)
 *   - KV cache (only processes new token each step)
 *   - Early exit (router-gated layer skipping)
 *   - NEON SIMD on ARM (auto-detected)
 *
 * Build:
 *   cc -O3 -o goby goby.c -lm                      # generic
 *   cc -O3 -march=native -o goby goby.c -lm         # native SIMD
 *   cc -O3 -mfpu=neon-fp-armv8 -o goby goby.c -lm   # RPi 4
 *
 * Usage:
 *   ./goby goby.bin -p "Turn on the lights"
 *   ./goby goby.bin -i                               # interactive
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 *  DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int vocab_size, d_model, n_layers, n_heads, n_kv_heads;
    int ffn_hidden, max_seq_len;
    int early_exit, min_exit_layer;
    float exit_threshold;
    /* derived */
    int head_dim, kv_dim, n_rep;
} Config;

typedef struct {
    float *norm_w;
    float *wq, *wk, *wv, *wo;
    float *ffn_gate, *ffn_up, *ffn_down;
    float *router_w;
    float *router_b;
} Layer;

typedef struct {
    Config cfg;
    float *tok_emb;      /* [vocab_size, d_model] — also used as lm_head */
    Layer *layers;
    float *final_norm_w;  /* [d_model] */
    float *rope_cos;      /* [max_seq_len, head_dim/2] */
    float *rope_sin;
} Model;

typedef struct {
    char **tokens;        /* id → byte string */
    int *token_lens;      /* id → length */
    int vocab_size;
    int *merges;          /* [n_merges * 2] pairs of token IDs */
    int n_merges;
    int byte_to_token[256];
} Tokenizer;

typedef struct {
    float *x, *xb, *xb2;        /* [d_model] buffers */
    float *q;                     /* [n_heads * head_dim] */
    float *att;                   /* [n_heads * max_seq_len] */
    float *ffn_buf1, *ffn_buf2;  /* [ffn_hidden] */
    float *logits;                /* [vocab_size] */
    float *key_cache;             /* [n_layers * max_seq_len * kv_dim] */
    float *val_cache;
} RunState;

/* ═══════════════════════════════════════════════════════════════════════════
 *  MATH OPS
 * ═══════════════════════════════════════════════════════════════════════════ */

static void rmsnorm(float *out, const float *x, const float *w, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * w[i];
}

static void matmul(float *out, const float *x, const float *w, int n, int d) {
    /* out[i] = dot(w[i], x) for i in 0..n-1.  w is [n, d], x is [d]. */
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        const float *wi = w + i * d;
#ifdef __ARM_NEON
        int j = 0;
        float32x4_t sv = vdupq_n_f32(0.0f);
        for (; j + 4 <= d; j += 4) {
            sv = vmlaq_f32(sv, vld1q_f32(x + j), vld1q_f32(wi + j));
        }
        sum = vaddvq_f32(sv);
        for (; j < d; j++) sum += x[j] * wi[j];
#else
        for (int j = 0; j < d; j++) sum += x[j] * wi[j];
#endif
        out[i] = sum;
    }
}

static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MODEL LOADING (mmap)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void *file_data;
static size_t file_size;

static void *read_ptr(void **cursor, size_t bytes) {
    void *p = *cursor;
    *cursor = (char *)*cursor + bytes;
    return p;
}

static int read_int(void **c) { return *(int *)read_ptr(c, 4); }
static unsigned read_uint(void **c) { return *(unsigned *)read_ptr(c, 4); }
static float read_float(void **c) { return *(float *)read_ptr(c, 4); }
static unsigned short read_ushort(void **c) { return *(unsigned short *)read_ptr(c, 2); }

static float *read_floats(void **c, int count) {
    return (float *)read_ptr(c, count * sizeof(float));
}

static int load_model(const char *path, Model *m, Tokenizer *tok) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    fstat(fd, &st);
    file_size = st.st_size;
    file_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (file_data == MAP_FAILED) { perror("mmap"); return -1; }

    void *cursor = file_data;

    /* Header */
    unsigned magic = read_uint(&cursor);
    if (magic != 0x47425931) { fprintf(stderr, "bad magic\n"); return -1; }
    read_uint(&cursor); /* version */

    /* Config */
    Config *c = &m->cfg;
    c->vocab_size   = read_int(&cursor);
    c->d_model      = read_int(&cursor);
    c->n_layers     = read_int(&cursor);
    c->n_heads      = read_int(&cursor);
    c->n_kv_heads   = read_int(&cursor);
    c->ffn_hidden   = read_int(&cursor);
    c->max_seq_len  = read_int(&cursor);
    c->early_exit   = read_int(&cursor);
    c->min_exit_layer = read_int(&cursor);
    c->exit_threshold = read_float(&cursor);
    c->head_dim = c->d_model / c->n_heads;
    c->kv_dim = c->n_kv_heads * c->head_dim;
    c->n_rep = c->n_heads / c->n_kv_heads;

    /* Tokenizer — vocab */
    tok->vocab_size = read_uint(&cursor);
    tok->tokens = (char **)malloc(tok->vocab_size * sizeof(char *));
    tok->token_lens = (int *)malloc(tok->vocab_size * sizeof(int));
    for (int i = 0; i < tok->vocab_size; i++) {
        int len = read_ushort(&cursor);
        tok->token_lens[i] = len;
        tok->tokens[i] = (char *)malloc(len + 1);
        memcpy(tok->tokens[i], read_ptr(&cursor, len), len);
        tok->tokens[i][len] = '\0';
    }

    /* Tokenizer — merges */
    tok->n_merges = read_uint(&cursor);
    tok->merges = (int *)malloc(tok->n_merges * 2 * sizeof(int));
    for (int i = 0; i < tok->n_merges; i++) {
        tok->merges[i * 2]     = read_uint(&cursor);
        tok->merges[i * 2 + 1] = read_uint(&cursor);
    }

    /* Tokenizer — byte-to-token */
    for (int i = 0; i < 256; i++) tok->byte_to_token[i] = read_uint(&cursor);

    /* Weights — tok_emb */
    m->tok_emb = read_floats(&cursor, c->vocab_size * c->d_model);

    /* Weights — layers */
    m->layers = (Layer *)malloc(c->n_layers * sizeof(Layer));
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->layers[i];
        l->norm_w   = read_floats(&cursor, c->d_model);
        l->wq       = read_floats(&cursor, c->n_heads * c->head_dim * c->d_model);
        l->wk       = read_floats(&cursor, c->kv_dim * c->d_model);
        l->wv       = read_floats(&cursor, c->kv_dim * c->d_model);
        l->wo       = read_floats(&cursor, c->d_model * c->d_model);
        l->ffn_gate = read_floats(&cursor, c->ffn_hidden * c->d_model);
        l->ffn_up   = read_floats(&cursor, c->ffn_hidden * c->d_model);
        l->ffn_down = read_floats(&cursor, c->d_model * c->ffn_hidden);
        if (c->early_exit) {
            l->router_w = read_floats(&cursor, c->d_model);  /* [1, d_model] */
            l->router_b = read_floats(&cursor, 1);
        }
    }

    m->final_norm_w = read_floats(&cursor, c->d_model);
    m->rope_cos = read_floats(&cursor, c->max_seq_len * c->head_dim / 2);
    m->rope_sin = read_floats(&cursor, c->max_seq_len * c->head_dim / 2);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  RUN STATE
 * ═══════════════════════════════════════════════════════════════════════════ */

static void init_run_state(RunState *s, Config *c) {
    int d = c->d_model;
    s->x        = (float *)calloc(d, sizeof(float));
    s->xb       = (float *)calloc(d, sizeof(float));
    s->xb2      = (float *)calloc(d, sizeof(float));
    s->q        = (float *)calloc(c->n_heads * c->head_dim, sizeof(float));
    s->att      = (float *)calloc(c->n_heads * c->max_seq_len, sizeof(float));
    s->ffn_buf1 = (float *)calloc(c->ffn_hidden, sizeof(float));
    s->ffn_buf2 = (float *)calloc(c->ffn_hidden, sizeof(float));
    s->logits   = (float *)calloc(c->vocab_size, sizeof(float));
    s->key_cache = (float *)calloc(c->n_layers * c->max_seq_len * c->kv_dim, sizeof(float));
    s->val_cache = (float *)calloc(c->n_layers * c->max_seq_len * c->kv_dim, sizeof(float));
    /* exit depth determined at runtime */
}

static void reset_state(RunState *s, Config *c) {
    memset(s->key_cache, 0, c->n_layers * c->max_seq_len * c->kv_dim * sizeof(float));
    memset(s->val_cache, 0, c->n_layers * c->max_seq_len * c->kv_dim * sizeof(float));
    /* exit depth determined at runtime */
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  FORWARD PASS (single token, with KV cache)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void forward(Model *m, RunState *s, int token, int pos, int max_layer) {
    Config *c = &m->cfg;
    int d = c->d_model;
    int hd = c->head_dim;
    int half_hd = hd / 2;
    int kv_dim = c->kv_dim;

    /* Token embedding */
    memcpy(s->x, m->tok_emb + token * d, d * sizeof(float));

    for (int layer = 0; layer < max_layer; layer++) {
        Layer *l = &m->layers[layer];

        /* RMSNorm */
        rmsnorm(s->xb, s->x, l->norm_w, d);

        /* ── Attention ──────────────────────────────────────────────── */
        /* Q, K, V projections */
        matmul(s->q, s->xb, l->wq, c->n_heads * hd, d);
        float *kv_k = s->key_cache + layer * c->max_seq_len * kv_dim + pos * kv_dim;
        float *kv_v = s->val_cache + layer * c->max_seq_len * kv_dim + pos * kv_dim;
        matmul(kv_k, s->xb, l->wk, kv_dim, d);
        matmul(kv_v, s->xb, l->wv, kv_dim, d);

        /* RoPE */
        float *cos_row = m->rope_cos + pos * half_hd;
        float *sin_row = m->rope_sin + pos * half_hd;
        for (int h = 0; h < c->n_heads; h++) {
            float *qh = s->q + h * hd;
            for (int j = 0; j < half_hd; j++) {
                float q0 = qh[j], q1 = qh[j + half_hd];
                qh[j]           = q0 * cos_row[j] - q1 * sin_row[j];
                qh[j + half_hd] = q1 * cos_row[j] + q0 * sin_row[j];
            }
        }
        for (int h = 0; h < c->n_kv_heads; h++) {
            float *kh = kv_k + h * hd;
            for (int j = 0; j < half_hd; j++) {
                float k0 = kh[j], k1 = kh[j + half_hd];
                kh[j]           = k0 * cos_row[j] - k1 * sin_row[j];
                kh[j + half_hd] = k1 * cos_row[j] + k0 * sin_row[j];
            }
        }

        /* Grouped query attention with KV cache */
        memset(s->xb2, 0, d * sizeof(float));
        for (int h = 0; h < c->n_heads; h++) {
            int kv_h = h / c->n_rep;  /* which KV head this Q head uses */
            float *qh = s->q + h * hd;
            float *att_h = s->att + h * c->max_seq_len;

            /* Compute attention scores for all cached positions */
            for (int t = 0; t <= pos; t++) {
                float *kt = s->key_cache + layer * c->max_seq_len * kv_dim + t * kv_dim + kv_h * hd;
                float score = 0.0f;
                for (int j = 0; j < hd; j++) score += qh[j] * kt[j];
                att_h[t] = score / sqrtf((float)hd);
            }

            /* Softmax over [0..pos] */
            softmax(att_h, pos + 1);

            /* Weighted sum of values */
            float *oh = s->xb2 + h * hd;
            for (int t = 0; t <= pos; t++) {
                float *vt = s->val_cache + layer * c->max_seq_len * kv_dim + t * kv_dim + kv_h * hd;
                float a = att_h[t];
                for (int j = 0; j < hd; j++) oh[j] += a * vt[j];
            }
        }

        /* Output projection (attn) */
        float attn_out[d];
        matmul(attn_out, s->xb2, l->wo, d, d);

        /* ── FFN (SwiGLU, parallel residual) ────────────────────────── */
        matmul(s->ffn_buf1, s->xb, l->ffn_gate, c->ffn_hidden, d);
        matmul(s->ffn_buf2, s->xb, l->ffn_up, c->ffn_hidden, d);
        for (int j = 0; j < c->ffn_hidden; j++)
            s->ffn_buf1[j] = silu_f(s->ffn_buf1[j]) * s->ffn_buf2[j];
        float ffn_out[d];
        matmul(ffn_out, s->ffn_buf1, l->ffn_down, d, c->ffn_hidden);

        /* Parallel residual: x = x + attn_out + ffn_out */
        for (int j = 0; j < d; j++)
            s->x[j] += attn_out[j] + ffn_out[j];
    }

    /* Final norm + lm_head (tok_emb weights, transposed) */
    rmsnorm(s->xb, s->x, m->final_norm_w, d);
    matmul(s->logits, s->xb, m->tok_emb, c->vocab_size, d);
}

/* Early exit: check router confidence after full prompt processing */
static int check_exit_depth(Model *m, RunState *s) {
    Config *c = &m->cfg;
    if (!c->early_exit) return c->n_layers;
    for (int i = c->min_exit_layer; i < c->n_layers - 1; i++) {
        Layer *l = &m->layers[i];
        float dot = l->router_b[0];
        for (int j = 0; j < c->d_model; j++) dot += s->x[j] * l->router_w[j];
        float conf = 1.0f / (1.0f + expf(-dot));
        if (conf > c->exit_threshold) return i + 1;
    }
    return c->n_layers;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  TOKENIZER
 * ═══════════════════════════════════════════════════════════════════════════ */

static int *bpe_encode(Tokenizer *tok, const char *text, int *n_tokens) {
    int len = strlen(text);
    int *tokens = (int *)malloc((len + 128) * sizeof(int));
    int n = 0;

    /* Check for special tokens at the beginning */
    const char *specials[] = {
        "<|im_start|>", "<|im_end|>", "<think>", "</think>",
        "<tool_call>", "</tool_call>", "<pad>", NULL
    };
    int special_ids[] = {1, 2, 3, 4, 5, 6, 0};

    int i = 0;
    while (i < len) {
        /* Try special tokens */
        int matched = 0;
        for (int s = 0; specials[s]; s++) {
            int slen = strlen(specials[s]);
            if (i + slen <= len && strncmp(text + i, specials[s], slen) == 0) {
                tokens[n++] = special_ids[s];
                i += slen;
                matched = 1;
                break;
            }
        }
        if (matched) continue;

        /* Map byte to initial token */
        tokens[n++] = tok->byte_to_token[(unsigned char)text[i]];
        i++;
    }

    /* Apply BPE merges */
    for (int m = 0; m < tok->n_merges; m++) {
        int id1 = tok->merges[m * 2];
        int id2 = tok->merges[m * 2 + 1];
        /* Find merged token ID: the token whose bytes = bytes(id1) + bytes(id2) */
        /* For simplicity, search vocab. In production, pre-build a merge→result map. */
        int result_id = -1;
        int len1 = tok->token_lens[id1];
        int len2 = tok->token_lens[id2];
        int merged_len = len1 + len2;
        for (int v = 0; v < tok->vocab_size; v++) {
            if (tok->token_lens[v] == merged_len &&
                memcmp(tok->tokens[v], tok->tokens[id1], len1) == 0 &&
                memcmp(tok->tokens[v] + len1, tok->tokens[id2], len2) == 0) {
                result_id = v;
                break;
            }
        }
        if (result_id < 0) continue;

        for (int j = 0; j < n - 1; j++) {
            if (tokens[j] == id1 && tokens[j + 1] == id2) {
                tokens[j] = result_id;
                memmove(&tokens[j + 1], &tokens[j + 2], (n - j - 2) * sizeof(int));
                n--;
                j--;
            }
        }
    }

    *n_tokens = n;
    return tokens;
}

static void decode_token(Tokenizer *tok, int id, char *buf, int buf_size) {
    if (id >= 0 && id < tok->vocab_size) {
        int len = tok->token_lens[id];
        if (len >= buf_size) len = buf_size - 1;
        memcpy(buf, tok->tokens[id], len);
        buf[len] = '\0';
    } else {
        buf[0] = '\0';
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  SAMPLING
 * ═══════════════════════════════════════════════════════════════════════════ */

static int sample_topk(float *logits, int vocab_size, float temperature, int top_k) {
    if (temperature < 1e-6f) {
        /* Greedy */
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* Apply temperature */
    for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;

    /* Top-K: find Kth largest, mask below it */
    if (top_k > 0 && top_k < vocab_size) {
        float threshold = -1e9f;
        /* Partial sort: find top_k-th value */
        float *tmp = (float *)malloc(vocab_size * sizeof(float));
        memcpy(tmp, logits, vocab_size * sizeof(float));
        /* Simple selection for small K */
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < vocab_size; j++)
                if (tmp[j] > tmp[best]) best = j;
            float t = tmp[i]; tmp[i] = tmp[best]; tmp[best] = t;
        }
        threshold = tmp[top_k - 1];
        free(tmp);
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < threshold) logits[i] = -1e9f;
    }

    softmax(logits, vocab_size);

    /* Sample from distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  GENERATION
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int exit_depth;
    int tokens_generated;
    double prompt_ms;
    double gen_ms;
    double tokens_per_sec;
} GenStats;

static void generate(Model *m, Tokenizer *tok, RunState *s,
                     int *prompt, int prompt_len, int max_tokens,
                     float temperature, int top_k, GenStats *stats) {
    Config *c = &m->cfg;
    reset_state(s, c);

    /* Phase 1: Process prompt tokens (all layers, fills KV cache) */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < prompt_len; i++) {
        forward(m, s, prompt[i], i, c->n_layers);
    }

    /* Determine early exit depth from router on last prompt token */
    int exit_depth = check_exit_depth(m, s);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prompt_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* Phase 2: Generate tokens with KV cache at fixed depth */
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int pos = prompt_len;
    int next_token = sample_topk(s->logits, c->vocab_size, temperature, top_k);
    int gen_count = 0;
    char decoded[256];

    for (int t = 0; t < max_tokens; t++) {
        if (next_token == 2) break;  /* <|im_end|> */

        decode_token(tok, next_token, decoded, sizeof(decoded));
        printf("%s", decoded);
        fflush(stdout);
        gen_count++;

        if (pos >= c->max_seq_len - 1) break;

        forward(m, s, next_token, pos, exit_depth);
        next_token = sample_topk(s->logits, c->vocab_size, temperature, top_k);
        pos++;
    }
    printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gen_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    if (stats) {
        stats->exit_depth = exit_depth;
        stats->tokens_generated = gen_count;
        stats->prompt_ms = prompt_ms;
        stats->gen_ms = gen_ms;
        stats->tokens_per_sec = gen_count > 0 ? gen_count / (gen_ms / 1000.0) : 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  BENCHMARK
 * ═══════════════════════════════════════════════════════════════════════════ */

static void benchmark(Model *m, Tokenizer *tok, RunState *s) {
    Config *c = &m->cfg;
    int prompt_len = 32, gen_len = 64, n_runs = 5;

    printf("\n══════════════════════════════════════════\n");
    printf("GobyLLM Benchmark (C runtime)\n");
    printf("  Model: %dM params, %d layers\n", (int)(c->vocab_size * c->d_model / 1e6 + c->n_layers * 2.0), c->n_layers);
    printf("  Prompt: %d tokens, Generate: %d tokens\n", prompt_len, gen_len);
    printf("══════════════════════════════════════════\n");

    int prompt[32];
    for (int i = 0; i < prompt_len; i++) prompt[i] = (rand() % (c->vocab_size - 7)) + 7;

    double total_prompt = 0, total_gen = 0;
    int total_depth = 0;

    for (int r = 0; r < n_runs; r++) {
        GenStats stats = {0};
        /* Suppress output during benchmark */
        FILE *devnull = fopen("/dev/null", "w");
        FILE *saved_stdout = stdout;
        stdout = devnull;
        generate(m, tok, s, prompt, prompt_len, gen_len, 0.8f, 40, &stats);
        stdout = saved_stdout;
        fclose(devnull);

        total_prompt += stats.prompt_ms;
        total_gen += stats.gen_ms;
        total_depth += stats.exit_depth;
    }

    double avg_prompt = total_prompt / n_runs;
    double avg_gen = total_gen / n_runs;
    double avg_depth = (double)total_depth / n_runs;
    double tok_per_sec = gen_len / (avg_gen / 1000.0);

    printf("\n  Prompt (%d tokens):   %7.1f ms\n", prompt_len, avg_prompt);
    printf("  Generate (%d tokens): %7.1f ms  (%.1f tok/s)\n", gen_len, avg_gen, tok_per_sec);
    printf("  Avg exit depth:      %.1f / %d layers\n", avg_depth, c->n_layers);
    printf("  Compute saved (EE):  %.0f%%\n", (1.0 - avg_depth / c->n_layers) * 100);

    size_t model_bytes = file_size;
    size_t cache_bytes = c->n_layers * c->max_seq_len * c->kv_dim * 2 * sizeof(float);
    printf("\n  Model file:  %.1f MB\n", model_bytes / 1e6);
    printf("  KV cache:    %.1f MB\n", cache_bytes / 1e6);
    printf("  Total RAM:   %.1f MB\n", (model_bytes + cache_bytes) / 1e6);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "GobyLLM C Runtime\n\n");
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s model.bin -p \"prompt\"      Generate from prompt\n", argv[0]);
        fprintf(stderr, "  %s model.bin -i               Interactive mode\n", argv[0]);
        fprintf(stderr, "  %s model.bin -b               Benchmark\n", argv[0]);
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  -t <float>   Temperature (default 0.7)\n");
        fprintf(stderr, "  -k <int>     Top-K (default 40)\n");
        fprintf(stderr, "  -n <int>     Max tokens (default 256)\n");
        return 1;
    }

    srand(time(NULL));

    const char *model_path = argv[1];
    const char *prompt = NULL;
    int interactive = 0, do_bench = 0;
    float temperature = 0.7f;
    int top_k = 40, max_tokens = 256;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-i") == 0) interactive = 1;
        else if (strcmp(argv[i], "-b") == 0) do_bench = 1;
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
    }

    Model model;
    Tokenizer tok;
    printf("Loading %s...\n", model_path);
    if (load_model(model_path, &model, &tok) != 0) return 1;

    Config *c = &model.cfg;
    printf("GobyLLM: vocab=%d, d=%d, layers=%d, heads=%dQ/%dKV, ffn=%d\n",
           c->vocab_size, c->d_model, c->n_layers, c->n_heads, c->n_kv_heads, c->ffn_hidden);
    printf("Early exit: %s (min_layer=%d, threshold=%.2f)\n",
           c->early_exit ? "ON" : "OFF", c->min_exit_layer, c->exit_threshold);

    RunState state;
    init_run_state(&state, c);

    if (do_bench) {
        benchmark(&model, &tok, &state);
    } else if (interactive) {
        char input[4096];
        printf("\nGobyLLM Interactive (type 'quit' to exit)\n");
        while (1) {
            printf("\nYou> ");
            if (!fgets(input, sizeof(input), stdin)) break;
            input[strcspn(input, "\n")] = '\0';
            if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
            if (input[0] == '\0') continue;

            /* Build chat prompt */
            char full_prompt[8192];
            snprintf(full_prompt, sizeof(full_prompt),
                     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                     "<|im_start|>user\n%s<|im_end|>\n"
                     "<|im_start|>assistant\n", input);

            int n_tokens;
            int *tokens = bpe_encode(&tok, full_prompt, &n_tokens);

            GenStats stats;
            printf("Goby> ");
            generate(&model, &tok, &state, tokens, n_tokens, max_tokens,
                     temperature, top_k, &stats);
            printf("  [%.1f tok/s, depth %d/%d, %.0f%% saved]\n",
                   stats.tokens_per_sec, stats.exit_depth, c->n_layers,
                   (1.0 - (double)stats.exit_depth / c->n_layers) * 100);
            free(tokens);
        }
    } else if (prompt) {
        /* Build chat prompt */
        char full_prompt[8192];
        snprintf(full_prompt, sizeof(full_prompt),
                 "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\n%s<|im_end|>\n"
                 "<|im_start|>assistant\n", prompt);

        int n_tokens;
        int *tokens = bpe_encode(&tok, full_prompt, &n_tokens);

        GenStats stats;
        generate(&model, &tok, &state, tokens, n_tokens, max_tokens,
                 temperature, top_k, &stats);
        fprintf(stderr, "\n[%d tokens, %.1f tok/s, depth %d/%d, prompt %.0fms, gen %.0fms]\n",
                stats.tokens_generated, stats.tokens_per_sec,
                stats.exit_depth, c->n_layers, stats.prompt_ms, stats.gen_ms);
        free(tokens);
    }

    /* Cleanup */
    munmap(file_data, file_size);
    return 0;
}

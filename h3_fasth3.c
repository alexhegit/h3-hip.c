#include "h3_fasth3.h"

#include "h3_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef H3_HIP

int h3_fasth3_apply(h3_gpu *gpu, h3_gpu_tensor *weight, const char *name,
                    uint32_t rows, uint32_t cols, int weight_f32,
                    char *error, size_t error_size) {
    const char *path = getenv("H3_FASTH3_LORA");
    (void)gpu;
    (void)weight;
    (void)name;
    (void)rows;
    (void)cols;
    (void)weight_f32;
    if (!path || !path[0] || !strcmp(path, "0")) return 1;
    if (error && error_size)
        snprintf(error, error_size, "FastH3 LoRA requires the HIP backend");
    return 0;
}

int h3_fasth3_finish(char *error, size_t error_size) {
    const char *path = getenv("H3_FASTH3_LORA");
    if (!path || !path[0] || !strcmp(path, "0")) return 1;
    if (error && error_size)
        snprintf(error, error_size, "FastH3 LoRA requires the HIP backend");
    return 0;
}

int h3_fasth3_load_gate(h3_gpu *gpu, unsigned block, uint32_t rows,
                        uint32_t cols, h3_gpu_tensor **out, char *error,
                        size_t error_size) {
    (void)gpu;
    (void)block;
    (void)rows;
    (void)cols;
    (void)out;
    if (error && error_size)
        snprintf(error, error_size, "VSA requires the HIP backend");
    return 0;
}

#else

enum { FASTH3_PLAIN = 0, FASTH3_QKV = 1, FASTH3_FC1 = 2 };
enum { FASTH3_HEADS = 56u, FASTH3_HEAD_DIM = 128u, FASTH3_FFN = 14336u };

static struct {
    int open;
    int failed;
    float strength;
    h3_st_header header;
    unsigned char *used;
    h3_gpu_tensor *qkv_delta;
    char path[1024];
} fasth3;

static void fasth3_fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static int fasth3_enabled(void) {
    const char *path = getenv("H3_FASTH3_LORA");
    return path && path[0] && strcmp(path, "0") != 0;
}

static int ends_with(const char *text, const char *suffix) {
    size_t text_len = strlen(text);
    size_t suffix_len = strlen(suffix);
    return text_len >= suffix_len &&
           strcmp(text + text_len - suffix_len, suffix) == 0;
}

static int replace_span(char *text, size_t cap, const char *from,
                        const char *to) {
    char *hit = strstr(text, from);
    if (!hit) return 0;
    size_t from_len = strlen(from);
    size_t to_len = strlen(to);
    size_t tail = strlen(hit + from_len);
    if ((size_t)(hit - text) + to_len + tail + 1 > cap) return -1;
    memmove(hit + to_len, hit + from_len, tail + 1);
    memcpy(hit, to, to_len);
    return 1;
}

static int map_checkpoint(const char *checkpoint, char *stem, size_t stem_cap,
                          int *kind, int *is_bias) {
    if (snprintf(stem, stem_cap, "%s", checkpoint) >= (int)stem_cap) return 0;
    *is_bias = 0;
    *kind = FASTH3_PLAIN;
    if (ends_with(stem, ".weight")) {
        stem[strlen(stem) - strlen(".weight")] = '\0';
    } else if (ends_with(stem, ".bias")) {
        stem[strlen(stem) - strlen(".bias")] = '\0';
        *is_bias = 1;
    }
    if (strstr(stem, ".attn.qkv_proj")) {
        *kind = FASTH3_QKV;
        if (replace_span(stem, stem_cap, ".attn.qkv_proj", ".attn") < 0)
            return 0;
    } else if (ends_with(stem, ".mlp.fc1")) {
        *kind = FASTH3_FC1;
        if (replace_span(stem, stem_cap, ".mlp.fc1", ".ff.net.0.proj") < 0)
            return 0;
    } else if (ends_with(stem, ".attn.out_proj")) {
        if (replace_span(stem, stem_cap, ".attn.out_proj", ".attn.to_out.0") < 0)
            return 0;
    } else if (ends_with(stem, ".mlp.fc2")) {
        if (replace_span(stem, stem_cap, ".mlp.fc2", ".ff.net.2") < 0)
            return 0;
    } else if (!strcmp(stem, "final_layer.adaln_proj.linear")) {
        snprintf(stem, stem_cap, "norm_out.linear");
    } else if (!strcmp(stem, "final_layer.norm")) {
        snprintf(stem, stem_cap, "norm_out.norm");
    } else if (!strcmp(stem, "final_layer.video_out")) {
        snprintf(stem, stem_cap, "proj_out");
    } else if (!strcmp(stem, "final_layer.audio_out")) {
        snprintf(stem, stem_cap, "audio_proj_out");
    } else if (!strcmp(stem, "video_patch_proj")) {
        snprintf(stem, stem_cap, "proj_in");
    } else if (!strcmp(stem, "audio_patch_proj")) {
        snprintf(stem, stem_cap, "audio_proj_in");
    } else if (!strcmp(stem, "time_embedder.proj_in")) {
        snprintf(stem, stem_cap, "time_embedder.linear_1");
    } else if (!strcmp(stem, "time_embedder.proj_out")) {
        snprintf(stem, stem_cap, "time_embedder.linear_2");
    } else if (!strcmp(stem, "condition_proj")) {
        snprintf(stem, stem_cap, "context_embedder");
    }
    if (!strncmp(stem, "token_refiner.blocks.", 21)) {
        if (replace_span(stem, stem_cap, "token_refiner.blocks.",
                         "token_refiner.refiner_blocks.") < 0)
            return 0;
    } else if (!strncmp(stem, "blocks.", 7)) {
        if (replace_span(stem, stem_cap, "blocks.", "transformer_blocks.") < 0)
            return 0;
    }
    return 1;
}

static void mark_tensor(const h3_st_tensor *tensor) {
    if (!tensor || !fasth3.used) return;
    size_t index = (size_t)(tensor - fasth3.header.tensors);
    if (index < fasth3.header.tensor_count) fasth3.used[index] = 1;
}

static const h3_st_tensor *find_key(const char *key) {
    const h3_st_tensor *tensor = h3_st_find(&fasth3.header, key);
    if (tensor) mark_tensor(tensor);
    return tensor;
}

static int shape_is(const h3_st_tensor *tensor, uint32_t rows, uint32_t cols) {
    if (!tensor || tensor->dtype != H3_DTYPE_BF16) return 0;
    if (cols == 1 && tensor->ndim == 1 && tensor->shape[0] == rows) return 1;
    return tensor->ndim == 2 && tensor->shape[0] == rows &&
           tensor->shape[1] == cols;
}

static uint16_t *read_bf16(const h3_st_tensor *tensor, char *error,
                           size_t error_size) {
    uint64_t elements = h3_st_tensor_elements(tensor);
    if (!elements || elements > SIZE_MAX / sizeof(uint16_t)) {
        fasth3_fail(error, error_size, "FastH3 tensor is too large");
        return NULL;
    }
    uint16_t *data = malloc((size_t)elements * sizeof(uint16_t));
    if (!data) {
        fasth3_fail(error, error_size, "out of memory reading FastH3 tensor");
        return NULL;
    }
    if (!h3_st_read_data(&fasth3.header, tensor, data,
                         (size_t)elements * sizeof(uint16_t), error,
                         error_size)) {
        free(data);
        return NULL;
    }
    return data;
}

static h3_gpu_tensor *upload_bf16(h3_gpu *gpu, const uint16_t *data,
                                  size_t elements, char *error,
                                  size_t error_size) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, data, elements);
    if (!tensor)
        fasth3_fail(error, error_size, "cannot upload FastH3 tensor");
    return tensor;
}

static int resolve_adapter_path(const char *input, char *output, size_t cap,
                                char *error, size_t error_size) {
    struct stat info;
    if (stat(input, &info) != 0) {
        fasth3_fail(error, error_size, "FastH3 LoRA path does not exist");
        return 0;
    }
    if (S_ISDIR(info.st_mode)) {
        if (snprintf(output, cap, "%s/adapter_model.safetensors", input) >=
            (int)cap) {
            fasth3_fail(error, error_size, "FastH3 LoRA path is too long");
            return 0;
        }
        return 1;
    }
    if (snprintf(output, cap, "%s", input) >= (int)cap) {
        fasth3_fail(error, error_size, "FastH3 LoRA path is too long");
        return 0;
    }
    return 1;
}

static int open_adapter(char *error, size_t error_size) {
    const char *env = getenv("H3_FASTH3_LORA");
    const char *strength_env = getenv("H3_FASTH3_STRENGTH");
    char path[1024];
    if (fasth3.open) return 1;
    if (fasth3.failed) {
        fasth3_fail(error, error_size, "FastH3 LoRA already failed to open");
        return 0;
    }
    fasth3.strength = 1.0f;
    if (strength_env && *strength_env) {
        fasth3.strength = strtof(strength_env, NULL);
        if (!(fasth3.strength >= 0.0f)) {
            fasth3_fail(error, error_size, "H3_FASTH3_STRENGTH must be >= 0");
            fasth3.failed = 1;
            return 0;
        }
    }
    if (!env || !resolve_adapter_path(env, path, sizeof(path), error,
                                      error_size) ||
        !h3_st_read_header(path, &fasth3.header, error, error_size)) {
        fasth3.failed = 1;
        return 0;
    }
    fasth3.used = calloc(fasth3.header.tensor_count, 1);
    if (!fasth3.used) {
        h3_st_free_header(&fasth3.header);
        fasth3_fail(error, error_size, "out of memory indexing FastH3 LoRA");
        fasth3.failed = 1;
        return 0;
    }
    snprintf(fasth3.path, sizeof(fasth3.path), "%s", path);
    fasth3.open = 1;
    fprintf(stderr,
            "h3: FastH3 LoRA %s strength %.3f "
            "(W += B @ A, then .diff; 4 steps 999/749/500/250)\n",
            fasth3.path, fasth3.strength);
    return 1;
}

static int apply_diff(h3_gpu *gpu, h3_gpu_tensor *weight,
                      const h3_st_tensor *tensor, uint32_t rows, uint32_t cols,
                      int weight_f32, char *error, size_t error_size) {
    size_t count = (size_t)rows * (size_t)cols;
    uint16_t *host;
    h3_gpu_tensor *device;
    int ok;
    if (!shape_is(tensor, rows, cols)) {
        fasth3_fail(error, error_size, "FastH3 diff shape does not match");
        return 0;
    }
    host = read_bf16(tensor, error, error_size);
    if (!host) return 0;
    device = upload_bf16(gpu, host, count, error, error_size);
    free(host);
    if (!device) return 0;
    ok = weight_f32 ? h3_gpu_fasth3_axpy_f32(gpu, weight, device, count,
                                             fasth3.strength)
                    : h3_gpu_fasth3_axpy_bf16(gpu, weight, device, count,
                                              fasth3.strength);
    h3_gpu_tensor_free(device);
    if (!ok) fasth3_fail(error, error_size, "cannot add FastH3 diff");
    return ok;
}

static int load_factor(h3_gpu *gpu, const char *key, uint32_t rows,
                       uint32_t cols, h3_gpu_tensor **out, char *error,
                       size_t error_size) {
    const h3_st_tensor *tensor = find_key(key);
    uint16_t *host;
    if (!tensor) {
        snprintf(error, error_size, "FastH3 adapter is missing %s", key);
        return 0;
    }
    if (!shape_is(tensor, rows, cols)) {
        snprintf(error, error_size, "FastH3 %s has the wrong shape", key);
        return 0;
    }
    host = read_bf16(tensor, error, error_size);
    if (!host) return 0;
    *out = upload_bf16(gpu, host, (size_t)rows * (size_t)cols, error,
                       error_size);
    free(host);
    return *out != NULL;
}

static int apply_gemm(h3_gpu *gpu, h3_gpu_tensor *weight, size_t weight_elem,
                      const char *a_key, const char *b_key, uint32_t rows,
                      uint32_t cols, float beta, char *error,
                      size_t error_size) {
    h3_gpu_tensor *a = NULL;
    h3_gpu_tensor *b = NULL;
    const h3_st_tensor *a_tensor = h3_st_find(&fasth3.header, a_key);
    uint32_t rank;
    int ok = 0;
    if (!a_tensor || a_tensor->ndim != 2 || a_tensor->shape[1] != cols ||
        a_tensor->shape[0] == 0 || a_tensor->shape[0] > 4096u) {
        snprintf(error, error_size, "FastH3 %s is not rank x %u", a_key, cols);
        return 0;
    }
    rank = (uint32_t)a_tensor->shape[0];
    if (!load_factor(gpu, a_key, rank, cols, &a, error, error_size) ||
        !load_factor(gpu, b_key, rows, rank, &b, error, error_size))
        goto done;
    ok = h3_gpu_fasth3_gemm_bf16(gpu, weight, weight_elem, b, 0, a, rows, cols,
                                 rank, fasth3.strength, beta);
    if (!ok) fasth3_fail(error, error_size, "FastH3 LoRA GEMM failed");
done:
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(b);
    return ok;
}

static int ensure_qkv_delta(h3_gpu *gpu, uint32_t cols, char *error,
                            size_t error_size) {
    size_t elements = (size_t)FASTH3_HEADS * FASTH3_HEAD_DIM * cols;
    if (fasth3.qkv_delta) return 1;
    fasth3.qkv_delta = h3_gpu_tensor_new_bf16_device(gpu, elements);
    if (!fasth3.qkv_delta) {
        fasth3_fail(error, error_size, "cannot allocate FastH3 QKV delta");
        return 0;
    }
    return 1;
}

static int apply_qkv(h3_gpu *gpu, h3_gpu_tensor *weight, const char *stem,
                     uint32_t rows, uint32_t cols, char *error,
                     size_t error_size) {
    static const char *which_name[3] = {"to_q", "to_k", "to_v"};
    uint32_t inner = FASTH3_HEADS * FASTH3_HEAD_DIM;
    if (rows != inner * 3u || cols == 0) {
        fasth3_fail(error, error_size, "FastH3 QKV weight has the wrong shape");
        return 0;
    }
    if (!ensure_qkv_delta(gpu, cols, error, error_size)) return 0;
    for (uint32_t which = 0; which < 3u; which++) {
        char a_key[320];
        char b_key[320];
        if (snprintf(a_key, sizeof(a_key), "%s.%s.lora_A.weight", stem,
                     which_name[which]) >= (int)sizeof(a_key) ||
            snprintf(b_key, sizeof(b_key), "%s.%s.lora_B.weight", stem,
                     which_name[which]) >= (int)sizeof(b_key)) {
            fasth3_fail(error, error_size, "FastH3 QKV key is too long");
            return 0;
        }
        if (!apply_gemm(gpu, fasth3.qkv_delta, 0, a_key, b_key, inner, cols,
                        0.0f, error, error_size) ||
            !h3_gpu_fasth3_qkv_scatter_bf16(gpu, weight, fasth3.qkv_delta,
                                            FASTH3_HEADS, FASTH3_HEAD_DIM, cols,
                                            which)) {
            if (error && error_size && !error[0])
                fasth3_fail(error, error_size, "FastH3 QKV scatter failed");
            return 0;
        }
    }
    return 1;
}

static int apply_fc1(h3_gpu *gpu, h3_gpu_tensor *weight, const char *stem,
                     uint32_t rows, uint32_t cols, char *error,
                     size_t error_size) {
    char a_key[320];
    char b_key[320];
    h3_gpu_tensor *a = NULL;
    h3_gpu_tensor *b = NULL;
    const h3_st_tensor *a_tensor;
    uint32_t rank;
    int ok = 0;
    size_t value_elem;
    size_t gate_b_elem;
    if (rows != FASTH3_FFN * 2u) {
        fasth3_fail(error, error_size, "FastH3 fc1 weight has the wrong shape");
        return 0;
    }
    if (snprintf(a_key, sizeof(a_key), "%s.lora_A.weight", stem) >=
            (int)sizeof(a_key) ||
        snprintf(b_key, sizeof(b_key), "%s.lora_B.weight", stem) >=
            (int)sizeof(b_key)) {
        fasth3_fail(error, error_size, "FastH3 fc1 key is too long");
        return 0;
    }
    a_tensor = h3_st_find(&fasth3.header, a_key);
    if (!a_tensor || a_tensor->ndim != 2 || a_tensor->shape[1] != cols ||
        a_tensor->shape[0] == 0 || a_tensor->shape[0] > 4096u) {
        snprintf(error, error_size, "FastH3 %s is not rank x %u", a_key, cols);
        return 0;
    }
    rank = (uint32_t)a_tensor->shape[0];
    if (!load_factor(gpu, a_key, rank, cols, &a, error, error_size) ||
        !load_factor(gpu, b_key, rows, rank, &b, error, error_size))
        goto done;
    /* Diffusers packs [value; gate]. This tree packs [gate; value]. */
    value_elem = (size_t)FASTH3_FFN * cols;
    gate_b_elem = (size_t)FASTH3_FFN * rank;
    ok = h3_gpu_fasth3_gemm_bf16(gpu, weight, value_elem, b, 0, a, FASTH3_FFN,
                                 cols, rank, fasth3.strength, 1.0f) &&
         h3_gpu_fasth3_gemm_bf16(gpu, weight, 0, b, gate_b_elem, a, FASTH3_FFN,
                                 cols, rank, fasth3.strength, 1.0f);
    if (!ok) fasth3_fail(error, error_size, "FastH3 fc1 LoRA GEMM failed");
done:
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(b);
    return ok;
}

static int apply_plain(h3_gpu *gpu, h3_gpu_tensor *weight, const char *stem,
                       int is_bias, uint32_t rows, uint32_t cols,
                       int weight_f32, char *error, size_t error_size) {
    char key[320];
    const h3_st_tensor *diff;
    const char *suffix = is_bias ? ".diff_b" : ".diff";
    if (!is_bias) {
        char a_key[320];
        char b_key[320];
        if (snprintf(a_key, sizeof(a_key), "%s.lora_A.weight", stem) >=
            (int)sizeof(a_key)) {
            fasth3_fail(error, error_size, "FastH3 key is too long");
            return 0;
        }
        if (h3_st_find(&fasth3.header, a_key)) {
            if (snprintf(b_key, sizeof(b_key), "%s.lora_B.weight", stem) >=
                (int)sizeof(b_key)) {
                fasth3_fail(error, error_size, "FastH3 key is too long");
                return 0;
            }
            if (weight_f32) {
                fasth3_fail(error, error_size,
                            "FastH3 low-rank update on an f32 weight is not supported");
                return 0;
            }
            if (!apply_gemm(gpu, weight, 0, a_key, b_key, rows, cols, 1.0f,
                            error, error_size))
                return 0;
        }
    }
    if (snprintf(key, sizeof(key), "%s%s", stem, suffix) >= (int)sizeof(key)) {
        fasth3_fail(error, error_size, "FastH3 key is too long");
        return 0;
    }
    diff = h3_st_find(&fasth3.header, key);
    if (!diff) return 1;
    mark_tensor(diff);
    return apply_diff(gpu, weight, diff, rows, cols, weight_f32, error,
                      error_size);
}

static float fasth3_bf16_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static uint16_t fasth3_f32_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

int h3_fasth3_load_gate(h3_gpu *gpu, unsigned block, uint32_t rows,
                        uint32_t cols, h3_gpu_tensor **out, char *error,
                        size_t error_size) {
    char key[320];
    char local[256];
    const h3_st_tensor *tensor;
    uint16_t *host = NULL;
    size_t count;
    if (!error || !error_size) {
        error = local;
        error_size = sizeof(local);
    }
    if (!gpu || !out || !rows || !cols) {
        fasth3_fail(error, error_size, "invalid FastH3 gate request");
        return 0;
    }
    *out = NULL;
    if (!open_adapter(error, error_size)) return 0;
    if (snprintf(key, sizeof(key),
                 "transformer_blocks.%u.attn.to_gate_compress.set_weight",
                 block) >= (int)sizeof(key)) {
        fasth3_fail(error, error_size, "FastH3 gate key is too long");
        return 0;
    }
    tensor = h3_st_find(&fasth3.header, key);
    if (!tensor) {
        snprintf(error, error_size,
                 "FastH3 adapter is missing %s; --vsa needs vsa-datafree",
                 key);
        return 0;
    }
    if (tensor->ndim != 2 || tensor->shape[0] != (uint64_t)rows ||
        tensor->shape[1] != (uint64_t)cols ||
        (tensor->dtype != H3_DTYPE_BF16 && tensor->dtype != H3_DTYPE_F32)) {
        snprintf(error, error_size, "FastH3 %s has the wrong shape", key);
        return 0;
    }
    count = (size_t)rows * (size_t)cols;
    host = malloc(count * sizeof(uint16_t));
    if (!host) {
        fasth3_fail(error, error_size, "out of memory reading FastH3 gate");
        return 0;
    }
    if (tensor->dtype == H3_DTYPE_BF16) {
        if (!h3_st_read_data(&fasth3.header, tensor, host,
                             count * sizeof(uint16_t), error, error_size)) {
            free(host);
            return 0;
        }
    } else {
        float *wide = malloc(count * sizeof(float));
        if (!wide ||
            !h3_st_read_data(&fasth3.header, tensor, wide,
                             count * sizeof(float), error, error_size)) {
            free(wide);
            free(host);
            if (error && error_size && !error[0])
                fasth3_fail(error, error_size,
                            "out of memory reading FastH3 gate");
            return 0;
        }
        for (size_t index = 0; index < count; index++)
            host[index] = fasth3_f32_bf16(wide[index]);
        free(wide);
    }
    if (fasth3.strength != 1.0f) {
        for (size_t index = 0; index < count; index++)
            host[index] = fasth3_f32_bf16(fasth3_bf16_f32(host[index]) *
                                          fasth3.strength);
    }
    *out = upload_bf16(gpu, host, count, error, error_size);
    free(host);
    if (!*out) return 0;
    mark_tensor(tensor);
    return 1;
}

int h3_fasth3_apply(h3_gpu *gpu, h3_gpu_tensor *weight, const char *name,
                    uint32_t rows, uint32_t cols, int weight_f32,
                    char *error, size_t error_size) {
    char stem[256];
    int kind = FASTH3_PLAIN;
    int is_bias = 0;
    char local[256];
    if (!fasth3_enabled()) return 1;
    if (!error || !error_size) {
        error = local;
        error_size = sizeof(local);
    }
    if (!gpu || !weight || !name || !rows || !cols) {
        fasth3_fail(error, error_size, "invalid FastH3 merge request");
        return 0;
    }
    if (!open_adapter(error, error_size)) return 0;
    if (!map_checkpoint(name, stem, sizeof(stem), &kind, &is_bias)) {
        fasth3_fail(error, error_size, "cannot map FastH3 checkpoint name");
        return 0;
    }
    if (kind == FASTH3_QKV)
        return apply_qkv(gpu, weight, stem, rows, cols, error, error_size);
    if (kind == FASTH3_FC1)
        return apply_fc1(gpu, weight, stem, rows, cols, error, error_size);
    return apply_plain(gpu, weight, stem, is_bias, rows, cols, weight_f32,
                       error, error_size);
}

int h3_fasth3_finish(char *error, size_t error_size) {
    char local[256];
    size_t pending = 0;
    size_t applied = 0;
    if (!fasth3_enabled()) return 1;
    if (!error || !error_size) {
        error = local;
        error_size = sizeof(local);
    }
    if (!fasth3.open && !open_adapter(error, error_size)) return 0;
    const char *vsa_env = getenv("H3_VSA");
    int vsa = vsa_env && vsa_env[0] && strcmp(vsa_env, "0") != 0;
    int gate_unused = 0;
    for (size_t index = 0; index < fasth3.header.tensor_count; index++) {
        const char *name = fasth3.header.tensors[index].name;
        if (fasth3.used[index]) {
            applied++;
            continue;
        }
        if (name && strstr(name, "to_gate_compress")) gate_unused = 1;
        if (!pending && error && error_size && name) {
            snprintf(error, error_size, "FastH3 key was not applied: %s",
                     name);
        }
        pending++;
    }
    if (pending) {
        if (gate_unused && !vsa && error && error_size)
            snprintf(error, error_size,
                     "FastH3 adapter has to_gate_compress; pass --vsa");
        fprintf(stderr, "h3: FastH3 left %zu adapter tensors unused\n",
                pending);
        return 0;
    }
    fprintf(stderr, "h3: FastH3 merged %zu adapter tensors\n", applied);
    return 1;
}

#endif

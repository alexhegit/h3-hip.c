#include "h3d_config.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *getenv_or(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value && *value ? value : fallback;
}

static int getenv_int(const char *name, int fallback) {
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0) return fallback;
    return (int)parsed;
}

static uint64_t getenv_u64(const char *name, uint64_t fallback) {
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) return fallback;
    return (uint64_t)parsed;
}

static void ensure_dir(const char *path) {
    mkdir(path, 0755);
}

void h3d_config_init(h3d_ctx *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->bind_addr, sizeof(ctx->bind_addr), "127.0.0.1");
    ctx->port = H3D_DEFAULT_PORT;
    ctx->quota_bytes = 20ULL * 1024 * 1024 * 1024;
    ctx->timeout_tier1 = 1200;
    ctx->timeout_tier2 = 3600;
    ctx->timeout_tier3 = 7200;
    snprintf(ctx->log_level, sizeof(ctx->log_level), "INFO");
    ctx->max_queued_per_client = 8;
    ctx->prompt_max_chars = 8000;
    ctx->gpu_indices[0] = 0;
    ctx->gpu_count = 1;
}

int h3d_config_load(h3d_ctx *ctx) {
    const char *bind = getenv("H3D_BIND");
    if (bind && *bind) snprintf(ctx->bind_addr, sizeof(ctx->bind_addr), "%s", bind);

    ctx->port = getenv_int("H3D_PORT", H3D_DEFAULT_PORT);

    const char *model = getenv("H3D_MODEL_PATH");
    if (model && *model)
        snprintf(ctx->model_path, sizeof(ctx->model_path), "%s", model);

    const char *output = getenv_or("H3D_OUTPUT_ROOT", "~/.h3d/outputs");
    snprintf(ctx->output_root, sizeof(ctx->output_root), "%s", output);

    const char *media = getenv_or("H3D_MEDIA_ROOT", "~/.h3d/media");
    snprintf(ctx->media_root, sizeof(ctx->media_root), "%s", media);

    ctx->quota_bytes = getenv_u64("H3D_QUOTA_BYTES", ctx->quota_bytes);
    ctx->timeout_tier1 = getenv_int("H3D_TIMEOUT_TIER1", ctx->timeout_tier1);
    ctx->timeout_tier2 = getenv_int("H3D_TIMEOUT_TIER2", ctx->timeout_tier2);
    ctx->timeout_tier3 = getenv_int("H3D_TIMEOUT_TIER3", ctx->timeout_tier3);

    const char *log_level = getenv("H3D_LOG_LEVEL");
    if (log_level && *log_level)
        snprintf(ctx->log_level, sizeof(ctx->log_level), "%s", log_level);

    const char *log_dir = getenv_or("H3D_LOG_DIR", "~/.h3d/logs");
    snprintf(ctx->log_dir, sizeof(ctx->log_dir), "%s", log_dir);

    ctx->max_queued_per_client = getenv_int("H3D_MAX_QUEUED", ctx->max_queued_per_client);
    ctx->prompt_max_chars = getenv_int("H3D_PROMPT_MAX_CHARS", ctx->prompt_max_chars);

    /* Parse GPU list */
    const char *gpus = getenv_or("H3D_GPUS", "0");
    ctx->gpu_count = 0;
    const char *p = gpus;
    while (*p && ctx->gpu_count < 16) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        char *end = NULL;
        long val = strtol(p, &end, 10);
        if (end == p) break;
        ctx->gpu_indices[ctx->gpu_count++] = (int)val;
        p = end;
    }
    if (ctx->gpu_count == 0) {
        ctx->gpu_indices[0] = 0;
        ctx->gpu_count = 1;
    }

    /* Expand ~ in paths */
    const char *home = getenv("HOME");
    if (home) {
        char expanded[1024];
        if (ctx->output_root[0] == '~') {
            snprintf(expanded, sizeof(expanded), "%s%s", home, ctx->output_root + 1);
            snprintf(ctx->output_root, sizeof(ctx->output_root), "%s", expanded);
        }
        if (ctx->media_root[0] == '~') {
            snprintf(expanded, sizeof(expanded), "%s%s", home, ctx->media_root + 1);
            snprintf(ctx->media_root, sizeof(ctx->media_root), "%s", expanded);
        }
        if (ctx->log_dir[0] == '~') {
            snprintf(expanded, sizeof(expanded), "%s%s", home, ctx->log_dir + 1);
            snprintf(ctx->log_dir, sizeof(ctx->log_dir), "%s", expanded);
        }
    }

    /* Ensure directories exist */
    ensure_dir(ctx->output_root);
    ensure_dir(ctx->media_root);
    ensure_dir(ctx->log_dir);

    return 0;
}

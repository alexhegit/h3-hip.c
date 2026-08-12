#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size) {
    (void)tokenizer_json;
    if (error && error_size) {
        snprintf(error, error_size,
                 "HIP build tokenizer is not implemented yet");
    }
    return NULL;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    free(tokenizer);
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int add_special, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    (void)tokenizer;
    (void)utf8;
    (void)add_special;
    (void)ids;
    (void)count;
    if (error && error_size) {
        snprintf(error, error_size,
                 "HIP build tokenizer is not implemented yet");
    }
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    (void)tokenizer;
    (void)ids;
    (void)count;
    if (error && error_size) {
        snprintf(error, error_size,
                 "HIP build tokenizer is not implemented yet");
    }
    return NULL;
}

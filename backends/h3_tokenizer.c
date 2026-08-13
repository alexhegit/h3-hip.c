#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

typedef struct {
    uint32_t value;
    size_t byte_offset;
    size_t byte_length;
} h3_codepoint;

typedef struct h3_tok_entry {
    char *key;
    uint32_t value;
    struct h3_tok_entry *next;
} h3_tok_entry;

typedef struct {
    h3_tok_entry **buckets;
    size_t bucket_count;
} h3_tok_map;

typedef struct h3_tok_ids {
    h3_tok_entry header;
    uint32_t *values;
    size_t count;
} h3_tok_ids;

typedef struct {
    h3_tok_map vocab;
    h3_tok_map merge_ranks;
    h3_tok_map added_tokens;
    h3_tok_map bpe_cache;
    char **inverse_vocab;
    char **inverse_added;
    size_t inverse_capacity;
    char **added_alternatives;
    size_t added_count;
    char *byte_encoder[256];
    int16_t byte_decoder[324];
} h3_tokenizer_impl;

struct h3_tokenizer {
    h3_tokenizer_impl impl;
};

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
} h3_json_cursor;

static void h3_tok_set_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
}

static void h3_json_ws(h3_json_cursor *cursor) {
    while (cursor->at < cursor->end &&
           (*cursor->at == ' ' || *cursor->at == '\n' ||
            *cursor->at == '\r' || *cursor->at == '\t')) {
        cursor->at++;
    }
}

static int h3_json_fail(h3_json_cursor *cursor, const char *message) {
    h3_tok_set_error(cursor->error, cursor->error_size, message);
    return 0;
}

static int h3_json_take(h3_json_cursor *cursor, char expected) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != expected) {
        return h3_json_fail(cursor, "malformed tokenizer JSON");
    }
    cursor->at++;
    return 1;
}

static int h3_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static void h3_append_utf8(char *out, int32_t *length, int32_t capacity,
                           UChar32 codepoint) {
    if (*length >= capacity || codepoint < 0 || codepoint > 0x10ffff) return;
    if (codepoint <= 0x7f) {
        out[(*length)++] = (char)codepoint;
    } else if (codepoint <= 0x7ff) {
        out[(*length)++] = (char)(0xc0 | (codepoint >> 6));
        out[(*length)++] = (char)(0x80 | (codepoint & 0x3f));
    } else if (codepoint <= 0xffff) {
        out[(*length)++] = (char)(0xe0 | (codepoint >> 12));
        out[(*length)++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        out[(*length)++] = (char)(0x80 | (codepoint & 0x3f));
    } else {
        out[(*length)++] = (char)(0xf0 | (codepoint >> 18));
        out[(*length)++] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
        out[(*length)++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        out[(*length)++] = (char)(0x80 | (codepoint & 0x3f));
    }
}

static UChar32 h3_next_utf8(const char *text, int32_t *offset, int32_t length) {
    const unsigned char *bytes = (const unsigned char *)text;
    int32_t index = *offset;
    if (index >= length) return -1;
    unsigned char lead = bytes[index++];
    UChar32 codepoint = 0;
    if (lead < 0x80) {
        codepoint = lead;
    } else if ((lead >> 5) == 0x6 && index < length) {
        codepoint = ((UChar32)(lead & 0x1f) << 6) |
                    (bytes[index++] & 0x3f);
    } else if ((lead >> 4) == 0xe && index + 1 < length) {
        codepoint = ((UChar32)(lead & 0x0f) << 12) |
                    ((UChar32)(bytes[index++] & 0x3f) << 6) |
                    (bytes[index++] & 0x3f);
    } else if ((lead >> 3) == 0x1e && index + 2 < length) {
        codepoint = ((UChar32)(lead & 0x07) << 18) |
                    ((UChar32)(bytes[index++] & 0x3f) << 12) |
                    ((UChar32)(bytes[index++] & 0x3f) << 6) |
                    (bytes[index++] & 0x3f);
    } else {
        return -1;
    }
    *offset = index;
    return codepoint;
}

static char *h3_json_string(h3_json_cursor *cursor) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        h3_json_fail(cursor, "expected JSON string");
        return NULL;
    }
    cursor->at++;
    size_t maximum = (size_t)(cursor->end - cursor->at);
    char *result = malloc(maximum + 1);
    if (!result) {
        h3_json_fail(cursor, "out of memory parsing tokenizer JSON");
        return NULL;
    }
    size_t length = 0;
    while (cursor->at < cursor->end && *cursor->at != '"') {
        unsigned char value = (unsigned char)*cursor->at++;
        if (value == '\\') {
            if (cursor->at >= cursor->end) goto malformed;
            value = (unsigned char)*cursor->at++;
            switch (value) {
                case '"': result[length++] = '"'; break;
                case '\\': result[length++] = '\\'; break;
                case '/': result[length++] = '/'; break;
                case 'b': result[length++] = '\b'; break;
                case 'f': result[length++] = '\f'; break;
                case 'n': result[length++] = '\n'; break;
                case 'r': result[length++] = '\r'; break;
                case 't': result[length++] = '\t'; break;
                case 'u': {
                    if (cursor->end - cursor->at < 4) goto malformed;
                    int codepoint = 0;
                    for (int index = 0; index < 4; index++) {
                        int digit = h3_hex(cursor->at[index]);
                        if (digit < 0) goto malformed;
                        codepoint = codepoint * 16 + digit;
                    }
                    cursor->at += 4;
                    UChar32 cp = (UChar32)codepoint;
                    char utf8[4];
                    int32_t units = 0;
                    h3_append_utf8(utf8, &units, 4, cp);
                    if (units < 0) goto malformed;
                    memcpy(result + length, utf8, (size_t)units);
                    length += (size_t)units;
                    break;
                }
                default: goto malformed;
            }
        } else {
            if (value < 0x20) goto malformed;
            result[length++] = (char)value;
        }
    }
    if (cursor->at >= cursor->end) goto malformed;
    cursor->at++;
    result[length] = '\0';
    return result;

malformed:
    free(result);
    h3_json_fail(cursor, "malformed JSON string escape");
    return NULL;
}

static int h3_json_uint(h3_json_cursor *cursor, uint64_t *result) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at < '0' || *cursor->at > '9') {
        return h3_json_fail(cursor, "expected unsigned JSON integer");
    }
    uint64_t value = 0;
    while (cursor->at < cursor->end && *cursor->at >= '0' && *cursor->at <= '9') {
        unsigned digit = (unsigned)(*cursor->at - '0');
        if (value > (UINT64_MAX - digit) / 10) {
            return h3_json_fail(cursor, "integer overflow in tokenizer JSON");
        }
        value = value * 10 + digit;
        cursor->at++;
    }
    *result = value;
    return 1;
}

static int h3_json_skip(h3_json_cursor *cursor);

static int h3_json_skip_compound(h3_json_cursor *cursor, char open, char close) {
    if (!h3_json_take(cursor, open)) return 0;
    h3_json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == close) {
        cursor->at++;
        return 1;
    }
    for (;;) {
        if (open == '{') {
            char *key = h3_json_string(cursor);
            if (!key) return 0;
            free(key);
            if (!h3_json_take(cursor, ':')) return 0;
        }
        if (!h3_json_skip(cursor)) return 0;
        h3_json_ws(cursor);
        if (cursor->at >= cursor->end) return h3_json_fail(cursor, "unterminated JSON value");
        if (*cursor->at == close) {
            cursor->at++;
            return 1;
        }
        if (*cursor->at++ != ',') return h3_json_fail(cursor, "expected JSON comma");
    }
}

static int h3_json_skip(h3_json_cursor *cursor) {
    h3_json_ws(cursor);
    if (cursor->at >= cursor->end) return h3_json_fail(cursor, "missing JSON value");
    if (*cursor->at == '"') {
        char *value = h3_json_string(cursor);
        if (!value) return 0;
        free(value);
        return 1;
    }
    if (*cursor->at == '{') return h3_json_skip_compound(cursor, '{', '}');
    if (*cursor->at == '[') return h3_json_skip_compound(cursor, '[', ']');
    const char *start = cursor->at;
    while (cursor->at < cursor->end && *cursor->at != ',' &&
           *cursor->at != '}' && *cursor->at != ']' &&
           *cursor->at != ' ' && *cursor->at != '\n' &&
           *cursor->at != '\r' && *cursor->at != '\t') {
        cursor->at++;
    }
    if (cursor->at == start) return h3_json_fail(cursor, "invalid JSON scalar");
    return 1;
}

static const char *h3_json_find_object_key(const char *json, const char *end,
                                           const char *object_key) {
    size_t key_len = strlen(object_key);
    char pattern[256];
    if (key_len + 4 >= sizeof(pattern)) return NULL;
    snprintf(pattern, sizeof(pattern), "\"%s\"", object_key);
    for (const char *cursor = json; cursor + key_len + 2 < end; cursor++) {
        if (memcmp(cursor, pattern, key_len + 2) != 0) continue;
        const char *after = cursor + key_len + 2;
        while (after < end && (*after == ' ' || *after == '\n' || *after == '\r' ||
                              *after == '\t')) {
            after++;
        }
        if (after < end && *after == ':') return after + 1;
    }
    return NULL;
}

static unsigned h3_tok_hash(const char *key, size_t bucket_count) {
    unsigned hash = 2166136261u;
    for (const unsigned char *cursor = (const unsigned char *)key; *cursor; cursor++) {
        hash ^= *cursor;
        hash *= 16777619u;
    }
    return hash % (unsigned)bucket_count;
}

static void h3_tok_map_init(h3_tok_map *map, size_t bucket_count) {
    map->bucket_count = bucket_count;
    map->buckets = calloc(bucket_count, sizeof(*map->buckets));
}

static void h3_tok_map_free(h3_tok_map *map) {
    if (!map->buckets) return;
    for (size_t index = 0; index < map->bucket_count; index++) {
        h3_tok_entry *entry = map->buckets[index];
        while (entry) {
            h3_tok_entry *next = entry->next;
            free(entry->key);
            if (entry->value == UINT32_MAX) {
                h3_tok_ids *ids = (h3_tok_ids *)(void *)entry;
                free(ids->values);
            }
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    map->buckets = NULL;
    map->bucket_count = 0;
}

static int h3_tok_map_put(h3_tok_map *map, const char *key, uint32_t value) {
    unsigned bucket = h3_tok_hash(key, map->bucket_count);
    for (h3_tok_entry *entry = map->buckets[bucket]; entry; entry = entry->next) {
        if (!strcmp(entry->key, key)) {
            entry->value = value;
            return 1;
        }
    }
    h3_tok_entry *entry = malloc(sizeof(*entry));
    if (!entry) return 0;
    entry->key = strdup(key);
    if (!entry->key) {
        free(entry);
        return 0;
    }
    entry->value = value;
    entry->next = map->buckets[bucket];
    map->buckets[bucket] = entry;
    return 1;
}

static int h3_tok_map_get(const h3_tok_map *map, const char *key, uint32_t *value) {
    if (!map->buckets) return 0;
    unsigned bucket = h3_tok_hash(key, map->bucket_count);
    for (h3_tok_entry *entry = map->buckets[bucket]; entry; entry = entry->next) {
        if (!strcmp(entry->key, key)) {
            *value = entry->value;
            return 1;
        }
    }
    return 0;
}

static int h3_tok_map_put_ids(h3_tok_map *map, const char *key,
                              const uint32_t *values, size_t count) {
    unsigned bucket = h3_tok_hash(key, map->bucket_count);
    for (h3_tok_entry *entry = map->buckets[bucket]; entry; entry = entry->next) {
        if (!strcmp(entry->key, key)) {
            h3_tok_ids *ids = (h3_tok_ids *)(void *)entry;
            free(ids->values);
            ids->values = malloc(count * sizeof(*ids->values));
            if (!ids->values) return 0;
            memcpy(ids->values, values, count * sizeof(*values));
            ids->count = count;
            return 1;
        }
    }
    h3_tok_ids *ids = calloc(1, sizeof(*ids));
    if (!ids) return 0;
    ids->values = malloc(count * sizeof(*ids->values));
    if (!ids->values) {
        free(ids);
        return 0;
    }
    memcpy(ids->values, values, count * sizeof(*values));
    ids->count = count;
    ids->header.value = UINT32_MAX;
    ids->header.key = strdup(key);
    if (!ids->header.key) {
        free(ids->values);
        free(ids);
        return 0;
    }
    ids->header.next = map->buckets[bucket];
    map->buckets[bucket] = &ids->header;
    return 1;
}

static const h3_tok_ids *h3_tok_map_get_ids(const h3_tok_map *map, const char *key) {
    if (!map->buckets) return NULL;
    unsigned bucket = h3_tok_hash(key, map->bucket_count);
    for (h3_tok_entry *entry = map->buckets[bucket]; entry; entry = entry->next) {
        if (!strcmp(entry->key, key) && entry->value == UINT32_MAX) {
            return (const h3_tok_ids *)(const void *)entry;
        }
    }
    return NULL;
}

static char *h3_codepoint_string(uint32_t value) {
    char buffer[5];
    int32_t length = 0;
    h3_append_utf8(buffer, &length, 5, (UChar32)value);
    if (length < 0) return NULL;
    buffer[length] = '\0';
    return strdup(buffer);
}

static int h3_letter(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
           category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
           category == U_OTHER_LETTER;
}

static int h3_number(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
           category == U_OTHER_NUMBER;
}

static int h3_space(uint32_t value) {
    return u_isUWhiteSpace((UChar32)value) || (value >= 0x1c && value <= 0x1f);
}

static h3_codepoint *h3_codepoints(const char *text, size_t text_length,
                                   size_t *count) {
    size_t capacity = text_length ? text_length : 1;
    h3_codepoint *points = calloc(capacity, sizeof(*points));
    if (!points) return NULL;
    size_t used = 0;
    int32_t offset = 0;
    while (offset < (int32_t)text_length) {
        int32_t start = offset;
        UChar32 value = h3_next_utf8(text, &offset, (int32_t)text_length);
        if (value < 0) {
            free(points);
            return NULL;
        }
        if (used == capacity) {
            capacity *= 2;
            h3_codepoint *grown = realloc(points, capacity * sizeof(*points));
            if (!grown) {
                free(points);
                return NULL;
            }
            points = grown;
        }
        points[used++] = (h3_codepoint){(uint32_t)value, (size_t)start,
                                        (size_t)(offset - start)};
    }
    *count = used;
    return points;
}

static char *h3_slice(const char *text, const h3_codepoint *points,
                        size_t start, size_t stop) {
    size_t byte_start = points[start].byte_offset;
    size_t byte_end = points[stop - 1].byte_offset + points[stop - 1].byte_length;
    size_t length = byte_end - byte_start;
    char *result = malloc(length + 1);
    if (!result) return NULL;
    memcpy(result, text + byte_start, length);
    result[length] = '\0';
    return result;
}

static size_t h3_contraction(const h3_codepoint *points, size_t count,
                             size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} h3_string_list;

static int h3_string_list_push(h3_string_list *list, char *item) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 8;
        char **items = realloc(list->items, capacity * sizeof(*items));
        if (!items) return 0;
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = item;
    return 1;
}

static void h3_string_list_free(h3_string_list *list) {
    for (size_t index = 0; index < list->count; index++) {
        free(list->items[index]);
        list->items[index] = NULL;
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

static char **h3_pretokenize(const char *text, size_t *piece_count) {
    h3_string_list pieces = {0};
    size_t count = 0;
    h3_codepoint *points = h3_codepoints(text, strlen(text), &count);
    if (!points) return NULL;
    size_t index = 0;
    while (index < count) {
        size_t contraction = h3_contraction(points, count, index);
        if (contraction) {
            char *piece = h3_slice(text, points, index, index + contraction);
            if (!piece || !h3_string_list_push(&pieces, piece)) goto fail;
            index += contraction;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (h3_letter(value)) {
            /* already at first letter */
        } else if (value != '\r' && value != '\n' && !h3_number(value) &&
                   index + 1 < count && h3_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < count && h3_letter(points[stop].value)) stop++;
            char *piece = h3_slice(text, points, index, stop);
            if (!piece || !h3_string_list_push(&pieces, piece)) goto fail;
            index = stop;
            continue;
        }
        if (h3_number(value)) {
            char *piece = h3_slice(text, points, index, index + 1);
            if (!piece || !h3_string_list_push(&pieces, piece)) goto fail;
            index++;
            continue;
        }
        size_t punct_start = index +
            (value == ' ' && index + 1 < count &&
             !h3_space(points[index + 1].value) &&
             !h3_letter(points[index + 1].value) &&
             !h3_number(points[index + 1].value));
        size_t stop = punct_start;
        while (stop < count && !h3_space(points[stop].value) &&
               !h3_letter(points[stop].value) &&
               !h3_number(points[stop].value)) {
            stop++;
        }
        if (stop > punct_start) {
            while (stop < count &&
                   (points[stop].value == '\r' || points[stop].value == '\n')) {
                stop++;
            }
            char *piece = h3_slice(text, points, index, stop);
            if (!piece || !h3_string_list_push(&pieces, piece)) goto fail;
            index = stop;
            continue;
        }
        if (h3_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < count &&
                   h3_space(points[whitespace_end].value)) {
                whitespace_end++;
            }
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++) {
                if (points[cursor].value == '\r' || points[cursor].value == '\n') {
                    newline_end = (ptrdiff_t)cursor + 1;
                }
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == count) piece_end = whitespace_end;
            else if (whitespace_end - index > 1) piece_end = whitespace_end - 1;
            else piece_end = index + 1;
            char *piece = h3_slice(text, points, index, piece_end);
            if (!piece || !h3_string_list_push(&pieces, piece)) goto fail;
            index = piece_end;
            continue;
        }
        goto fail;
    }
    free(points);
    *piece_count = pieces.count;
    return pieces.items;

fail:
    free(points);
    h3_string_list_free(&pieces);
    return NULL;
}

static char *h3_pair_key(const char *left, const char *right) {
    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    char *key = malloc(left_len + right_len + 3);
    if (!key) return NULL;
    memcpy(key, left, left_len);
    key[left_len] = (char)0xff;
    key[left_len + 1] = (char)0xff;
    memcpy(key + left_len + 2, right, right_len + 1);
    return key;
}

static uint32_t *h3_bpe(h3_tokenizer_impl *tokenizer, const char *piece,
                        size_t *out_count, char *error, size_t error_size) {
    const unsigned char *bytes = (const unsigned char *)piece;
    char *encoded = NULL;
    size_t encoded_cap = strlen(piece) * 4 + 1;
    encoded = malloc(encoded_cap);
    if (!encoded) return NULL;
    size_t encoded_len = 0;
    for (size_t index = 0; bytes[index]; index++) {
        const char *symbol = tokenizer->byte_encoder[bytes[index]];
        size_t symbol_len = strlen(symbol);
        if (encoded_len + symbol_len + 1 > encoded_cap) {
            encoded_cap *= 2;
            char *grown = realloc(encoded, encoded_cap);
            if (!grown) {
                free(encoded);
                return NULL;
            }
            encoded = grown;
        }
        memcpy(encoded + encoded_len, symbol, symbol_len);
        encoded_len += symbol_len;
    }
    encoded[encoded_len] = '\0';

    const h3_tok_ids *cached = h3_tok_map_get_ids(&tokenizer->bpe_cache, encoded);
    if (cached) {
        uint32_t *copy = malloc(cached->count * sizeof(*copy));
        if (!copy) {
            free(encoded);
            return NULL;
        }
        memcpy(copy, cached->values, cached->count * sizeof(*copy));
        *out_count = cached->count;
        free(encoded);
        return copy;
    }

    h3_string_list symbols = {0};
    int32_t encoded_offset = 0;
    int32_t encoded_length = (int32_t)encoded_len;
    while (encoded_offset < encoded_length) {
        int32_t start = encoded_offset;
        if (h3_next_utf8(encoded, &encoded_offset, encoded_length) < 0) goto bpe_fail;
        char *symbol = strndup(encoded + start, (size_t)(encoded_offset - start));
        if (!symbol || !h3_string_list_push(&symbols, symbol)) {
            free(symbol);
            goto bpe_fail;
        }
    }
    while (symbols.count > 1) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = 0;
        int found = 0;
        for (size_t index = 0; index + 1 < symbols.count; index++) {
            char *pair = h3_pair_key(symbols.items[index], symbols.items[index + 1]);
            if (!pair) goto bpe_fail;
            uint32_t rank = 0;
            int has_rank = h3_tok_map_get(&tokenizer->merge_ranks, pair, &rank);
            free(pair);
            if (has_rank && (!found || rank < best_rank)) {
                best_rank = rank;
                best = index;
                found = 1;
            }
        }
        if (!found) break;
        char *left = symbols.items[best];
        char *right = symbols.items[best + 1];
        size_t merged_len = strlen(left) + strlen(right);
        char *merged = malloc(merged_len + 1);
        if (!merged) goto bpe_fail;
        memcpy(merged, left, strlen(left));
        memcpy(merged + strlen(left), right, strlen(right) + 1);
        h3_string_list next = {0};
        for (size_t index = 0; index < symbols.count;) {
            if (index + 1 < symbols.count &&
                !strcmp(symbols.items[index], left) &&
                !strcmp(symbols.items[index + 1], right)) {
                if (!h3_string_list_push(&next, merged)) {
                    free(merged);
                    h3_string_list_free(&next);
                    goto bpe_fail;
                }
                free(left);
                free(right);
                symbols.items[index] = NULL;
                symbols.items[index + 1] = NULL;
                index += 2;
            } else {
                if (!h3_string_list_push(&next, symbols.items[index])) {
                    h3_string_list_free(&next);
                    goto bpe_fail;
                }
                symbols.items[index] = NULL;
                index++;
            }
        }
        h3_string_list_free(&symbols);
        symbols = next;
    }
    uint32_t *ids = malloc(symbols.count * sizeof(*ids));
    if (!ids) goto bpe_fail;
    for (size_t index = 0; index < symbols.count; index++) {
        if (!h3_tok_map_get(&tokenizer->vocab, symbols.items[index], &ids[index])) {
            h3_tok_set_error(error, error_size, "BPE symbol is absent from vocabulary");
            free(ids);
            h3_string_list_free(&symbols);
            free(encoded);
            return NULL;
        }
    }
    h3_tok_map_put_ids(&tokenizer->bpe_cache, encoded, ids, symbols.count);
    *out_count = symbols.count;
    h3_string_list_free(&symbols);
    free(encoded);
    return ids;

bpe_fail:
    h3_string_list_free(&symbols);
    free(encoded);
    h3_tok_set_error(error, error_size, "BPE failure");
    return NULL;
}

static int h3_encode_plain(h3_tokenizer_impl *tokenizer, const char *text,
                           uint32_t **output, size_t *output_count,
                           char *error, size_t error_size) {
    size_t piece_count = 0;
    char **pieces = h3_pretokenize(text, &piece_count);
    if (!pieces && piece_count) {
        h3_tok_set_error(error, error_size, "unable to pre-tokenize input");
        return 0;
    }
    size_t capacity = 0;
    for (size_t index = 0; index < piece_count; index++) {
        size_t count = 0;
        uint32_t *ids = h3_bpe(tokenizer, pieces[index], &count, error, error_size);
        if (!ids) {
            for (size_t cleanup = 0; cleanup < piece_count; cleanup++) {
                free(pieces[cleanup]);
            }
            free(pieces);
            return 0;
        }
        if (*output_count + count > capacity) {
            capacity = capacity ? capacity * 2 : count;
            while (*output_count + count > capacity) capacity *= 2;
            uint32_t *grown = realloc(*output, capacity * sizeof(**output));
            if (!grown) {
                free(ids);
                for (size_t cleanup = 0; cleanup < piece_count; cleanup++) {
                    free(pieces[cleanup]);
                }
                free(pieces);
                free(*output);
                *output = NULL;
                h3_tok_set_error(error, error_size, "out of memory encoding prompt");
                return 0;
            }
            *output = grown;
        }
        memcpy(*output + *output_count, ids, count * sizeof(*ids));
        *output_count += count;
        free(ids);
    }
    for (size_t index = 0; index < piece_count; index++) free(pieces[index]);
    free(pieces);
    return 1;
}

static int h3_added_compare(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    size_t left_len = strlen(*a);
    size_t right_len = strlen(*b);
    if (left_len > right_len) return -1;
    if (left_len < right_len) return 1;
    return strcmp(*a, *b);
}

static int h3_added_match(h3_tokenizer_impl *tokenizer, const char *text,
                          size_t text_length, size_t start, size_t *match_start,
                          size_t *match_length, const char **token) {
    int found = 0;
    for (size_t index = 0; index < tokenizer->added_count; index++) {
        const char *candidate = tokenizer->added_alternatives[index];
        size_t candidate_len = strlen(candidate);
        if (start + candidate_len > text_length) continue;
        if (memcmp(text + start, candidate, candidate_len) != 0) continue;
        if (!found || start < *match_start ||
            (start == *match_start && candidate_len > *match_length)) {
            *match_start = start;
            *match_length = candidate_len;
            *token = candidate;
            found = 1;
        }
    }
    return found;
}

static char *h3_normalize_nfc(const char *utf8, char *error, size_t error_size) {
    if (!utf8[0]) return strdup("");
    UErrorCode status = U_ZERO_ERROR;
    const UNormalizer2 *normalizer = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status)) {
        h3_tok_set_error(error, error_size, "unable to acquire NFC normalizer");
        return NULL;
    }
    int32_t source_units = 0;
    u_strFromUTF8(NULL, 0, &source_units, utf8, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        h3_tok_set_error(error, error_size, "unable to size UTF-8 conversion");
        return NULL;
    }
    status = U_ZERO_ERROR;
    UChar *source = malloc((size_t)(source_units + 1) * sizeof(*source));
    if (!source) {
        h3_tok_set_error(error, error_size, "out of memory normalizing prompt");
        return NULL;
    }
    u_strFromUTF8(source, source_units + 1, &source_units, utf8, -1, &status);
    if (U_FAILURE(status)) {
        free(source);
        h3_tok_set_error(error, error_size, "UTF-8 conversion failed");
        return NULL;
    }
    int32_t dest_units = unorm2_normalize(normalizer, source, source_units, NULL, 0,
                                          &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        free(source);
        h3_tok_set_error(error, error_size, "unable to size NFC normalization");
        return NULL;
    }
    status = U_ZERO_ERROR;
    UChar *dest = malloc((size_t)(dest_units + 1) * sizeof(*dest));
    if (!dest) {
        free(source);
        h3_tok_set_error(error, error_size, "out of memory normalizing prompt");
        return NULL;
    }
    dest_units = unorm2_normalize(normalizer, source, source_units, dest, dest_units,
                                &status);
    free(source);
    if (U_FAILURE(status)) {
        free(dest);
        h3_tok_set_error(error, error_size, "NFC normalization failed");
        return NULL;
    }
    int32_t utf8_capacity = 0;
    u_strToUTF8(NULL, 0, &utf8_capacity, dest, dest_units, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        free(dest);
        h3_tok_set_error(error, error_size, "unable to size normalized UTF-8");
        return NULL;
    }
    status = U_ZERO_ERROR;
    char *result = malloc((size_t)utf8_capacity + 1);
    if (!result) {
        free(dest);
        h3_tok_set_error(error, error_size, "out of memory normalizing prompt");
        return NULL;
    }
    u_strToUTF8(result, utf8_capacity + 1, &utf8_capacity, dest, dest_units, &status);
    free(dest);
    if (U_FAILURE(status)) {
        free(result);
        h3_tok_set_error(error, error_size, "normalized UTF-8 conversion failed");
        return NULL;
    }
    result[utf8_capacity] = '\0';
    return result;
}

static void h3_tokenizer_impl_free(h3_tokenizer_impl *impl) {
    h3_tok_map_free(&impl->vocab);
    h3_tok_map_free(&impl->merge_ranks);
    h3_tok_map_free(&impl->added_tokens);
    h3_tok_map_free(&impl->bpe_cache);
    if (impl->inverse_vocab) {
        for (size_t index = 0; index < impl->inverse_capacity; index++) {
            free(impl->inverse_vocab[index]);
        }
        free(impl->inverse_vocab);
    }
    if (impl->inverse_added) {
        for (size_t index = 0; index < impl->inverse_capacity; index++) {
            free(impl->inverse_added[index]);
        }
        free(impl->inverse_added);
    }
    for (size_t index = 0; index < impl->added_count; index++) {
        free(impl->added_alternatives[index]);
    }
    free(impl->added_alternatives);
    for (size_t index = 0; index < 256; index++) free(impl->byte_encoder[index]);
}

static int h3_inverse_set(char **table, size_t capacity, uint32_t id,
                          const char *value) {
    if ((size_t)id >= capacity) return 0;
    free(table[id]);
    table[id] = strdup(value);
    return table[id] != NULL;
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        h3_tok_set_error(error, error_size, "tokenizer path is required");
        return NULL;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        h3_tok_set_error(error, error_size, "cannot read tokenizer");
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        h3_tok_set_error(error, error_size, "cannot seek tokenizer");
        return NULL;
    }
    long end = ftell(file);
    if (end < 0) {
        fclose(file);
        h3_tok_set_error(error, error_size, "invalid tokenizer size");
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        h3_tok_set_error(error, error_size, "cannot rewind tokenizer");
        return NULL;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fclose(file);
        free(json);
        h3_tok_set_error(error, error_size, "cannot read tokenizer JSON");
        return NULL;
    }
    fclose(file);
    json[end] = '\0';

    const char *model = h3_json_find_object_key(json, json + end, "model");
    const char *normalizer = h3_json_find_object_key(json, json + end, "normalizer");
    if (!model || !normalizer) {
        free(json);
        h3_tok_set_error(error, error_size, "unexpected tokenizer specification");
        return NULL;
    }
    h3_json_cursor cursor = {model, json + end, error, error_size};
    if (!h3_json_take(&cursor, '{')) {
        free(json);
        return NULL;
    }
    char *model_type = NULL;
    for (;;) {
        char *key = h3_json_string(&cursor);
        if (!key) {
            free(json);
            return NULL;
        }
        if (!h3_json_take(&cursor, ':')) {
            free(key);
            free(json);
            return NULL;
        }
        if (!strcmp(key, "type")) {
            model_type = h3_json_string(&cursor);
            free(key);
            if (!model_type) {
                free(json);
                return NULL;
            }
        } else if (!strcmp(key, "unk_token")) {
            h3_json_ws(&cursor);
            if (cursor.at >= cursor.end ||
                strncmp(cursor.at, "null", 4) != 0) {
                free(key);
                free(model_type);
                free(json);
                h3_tok_set_error(error, error_size,
                                 "unexpected tokenizer specification");
                return NULL;
            }
            cursor.at += 4;
            free(key);
        } else {
            if (!h3_json_skip(&cursor)) {
                free(key);
                free(model_type);
                free(json);
                return NULL;
            }
            free(key);
        }
        h3_json_ws(&cursor);
        if (cursor.at < cursor.end && *cursor.at == '}') {
            cursor.at++;
            break;
        }
        if (*cursor.at++ != ',') {
            free(model_type);
            free(json);
            h3_tok_set_error(error, error_size, "malformed tokenizer model");
            return NULL;
        }
    }
    if (!model_type || strcmp(model_type, "BPE")) {
        free(model_type);
        free(json);
        h3_tok_set_error(error, error_size, "unexpected tokenizer specification");
        return NULL;
    }
    free(model_type);

    cursor = (h3_json_cursor){normalizer, json + end, error, error_size};
    if (!h3_json_take(&cursor, '{')) {
        free(json);
        return NULL;
    }
    char *normalizer_type = NULL;
    for (;;) {
        char *key = h3_json_string(&cursor);
        if (!key) {
            free(json);
            return NULL;
        }
        if (!h3_json_take(&cursor, ':')) {
            free(key);
            free(json);
            return NULL;
        }
        if (!strcmp(key, "type")) {
            normalizer_type = h3_json_string(&cursor);
        } else if (!h3_json_skip(&cursor)) {
            free(key);
            free(normalizer_type);
            free(json);
            return NULL;
        }
        free(key);
        h3_json_ws(&cursor);
        if (cursor.at < cursor.end && *cursor.at == '}') {
            cursor.at++;
            break;
        }
        if (*cursor.at++ != ',') {
            free(normalizer_type);
            free(json);
            h3_tok_set_error(error, error_size, "malformed tokenizer normalizer");
            return NULL;
        }
    }
    if (!normalizer_type || strcmp(normalizer_type, "NFC")) {
        free(normalizer_type);
        free(json);
        h3_tok_set_error(error, error_size, "unexpected tokenizer specification");
        return NULL;
    }
    free(normalizer_type);

    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        free(json);
        h3_tok_set_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    h3_tok_map_init(&tokenizer->impl.vocab, 262144);
    h3_tok_map_init(&tokenizer->impl.merge_ranks, 65536);
    h3_tok_map_init(&tokenizer->impl.added_tokens, 4096);
    h3_tok_map_init(&tokenizer->impl.bpe_cache, 4096);

    const char *vocab_at = h3_json_find_object_key(json, json + end, "vocab");
    if (!vocab_at) {
        h3_tokenizer_free(tokenizer);
        free(json);
        h3_tok_set_error(error, error_size, "missing tokenizer vocabulary");
        return NULL;
    }
    cursor = (h3_json_cursor){vocab_at, json + end, error, error_size};
    if (!h3_json_take(&cursor, '{')) goto load_fail;
    uint32_t maximum_id = 0;
    h3_json_ws(&cursor);
    while (cursor.at < cursor.end && *cursor.at != '}') {
        h3_json_ws(&cursor);
        char *symbol = h3_json_string(&cursor);
        if (!symbol) goto load_fail;
        if (!h3_json_take(&cursor, ':')) {
            free(symbol);
            goto load_fail;
        }
        uint64_t token_id = 0;
        if (!h3_json_uint(&cursor, &token_id)) {
            free(symbol);
            goto load_fail;
        }
        if (token_id > UINT32_MAX) {
            free(symbol);
            h3_tok_set_error(error, error_size, "token ID overflow");
            goto load_fail;
        }
        if (!h3_tok_map_put(&tokenizer->impl.vocab, symbol, (uint32_t)token_id)) {
            free(symbol);
            h3_tok_set_error(error, error_size, "out of memory loading vocabulary");
            goto load_fail;
        }
        maximum_id = maximum_id > (uint32_t)token_id ? maximum_id : (uint32_t)token_id;
        free(symbol);
        h3_json_ws(&cursor);
        if (cursor.at < cursor.end && *cursor.at == ',') cursor.at++;
    }
    if (!h3_json_take(&cursor, '}')) goto load_fail;

    const char *merges_at = h3_json_find_object_key(json, json + end, "merges");
    if (!merges_at) {
        h3_tok_set_error(error, error_size, "missing tokenizer merges");
        goto load_fail;
    }
    cursor = (h3_json_cursor){merges_at, json + end, error, error_size};
    if (!h3_json_take(&cursor, '[')) goto load_fail;
    uint32_t rank = 0;
    h3_json_ws(&cursor);
    while (cursor.at < cursor.end && *cursor.at != ']') {
        h3_json_ws(&cursor);
        char *left = NULL;
        char *right = NULL;
        if (*cursor.at == '"') {
            char *entry = h3_json_string(&cursor);
            if (!entry) goto load_fail;
            char *space = strchr(entry, ' ');
            if (!space) {
                free(entry);
                h3_tok_set_error(error, error_size, "invalid tokenizer merge");
                goto load_fail;
            }
            *space = '\0';
            left = entry;
            right = strdup(space + 1);
            if (!right) {
                free(left);
                goto load_fail;
            }
        } else if (*cursor.at == '[') {
            if (!h3_json_take(&cursor, '[')) goto load_fail;
            left = h3_json_string(&cursor);
            if (!left) goto load_fail;
            if (!h3_json_take(&cursor, ',')) {
                free(left);
                goto load_fail;
            }
            right = h3_json_string(&cursor);
            if (!right) {
                free(left);
                goto load_fail;
            }
            if (!h3_json_take(&cursor, ']')) {
                free(left);
                free(right);
                goto load_fail;
            }
        } else {
            h3_tok_set_error(error, error_size, "invalid tokenizer merge");
            goto load_fail;
        }
        char *pair = h3_pair_key(left, right);
        if (!pair || !h3_tok_map_put(&tokenizer->impl.merge_ranks, pair, rank++)) {
            free(pair);
            free(left);
            free(right);
            h3_tok_set_error(error, error_size, "out of memory loading merges");
            goto load_fail;
        }
        free(pair);
        free(left);
        free(right);
        h3_json_ws(&cursor);
        if (cursor.at < cursor.end && *cursor.at == ',') cursor.at++;
    }
    if (!h3_json_take(&cursor, ']')) goto load_fail;

    const char *added_at = h3_json_find_object_key(json, json + end, "added_tokens");
    h3_string_list added_list = {0};
    if (added_at) {
        cursor = (h3_json_cursor){added_at, json + end, error, error_size};
        if (!h3_json_take(&cursor, '[')) goto load_fail;
        h3_json_ws(&cursor);
        while (cursor.at < cursor.end && *cursor.at != ']') {
            h3_json_ws(&cursor);
            if (!h3_json_take(&cursor, '{')) goto load_fail;
            char *content = NULL;
            uint64_t identifier = 0;
            for (;;) {
                char *key = h3_json_string(&cursor);
                if (!key) goto load_fail;
                if (!h3_json_take(&cursor, ':')) {
                    free(key);
                    goto load_fail;
                }
                if (!strcmp(key, "content")) {
                    content = h3_json_string(&cursor);
                } else if (!strcmp(key, "id")) {
                    if (!h3_json_uint(&cursor, &identifier)) {
                        free(key);
                        goto load_fail;
                    }
                } else if (!strcmp(key, "single_word") || !strcmp(key, "lstrip") ||
                           !strcmp(key, "rstrip") || !strcmp(key, "normalized")) {
                    h3_json_ws(&cursor);
                    if (cursor.at < cursor.end &&
                        strncmp(cursor.at, "true", 4) == 0) {
                        free(key);
                        free(content);
                        h3_tok_set_error(error, error_size,
                                         "unsupported added-token policy");
                        goto load_fail;
                    }
                    if (!h3_json_skip(&cursor)) {
                        free(key);
                        free(content);
                        goto load_fail;
                    }
                } else if (!h3_json_skip(&cursor)) {
                    free(key);
                    free(content);
                    goto load_fail;
                }
                free(key);
                h3_json_ws(&cursor);
                if (cursor.at < cursor.end && *cursor.at == '}') {
                    cursor.at++;
                    break;
                }
                if (*cursor.at++ != ',') goto load_fail;
            }
            if (!content || identifier > UINT32_MAX) goto load_fail;
            if (!h3_tok_map_put(&tokenizer->impl.added_tokens, content,
                                (uint32_t)identifier)) {
                free(content);
                h3_tok_set_error(error, error_size, "out of memory loading added tokens");
                goto load_fail;
            }
            maximum_id = maximum_id > (uint32_t)identifier ? maximum_id
                                                           : (uint32_t)identifier;
            char *copy = strdup(content);
            free(content);
            if (!copy || !h3_string_list_push(&added_list, copy)) {
                free(copy);
                goto load_fail;
            }
            h3_json_ws(&cursor);
            if (cursor.at < cursor.end && *cursor.at == ',') cursor.at++;
        }
        if (!h3_json_take(&cursor, ']')) goto load_fail;
    }
    tokenizer->impl.added_alternatives = added_list.items;
    tokenizer->impl.added_count = added_list.count;

    if (tokenizer->impl.added_count > 1) {
        qsort(tokenizer->impl.added_alternatives, tokenizer->impl.added_count,
              sizeof(*tokenizer->impl.added_alternatives), h3_added_compare);
    }

    tokenizer->impl.inverse_capacity = (size_t)maximum_id + 1;
    tokenizer->impl.inverse_vocab = calloc(tokenizer->impl.inverse_capacity,
                                           sizeof(*tokenizer->impl.inverse_vocab));
    tokenizer->impl.inverse_added = calloc(tokenizer->impl.inverse_capacity,
                                           sizeof(*tokenizer->impl.inverse_added));
    if (!tokenizer->impl.inverse_vocab || !tokenizer->impl.inverse_added) {
        h3_tok_set_error(error, error_size, "out of memory loading vocabulary");
        goto load_fail;
    }
    for (size_t bucket = 0; bucket < tokenizer->impl.vocab.bucket_count; bucket++) {
        for (h3_tok_entry *entry = tokenizer->impl.vocab.buckets[bucket]; entry;
             entry = entry->next) {
            if (!h3_inverse_set(tokenizer->impl.inverse_vocab,
                                tokenizer->impl.inverse_capacity,
                                entry->value, entry->key)) {
                h3_tok_set_error(error, error_size, "out of memory loading vocabulary");
                goto load_fail;
            }
        }
    }
    for (size_t bucket = 0; bucket < tokenizer->impl.added_tokens.bucket_count;
         bucket++) {
        for (h3_tok_entry *entry = tokenizer->impl.added_tokens.buckets[bucket];
             entry; entry = entry->next) {
            if (!h3_inverse_set(tokenizer->impl.inverse_added,
                                tokenizer->impl.inverse_capacity,
                                entry->value, entry->key)) {
                h3_tok_set_error(error, error_size,
                                 "out of memory loading added tokens");
                goto load_fail;
            }
        }
    }

    for (size_t index = 0; index < 324; index++) tokenizer->impl.byte_decoder[index] = -1;
    unsigned extra = 0;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t codepoint = visible ? byte : 256 + extra++;
        tokenizer->impl.byte_encoder[byte] = h3_codepoint_string(codepoint);
        if (!tokenizer->impl.byte_encoder[byte]) goto load_fail;
        tokenizer->impl.byte_decoder[codepoint] = (int16_t)byte;
    }

    free(json);
    return tokenizer;

load_fail:
    h3_tokenizer_free(tokenizer);
    free(json);
    return NULL;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    h3_tokenizer_impl_free(&tokenizer->impl);
    free(tokenizer);
}

static int h3_output_push(uint32_t **output, size_t *count, size_t *capacity,
                          uint32_t value, char *error, size_t error_size) {
    if (*count + 1 > *capacity) {
        size_t new_capacity = *capacity ? *capacity * 2 : 8;
        while (*count + 1 > new_capacity) new_capacity *= 2;
        uint32_t *grown = realloc(*output, new_capacity * sizeof(**output));
        if (!grown) {
            h3_tok_set_error(error, error_size, "out of memory encoding prompt");
            return 0;
        }
        *output = grown;
        *capacity = new_capacity;
    }
    (*output)[(*count)++] = value;
    return 1;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    h3_tokenizer_impl *impl = (h3_tokenizer_impl *)&tokenizer->impl;
    char *text = h3_normalize_nfc(utf8, error, error_size);
    if (!text) return 0;
    size_t text_length = strlen(text);
    uint32_t *output = NULL;
    size_t output_count = 0;
    size_t output_capacity = 0;
    size_t start = 0;
    while (start < text_length) {
        size_t match_start = 0;
        size_t match_length = 0;
        const char *added = NULL;
        if (!h3_added_match(impl, text, text_length, start,
                            &match_start, &match_length, &added)) {
            break;
        }
        if (match_start > start) {
            char *segment = strndup(text + start, match_start - start);
            if (!segment ||
                !h3_encode_plain(impl, segment, &output,
                                 &output_count, error, error_size)) {
                free(segment);
                free(text);
                free(output);
                return 0;
            }
            free(segment);
        }
        uint32_t token_id = 0;
        if (!h3_tok_map_get(&impl->added_tokens, added, &token_id)) {
            free(text);
            free(output);
            h3_tok_set_error(error, error_size, "unknown added token");
            return 0;
        }
        if (!h3_output_push(&output, &output_count, &output_capacity, token_id,
                            error, error_size)) {
            free(text);
            free(output);
            return 0;
        }
        start = match_start + match_length;
    }
    if (start < text_length &&
        !h3_encode_plain(impl, text + start, &output, &output_count,
                         error, error_size)) {
        free(text);
        free(output);
        return 0;
    }
    if (output_count == 0 && pad_empty) {
        if (!h3_output_push(&output, &output_count, &output_capacity,
                            H3_PAD_TOKEN_ID, error, error_size)) {
            free(text);
            free(output);
            return 0;
        }
    }
    free(text);
    *ids = output;
    *count = output_count;
    return 1;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer, const uint32_t *ids,
                          size_t count, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || (!ids && count)) return NULL;
    char *result = calloc(1, 1);
    if (!result) {
        h3_tok_set_error(error, error_size, "out of memory decoding tokens");
        return NULL;
    }
    size_t result_length = 0;
    unsigned char *bytes = NULL;
    size_t byte_count = 0;
    size_t byte_capacity = 0;
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if ((size_t)identifier >= tokenizer->impl.inverse_capacity) {
            free(result);
            free(bytes);
            h3_tok_set_error(error, error_size, "token ID is out of range");
            return NULL;
        }
        const char *added = tokenizer->impl.inverse_added[identifier];
        if (added) {
            if (byte_count) {
                char *chunk = malloc(byte_count + 1);
                if (!chunk) goto decode_fail;
                memcpy(chunk, bytes, byte_count);
                chunk[byte_count] = '\0';
                size_t new_length = result_length + byte_count;
                char *grown = realloc(result, new_length + 1);
                if (!grown) {
                    free(chunk);
                    goto decode_fail;
                }
                result = grown;
                memcpy(result + result_length, chunk, byte_count);
                result_length = new_length;
                result[result_length] = '\0';
                free(chunk);
                byte_count = 0;
            }
            size_t added_len = strlen(added);
            char *grown = realloc(result, result_length + added_len + 1);
            if (!grown) goto decode_fail;
            result = grown;
            memcpy(result + result_length, added, added_len);
            result_length += added_len;
            result[result_length] = '\0';
            continue;
        }
        const char *symbol = tokenizer->impl.inverse_vocab[identifier];
        if (!symbol) {
            free(result);
            free(bytes);
            h3_tok_set_error(error, error_size, "unknown token ID");
            return NULL;
        }
        int32_t offset = 0;
        int32_t symbol_length = (int32_t)strlen(symbol);
        while (offset < symbol_length) {
            UChar32 codepoint = h3_next_utf8(symbol, &offset, symbol_length);
            if (codepoint < 0 || (size_t)codepoint >= 324 ||
                tokenizer->impl.byte_decoder[codepoint] < 0) {
                free(result);
                free(bytes);
                h3_tok_set_error(error, error_size, "invalid byte-level token");
                return NULL;
            }
            unsigned char byte =
                (unsigned char)tokenizer->impl.byte_decoder[codepoint];
            if (byte_count + 1 > byte_capacity) {
                byte_capacity = byte_capacity ? byte_capacity * 2 : 16;
                unsigned char *grown = realloc(bytes, byte_capacity);
                if (!grown) goto decode_fail;
                bytes = grown;
            }
            bytes[byte_count++] = byte;
        }
    }
    if (byte_count) {
        char *chunk = malloc(byte_count + 1);
        if (!chunk) goto decode_fail;
        memcpy(chunk, bytes, byte_count);
        chunk[byte_count] = '\0';
        size_t new_length = result_length + byte_count;
        char *grown = realloc(result, new_length + 1);
        if (!grown) {
            free(chunk);
            goto decode_fail;
        }
        result = grown;
        memcpy(result + result_length, chunk, byte_count);
        result_length = new_length;
        result[result_length] = '\0';
        free(chunk);
    }
    free(bytes);
    return result;

decode_fail:
    free(result);
    free(bytes);
    h3_tok_set_error(error, error_size, "out of memory decoding tokens");
    return NULL;
}

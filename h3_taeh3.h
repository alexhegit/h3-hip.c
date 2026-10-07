#ifndef H3_TAEH3_H
#define H3_TAEH3_H

#include "h3_video_vae.h"

#include <stddef.h>

/* Decode normalized CT HW H3 latents with the TAEH3 tiny decoder.
 * Off unless --taeh3 is set. Audio decode is unchanged. */
int h3_taeh3_decode(const char *checkpoint, const float *latent_cthw,
                    int latent_t, int latent_h, int latent_w,
                    h3_video_frames *output, char *error, size_t error_size);

#endif

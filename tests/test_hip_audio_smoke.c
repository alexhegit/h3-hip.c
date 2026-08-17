#include "h3_audio_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    LATENT_LENGTH = 4,
    LATENT_CHANNELS = 32,
    STEREO = 2,
    HOP = 800
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hip_audio_smoke.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    const char *label = opaque ? (const char *)opaque : "audio";
    if (completed == 1 || completed == total)
        fprintf(stderr, "hip %s: %d/%d\n", label, completed, total);
}

static void check_finite(const float *values, size_t count, const char *label) {
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) die(label);
    }
}

static double energy(const float *values, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) sum += (double)values[i] * values[i];
    return sum / (double)count;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/home/amd/HF-MODELS/MiniMax-H3";
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", model_root);
    char error[512];

    size_t latent_count = (size_t)LATENT_CHANNELS * STEREO * LATENT_LENGTH;
    float *latent = calloc(latent_count, sizeof(*latent));
    if (!latent) die("out of memory");
    h3_audio_waveform decoded = {0};
    if (!h3_audio_vae_decode(weights, "h3_shaders.metal", latent,
                             LATENT_LENGTH, progress, "audio decode",
                             &decoded, error, sizeof(error)))
        die(error);
    if (decoded.channels != 2 || decoded.samples != LATENT_LENGTH * HOP ||
        decoded.sample_rate != 32000)
        die("audio decoder returned the wrong shape");
    size_t pcm_count = (size_t)decoded.channels * (size_t)decoded.samples;
    check_finite(decoded.pcm, pcm_count, "zero-latent PCM is non-finite");
    for (size_t i = 0; i < pcm_count; i++) {
        if (decoded.pcm[i] < -1.0f || decoded.pcm[i] > 1.0f)
            die("zero-latent PCM escaped [-1,1]");
    }
    printf("audio decode zeros: samples=%d convs=%llu submissions=%llu "
           "energy=%.4g\n",
           decoded.samples,
           (unsigned long long)decoded.gpu_stats.mps_conv_dispatches,
           (unsigned long long)decoded.gpu_stats.submissions,
           energy(decoded.pcm, pcm_count));
    if (decoded.gpu_stats.mps_conv_dispatches == 0)
        die("audio decoder reported zero convolutions");
    h3_audio_waveform_free(&decoded);

    int samples = LATENT_LENGTH * HOP;
    float *pcm = malloc((size_t)STEREO * (size_t)samples * sizeof(*pcm));
    if (!pcm) die("out of memory");
    for (int n = 0; n < samples; n++) {
        float s = 0.4f * sinf(2.0f * 3.14159265f * 440.0f * (float)n / 32000.0f);
        pcm[n] = s;
        pcm[(size_t)samples + (size_t)n] = -s;
    }
    h3_audio_latent encoded = {0};
    if (!h3_audio_vae_encode(weights, "h3_shaders.metal", pcm, samples,
                             progress, "audio encode", &encoded, error,
                             sizeof(error)))
        die(error);
    if (encoded.channels != LATENT_CHANNELS || encoded.stereo != STEREO ||
        encoded.length != LATENT_LENGTH)
        die("audio encoder returned the wrong shape");
    size_t encoded_count = (size_t)encoded.channels * (size_t)encoded.stereo *
                           (size_t)encoded.length;
    check_finite(encoded.values, encoded_count, "audio encoder non-finite");
    printf("audio encode sine: length=%d convs=%llu sdpa=%llu energy=%.4g\n",
           encoded.length,
           (unsigned long long)encoded.gpu_stats.mps_conv_dispatches,
           (unsigned long long)encoded.gpu_stats.mps_sdpa_dispatches,
           energy(encoded.values, encoded_count));
    if (encoded.gpu_stats.mps_conv_dispatches == 0)
        die("audio encoder reported zero convolutions");
    if (energy(encoded.values, encoded_count) < 1e-8)
        die("audio encoder collapsed a 440 Hz sine to ~0");

    if (!h3_audio_vae_decode(weights, "h3_shaders.metal", encoded.values,
                             encoded.length, progress, "audio roundtrip",
                             &decoded, error, sizeof(error)))
        die(error);
    check_finite(decoded.pcm, pcm_count, "roundtrip PCM is non-finite");
    double roundtrip_e = energy(decoded.pcm, pcm_count);
    printf("audio roundtrip energy=%.4g\n", roundtrip_e);
    if (roundtrip_e < 1e-8) die("audio roundtrip collapsed to silence");

    h3_audio_waveform_free(&decoded);
    h3_audio_latent_free(&encoded);
    free(latent);
    free(pcm);
    puts("ok: HIP AudioVAE encode/decode stays finite and non-collapsed");
    return 0;
}

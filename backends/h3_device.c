#include "h3_device.h"

#include <stdio.h>

#ifdef H3_METAL
#include "h3_metal.h"
#endif

int h3_device_probe(h3_device_info *info, char *error, size_t error_size) {
#ifdef H3_HIP
    return h3_hip_probe(info, error, error_size);
#elif defined(H3_METAL)
    return h3_metal_probe(info, error, error_size);
#else
    if (error && error_size) {
        snprintf(error, error_size, "no GPU backend selected at build time");
    }
    (void)info;
    return 0;
#endif
}

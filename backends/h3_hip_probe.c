#include "h3_device.h"

#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int h3_hip_device_index(void) {
    const char *value = getenv("H3_HIP_DEVICE");
    if (!value || !*value) return 0;
    int id = atoi(value);
    return id < 0 ? 0 : id;
}

int h3_hip_probe(h3_device_info *info, char *error, size_t error_size) {
    if (!info) return 0;
    memset(info, 0, sizeof(*info));

    int device_count = 0;
    hipError_t status = hipGetDeviceCount(&device_count);
    if (status != hipSuccess || device_count < 1) {
        if (error && error_size) {
            snprintf(error, error_size, "no HIP device available: %s",
                     hipGetErrorString(status));
        }
        return 0;
    }

    int device = h3_hip_device_index();
    if (device >= device_count) {
        if (error && error_size) {
            snprintf(error, error_size,
                     "H3_HIP_DEVICE=%d but only %d HIP device(s)", device,
                     device_count);
        }
        return 0;
    }

    hipDeviceProp_t props;
    status = hipGetDeviceProperties(&props, device);
    if (status != hipSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot query HIP device: %s",
                     hipGetErrorString(status));
        }
        return 0;
    }

    status = hipSetDevice(device);
    if (status != hipSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot select HIP device: %s",
                     hipGetErrorString(status));
        }
        return 0;
    }

    snprintf(info->name, sizeof(info->name), "%.127s", props.name);
    if (props.gcnArchName[0]) {
        snprintf(info->architecture, sizeof(info->architecture), "%.127s",
                 props.gcnArchName);
    } else {
        snprintf(info->architecture, sizeof(info->architecture), "unknown");
    }

    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        info->physical_memory = (uint64_t)pages * (uint64_t)page_size;
    }

    info->recommended_working_set = props.totalGlobalMem;
    info->max_buffer_length = props.totalGlobalMem;
    info->unified_memory = props.integrated ? 1 : 0;
    info->apple_gpu_family = 0;
    info->metal4 = 0;
    return 1;
}

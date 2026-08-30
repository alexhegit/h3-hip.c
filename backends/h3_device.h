#ifndef H3_DEVICE_H
#define H3_DEVICE_H

#include "h3.h"

int h3_hip_probe(h3_device_info *info, char *error, size_t error_size);
int h3_hip_device_index(void);
int h3_device_probe(h3_device_info *info, char *error, size_t error_size);

#endif

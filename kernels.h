
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

extern "C" void kernel1(uint *d_vec, uint N);
extern "C" void kernel2(uint *d_vec, uint N);
extern "C" void kernel3(uint *d_vec, uint N);
extern "C" void kernel4(uint *d_vec, uint N);
extern "C" void kernel5(uint *d_vec, uint N);
extern "C" void kernel6(uint *d_vec, uint N);
extern "C" void kernel7(uint *d_vec, uint N);
extern "C" void kernel8(uint *d_vec, uint N);
extern "C" void kernel9(uint *d_vec, uint N);
extern "C" void kernel10(uint *d_vec, uint N);

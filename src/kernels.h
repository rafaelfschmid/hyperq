
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

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 12
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 16
#endif

//extern "C" void kernel1(uint *h_a, uint *h_b, uint *h_c, uint num_threads, uint num_blocks, uint shared_size, uint computation, cudaStream_t stream);
extern "C" void kernel1(uint num_threads, uint num_blocks, uint shared_size, uint computation, cudaStream_t stream);
extern "C" void kernel2(uint *h_a, uint *h_b, uint *h_c, uint n, cudaStream_t stream);

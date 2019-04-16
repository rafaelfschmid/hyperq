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
#include <iostream>
#include "kernels.h"

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 12
#endif

void vectors_gen(uint* h_vec, int num_of_elements, int number_of_bits) {

	for (int i = 0; i < num_of_elements; i++)
	{
		h_vec[i] = rand() % number_of_bits;
	}
}

void getDeviceInformation() {
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, 0);

	if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		printf("No CUDA GPU has been detected\n");
	}
	else {
		printf("Device name:                %s\n", deviceProp.name);
		printf("Total Global Memory:        %d\n", deviceProp.regsPerMultiprocessor);
		printf("Total shared mem per block: %d\n", deviceProp.sharedMemPerMultiprocessor);
		printf("Total const mem size:       %d\n", deviceProp.maxThreadsPerMultiProcessor);
	}
}

int main(int argc, char **argv) {

	getDeviceInformation();

	uint num_of_elements=1048576;
	uint mem_size_vec = sizeof(int) * num_of_elements;
	uint *h_a = (uint *) malloc(mem_size_vec);
	uint *h_b = (uint *) malloc(mem_size_vec);
	uint *h_c = (uint *) malloc(mem_size_vec);

	srand(time(NULL));
	vectors_gen(h_b, num_of_elements, pow(2, EXP_BITS_SIZE));
	vectors_gen(h_c, num_of_elements, pow(2, EXP_BITS_SIZE));

	kernel1(h_a, h_b, h_c, num_of_elements);

	return 0;
}

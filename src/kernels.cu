#include <cuda.h>
#include "kernels.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

__global__ void kernel(uint *d_vec, uint n){
	// Get our global thread ID
	int id = blockIdx.x*blockDim.x+threadIdx.x;

	// Make sure we do not go out of bounds
	if (id < n)
		d_vec[id] = d_vec[id] + d_vec[id];
}

extern "C" void kernel1(uint *h_c, uint *h_a, uint *h_b, uint n){
	uint mem_size = sizeof(uint) * n;

	uint *d_a, *d_b, *d_c;
	cudaMalloc((void **) &d_a, mem_size);
	cudaMalloc((void **) &d_b, mem_size);
	cudaMalloc((void **) &d_c, mem_size);

	cudaMemcpy(d_b, h_vec, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_vec, mem_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n - 1) / dimBlock.x + 1);

	kernel<<<dimBlock, dimGrid>>>(d_vec, n);
}
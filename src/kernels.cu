#include <cuda.h>
#include "kernels.h"
#include <iostream>
#include <functional>
#include <iostream>
#include <list>
#include <vector>

template<typename T>
void print(T* vec, uint t) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";

}

void vectors_gen(uint* h_vec, int num_of_elements, int number_of_bits) {

	for (int i = 0; i < num_of_elements; i++) {
		h_vec[i] = rand() % number_of_bits;
	}
}

__global__ void kernel_base(uint *d_a, uint *d_b, uint *d_c, uint n,
		uint computation) {
	// Get our global thread ID
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int shared_memory[];
	//printf("id=%d ", id);

	// Make sure we do not go out of bounds
	if (id < n) {
		for (int i = 0; i < computation; i++) {
			for (int j = 0; j < computation; j++) {
				int a = d_b[id] + d_c[id];
				a *= j;
				int b = d_a[id] - d_b[id] - d_c[id];
				b *= j;
				d_a[id] = (a + b) / j;
			}
		}
	}
}

extern "C" void kernel3(uint num_threads, uint num_blocks, uint shared_size,
		uint computation, cudaStream_t stream) {

	uint num_of_elements = num_threads * num_blocks;
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_a = (uint *) malloc(mem_size_vec);
	uint *h_b = (uint *) malloc(mem_size_vec);
	uint *h_c = (uint *) malloc(mem_size_vec);

	srand (time(NULL));
	vectors_gen(h_a, num_of_elements, pow(2, EXP_BITS_SIZE));
	vectors_gen(h_b, num_of_elements, pow(2, EXP_BITS_SIZE));
	vectors_gen(h_c, num_of_elements, pow(2, EXP_BITS_SIZE));

	int n = num_threads * num_blocks;
	uint mem_size = sizeof(uint) * n;

	uint *d_a, *d_b, *d_c;
	cudaMalloc((void **) &d_a, mem_size);
	cudaMalloc((void **) &d_b, mem_size);
	cudaMalloc((void **) &d_c, mem_size);

//print(h_b, n);
	/*cudaMemcpyAsync(d_a, h_b, mem_size, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_b, h_b, mem_size, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_c, h_c, mem_size, cudaMemcpyHostToDevice, stream);*/
	cudaMemcpy(d_a, h_b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, mem_size, cudaMemcpyHostToDevice);

//dim3 dimBlock(BLOCK_SIZE);
//dim3 dimGrid((n - 1) / dimBlock.x + 1);

	kernel_base<<<num_blocks, num_threads, shared_size * (sizeof(uint)), stream>>>(
			d_a, d_b, d_c, n, computation);
	//printf("Kernel1\n");

	//cudaMemcpyAsync(h_a, d_a, mem_size, cudaMemcpyDeviceToHost, stream);
	cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);
//print(h_a, n);
}


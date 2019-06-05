#include <cuda.h>
#include "kernels.h"
#include <iostream>
#include <functional>
#include <iostream>
#include <list>
#include <vector>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

template<typename T>
void print(T* vec, uint t) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";

}

cudaStream_t streams[NUM_STREAMS];

template<int S>
class Scheduler {
	std::vector<std::function<void()>> functions;
	std::vector<int> map;
	int i=0;

public:
	template<typename Func, typename Type>
	void kernelCall(Func func, Type h_a, uint n) {
		auto funct = std::bind(func, h_a, n, i++);
		functions.push_back(funct);
		map.push_back(-1);
	}

	template<typename Func, typename Type>
	void kernelCall(Func func, Type h_a, Type h_b,	uint n) {
		auto funct = std::bind(func, h_a, h_b, n, i++);
		functions.push_back(funct);
		map.push_back(-1);
	}

	template<typename Func, typename Type>
	void kernelCall(Func func, Type h_a, Type h_b, Type h_c, uint n) {
		auto funct = std::bind(func, h_a, h_b, h_c, n, i++);
		functions.push_back(funct);
		map.push_back(-1);
	}

	void schedule(){
		int k = 0;
		for(auto funct : functions){
			map[k] = k;
			k=(++k) % S;
		}

	}

	void execute(){
		//cudaStream_t streams[S];
		for (int i = 0; i < S; i++) {
			cudaStreamCreate(&streams[i]);
		}

		for(auto funct : functions){
			funct();
		}
	}
};

__global__ void addVec(uint *d_a, uint *d_b, uint *d_c, uint n) {
	// Get our global thread ID
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("id=%d ", id);

	// Make sure we do not go out of bounds
	if (id < n)
		for (int i = 0; i < 20; i++) {
			d_a[id] = d_b[id] + d_c[id];
			d_a[id] = d_a[id] * 2;
			d_a[id] = d_a[id] - d_b[id] - d_c[id];
			d_a[id] = d_a[id] / 2;
		}
}

extern "C" void kernel1(uint *h_a, uint *h_b, uint *h_c, uint n, int k) {
	uint mem_size = sizeof(uint) * n;

	uint *d_a, *d_b, *d_c;
	cudaMalloc((void **) &d_a, mem_size);
	cudaMalloc((void **) &d_b, mem_size);
	cudaMalloc((void **) &d_c, mem_size);

	//print(h_b, n);

	cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, mem_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n - 1) / dimBlock.x + 1);

	printf("stream1=%d", streams[k]);
	//addVec<<<dimBlock, dimGrid, 0, stream>>>(d_a, d_b, d_c, n);
	addVec<<<dimBlock, dimGrid, 0, streams[k]>>>(d_a, d_b, d_c, n);
	printf("Kernel1\n");

	cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

	//print(h_a, n);
}

extern "C" void kernel2(uint *h_a, uint *h_b, uint *h_c, uint n, int k) {
	uint mem_size = sizeof(uint) * n;

	uint *d_a, *d_b, *d_c;
	cudaMalloc((void **) &d_a, mem_size);
	cudaMalloc((void **) &d_b, mem_size);
	cudaMalloc((void **) &d_c, mem_size);

	//print(h_b, n);

	cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, mem_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n - 1) / dimBlock.x + 1);

	printf("stream2=%d", streams[k]);
	//addVec<<<dimBlock, dimGrid, 0, stream>>>(d_a, d_b, d_c, n);
	addVec<<<dimBlock, dimGrid, 0, streams[k]>>>(d_a, d_b, d_c, n);
	printf("Kernel2\n");

	cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

	//print(h_a, n);
}

extern "C" void kernel3(uint *h_a, uint n){
	uint mem_size = sizeof(uint) * n;

	uint *d_a;
	cudaMalloc((void **) &d_a, mem_size);

	//print(h_b, n);

	cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n - 1) / dimBlock.x + 1);

	//addVec<<<dimBlock, dimGrid, 0, stream>>>(d_a, d_b, d_c, n);
	addVec<<<dimBlock, dimGrid>>>(d_a, d_a, d_a, n);
	printf("Kernel3\n");

	cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);
}


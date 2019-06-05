#include <cuda.h>
#include "kernels.h"
#include <iostream>
#include <list>
//#include <functional>
#include <typeinfo>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

//template <int N, typename T>
template<class Func, class Type>
struct Function {
	Func f;
	dim3 dimBlock;
	dim3 dimGrid;
	Type d_a = NULL;
	Type d_b = NULL;
	Type d_c = NULL;
	int n = 0;

	//template <class Func>
	Function(Func f, dim3 dimBlock, dim3 dimGrid, int n, Type d_a, Type d_b,
			Type d_c) {
		this->f = f;
		this->dimBlock = dimBlock;
		this->dimGrid = dimGrid;
		this->n = n;
		this->d_a = d_a;
		this->d_b = d_b;
		this->d_c = d_c;
	}

	void call() {
		f<<<dimBlock, dimGrid>>>(d_a, d_b, d_c, n);
	}
};

template<typename Func, typename Type>
class Scheduler {

	std::list<Function<Func, Type>*> list;

public:
	void kernelCall(Func func, dim3 dimBlock, dim3 dimGrid, Type d_a, uint n) {

	}

	void kernelCall(Func func, dim3 dimBlock, dim3 dimGrid, Type d_a, Type d_b,	uint n) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		func<<<dimBlock, dimGrid, 0, stream>>>(d_a, d_b, n);
	}

	void kernelCall(Func func, dim3 dimBlock, dim3 dimGrid, Type d_a, Type d_b, Type d_c, uint n) {
		Function<Func, Type> *f = new Function<Func, Type>(func, dimBlock,
				dimGrid, n, d_a, d_b, d_c);

		list.push_back(f);
	}

	void kernel_scheduling() {
		/*cudaStream_t streams[NUM_STREAMS];
		 for (int i = 0; i < NUM_STREAMS; i++) {
		 cudaStreamCreate(&streams[i]);
		 }

		 cudaStream_t stream;
		 cudaStreamCreate(&stream);*/
		//func<<<dimBlock, dimGrid, 0, stream>>>(d_a, n);
	}

	void execute() {
		for (auto funct : list) {
			funct->call();
		}
	}

};

template<typename T>
void print(T* vec, uint t) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";

}

__global__ void addVec(uint *d_a, uint *d_b, uint *d_c, uint n) {
	// Get our global thread ID
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("id=%d ", id);

	// Make sure we do not go out of bounds
	if (id < n)
		for (int i = 0; i < 20; i++) {
			d_a[id] = d_b[id] + d_c[id];
			d_a[id] = d_a[id] * 2;
			d_a[id] = d_a[id] - d_b[id] - d_c[id];
			d_a[id] = d_a[id] / 2;
		}
}

extern "C" void kernel1(uint *h_a, uint *h_b, uint *h_c, uint n) {
	uint mem_size = sizeof(uint) * n;

	uint *d_a, *d_b, *d_c;
	cudaMalloc((void **) &d_a, mem_size);
	cudaMalloc((void **) &d_b, mem_size);
	cudaMalloc((void **) &d_c, mem_size);

	//print(h_b, n);

	cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, mem_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n / 2 - 1) / dimBlock.x + 1);

	//addVec<<<dimBlock, dimGrid, 0, stream>>>(d_a, d_b, d_c, n);
	addVec<<<dimBlock, dimGrid>>>(d_a, d_b, d_c, n);

//	Function<>

	//Scheduler<void (*)(uint*, uint*, uint*, uint), uint*> s;
	//s.kernelCall(addVec, dimBlock, dimGrid, d_a, d_b, d_c, n);
	//s.kernel_scheduling();
	//s.execute();

	//auto teste = typedef(addVec);
	//std::cout << teste;

	cudaMemcpy(h_a, d_a, mem_size, cudaMemcpyDeviceToHost);

	//print(h_a, n);
}


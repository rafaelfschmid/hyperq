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
#include "kernels.h"
#include <functional>
#include <iostream>
#include <vector>
#include<tuple>
#include <future>

using namespace std;
using namespace std::placeholders;

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

//cudaStream_t streams[NUM_STREAMS];

class Scheduler {
	std::vector<std::tuple<void (*)(uint, uint, uint, uint, cudaStream_t), uint, uint, uint, uint>> functions;
	std::vector<int> map;
	int i=0;

public:
	cudaStream_t *streams;
	int num_streams;

	Scheduler(int num_streams){
		this->num_streams = num_streams;
		streams = new cudaStream_t[num_streams];
		for (int i = 0; i < this->num_streams; i++) {
			cudaStreamCreate(&streams[i]);
		}
	}

	template<typename Func>
	void kernelCall(Func func, uint num_threads, uint num_blocks, uint shared_size, uint computation) {
		auto funct = make_tuple(func, num_threads, num_blocks, shared_size, computation);
		functions.push_back(funct);
		map.push_back(-1);
	}

	void schedule(){
		int k = 0;
		int j = 0;
		for(auto funct : functions){
			//printf("k=%d ", k);
			map[j++] = k;
			k=(++k) % num_streams;
		}
	}

	void execute(){
		int k = 0;

		for(auto f : functions){
			//printf("\nk=%d", k);
			std::async(std::launch::async, get<0>(f),get<1>(f),get<2>(f),get<3>(f),get<4>(f),streams[map[k]]);
			k++;
		}
	}
};


int main(int argc, char **argv) {

	getDeviceInformation();

	Scheduler s(4);

	uint num_threads = 128;
	uint num_blocks = 32;
	uint shared_size = 64;
	uint computation = 1024;

	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*16);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size, computation);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*8);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*4);
	s.kernelCall(kernel1, num_threads, num_blocks, shared_size*2, computation*16);
	s.schedule();
	s.execute();


	return 0;
}

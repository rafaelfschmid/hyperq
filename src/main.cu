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

#include <dlfcn.h>
#include "kernels.h"
#include "cudahook.h"

#include <functional>
#include <iostream>
#include <vector>
#include<tuple>
#include <future>

#include <thread>

typedef void* my_lib_t;

my_lib_t MyLoadLib(const char* szMyLib) {
	return dlopen(szMyLib, RTLD_LAZY);
}

void MyUnloadLib(my_lib_t hMyLib) {
	dlclose(hMyLib);
}

void* MyLoadProc(my_lib_t hMyLib, const char* szMyProc) {
	return dlsym(hMyLib, szMyProc);
}

typedef bool (*scheduleKernels_t)(int, int);
my_lib_t hMyLib = NULL;
scheduleKernels_t scheduleKernels = NULL;

bool callcudahook(int n, int streams) {
  if (!(hMyLib = MyLoadLib("/home/rafael/cuda-workspace/hyperq/src/libcudahook.so"))) { /*error*/ }
  if (!(scheduleKernels = (scheduleKernels_t)MyLoadProc(hMyLib, "scheduleKernels"))) { /*error*/ }

  bool ret = scheduleKernels(n, streams);

  MyUnloadLib(hMyLib);

  return ret;
}

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
	std::vector<std::tuple<void (*)(uint, uint, uint, uint), uint, uint, uint, uint>> functions;
	//std::vector<std::tuple<void (*)(uint, uint, uint, uint, cudaStream_t), uint, uint, uint, uint>> functions;
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
		std::vector<std::future<void>> vec;
		for(auto f : functions){
			printf("testando0.0\n");
			//vec.push_back(std::async(std::launch::async, get<0>(f),get<1>(f),get<2>(f),get<3>(f),get<4>(f),streams[map[k]]));
			vec.push_back(std::async(std::launch::async, get<0>(f),get<1>(f),get<2>(f),get<3>(f),get<4>(f)));
			k++;
		}

		printf("testando0.1\n");
		callcudahook(vec.size(), num_streams);
		printf("testando0.2\n");
		for(k = 0; k < vec.size(); k++){
			vec[k].get();
			k++;
		}
	}
};




int main(int argc, char **argv) {

	getDeviceInformation();

	Scheduler s(8);
	//callcudahook();


	uint num_threads = 16;
	uint num_blocks = 2;
	uint shared_size = 16;
	uint computation = 2;

	int j = 1;
	for(int i = 1; i < 64; i++) {
		j = (j+1)%32+1;
		s.kernelCall(kernel1, num_threads*j, num_blocks*j*2, shared_size*j, computation*j*1000);
	}

	s.schedule();
	printf("testando0\n");
	s.execute();
	printf("testando1\n");
	//callcudahook();
	//printf("testando2\n");

	return 0;
}

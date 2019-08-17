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
#include <omp.h>

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

	void execute1(){
		bip::managed_shared_memory segment(bip::open_only, "shared_memory");
		SharedVector* kernels2 = segment.find<SharedVector>("Kernels2").first;

		int k = 0;
		for(auto f : functions){
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			get<0>(f)(get<1>(f),get<2>(f),get<3>(f),get<4>(f),cudaStream_t());
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);

			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			//std::cout << milliseconds << "\n";*/
			(*kernels2)[k].milliseconds = milliseconds;
			k++;
		}

	}

	void execute2(){
		int k = 0;

		std::vector<std::future<void>> vec;
		for(auto f : functions){
			//vec.push_back(std::async(std::launch::async, get<0>(f),get<1>(f),get<2>(f),get<3>(f),get<4>(f),streams[map[k]]));
			vec.push_back(std::async(std::launch::async, get<0>(f),get<1>(f),get<2>(f),get<3>(f),get<4>(f),cudaStream_t()));
			k++;
		}

		printf("testando0.2\n");
		for(k = 0; k < vec.size(); k++){
			printf("%s\n", get<0>(functions[k]));
			vec[k].get();
		}
	}

	void execute3(){

		/*omp_set_num_threads(num_streams);
		#pragma omp parallel
		{
			uint id = omp_get_thread_num(); //cpu_thread_id
			for (int i = 0; i < functions.size(); i+=num_streams) {
				uint k = i + id;
				get<0>(functions[k])(get<1>(functions[k]),get<2>(functions[k]),get<3>(functions[k]),get<4>(functions[k]),streams[map[k]]);
			}
		}*/

		int k = 0;
		for(auto f : functions){
			get<0>(f)(get<1>(f),get<2>(f),get<3>(f),get<4>(f),streams[map[k]]);
			k++;
		}

	}

};

int main(int argc, char **argv) {

	bip::shared_memory_object::remove("shared_memory");
	bip::managed_shared_memory segment(boost::interprocess::create_only, "shared_memory", 65536);

	// Index of threads
	int *id = segment.construct<int>("Index")(-1);

	SharedMap *kernels = segment.construct<SharedMap>("Kernels")( 3, boost::hash<ShmemString>(), std::equal_to<ShmemString>()
	        , segment.get_allocator<SharedMap>());

	SharedVector *kernels2 =  segment.construct<SharedVector>("Kernels2")(segment.get_allocator<SharedVector>()); //segment.get_segment_manager());

	getDeviceInformation();


	Scheduler s(4);

	int number_of_kernels = 4;
	int *max = segment.construct<int>("Max")(number_of_kernels);


	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	//srand(time(NULL));
	srand(0);
	for(int i = 0; i < number_of_kernels; i++) {
		uint num_threads = rand() % deviceProp.maxThreadsPerBlock;
		uint num_blocks = rand() % deviceProp.maxGridSize[0];
		uint shared_size = rand() % deviceProp.sharedMemPerBlock;
		uint computation = rand() % 10;

		printf("threads=%d ---- blocks=%d ---- shared_size=%d ---- comput=%d\n", num_threads, num_blocks, shared_size, computation);
		s.kernelCall(kernel3, num_threads, num_blocks, shared_size, computation*(i+1));
	}

	s.schedule();
	printf("testando0\n");
	s.execute1();
	printf("testando1\n");
	s.execute3();
	//callcudahook();
	//printf("testando2\n");

	std::cout << kernels->size() << "\n";
	for(SharedMap::iterator iter = kernels->begin(); iter != kernels->end(); iter++)
	{
		//printf("%d %s %f\n", iter->second.id, iter->first.data(), iter->second.microseconds);
		std::cout << iter->second.id << " " << iter->first.data() << " " << iter->second.milliseconds << "\n";
	}

	for(SharedVector::iterator iter = kernels2->begin(); iter != kernels2->end(); iter++)
	{
		//printf("%d %s %f\n", iter->second.id, iter->first.data(), iter->second.microseconds);
		std::cout << iter->id << " " << iter->milliseconds << "\n";
	}


	return 0;
}



/*#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>

#include <vector>
#include <thread>
#include <future>
#include <string.h>

#include <unistd.h>
#include <dlfcn.h>
#include <signal.h>

#include "cudahook.h"

void exec(const char* s){
	system(s);
}

int main(int argc, char **argv) {

	//printf("argc=%d", argc);

	bip::shared_memory_object::remove("shared_memory");
	bip::managed_shared_memory segment(boost::interprocess::create_only, "shared_memory", 65536);

	// Index of threads
	int *id = segment.construct<int>("Index")(-1);
	// Shared map of kernels
	//SharedMap *kernels =  segment.construct<SharedMap>("Kernels") (std::less<MapKey>() ,segment.get_segment_manager());


	SharedMap *kernels = segment.construct<SharedMap>("Kernels")( 3, boost::hash<ShmemString>(), std::equal_to<ShmemString>()
	        , segment.get_allocator<SharedMap>());



	exec(line.data());

	f_out << kernels->size() << "\n";
	for(SharedMap::iterator iter = kernels->begin(); iter != kernels->end(); iter++)
	{
		//printf("%d %s %f\n", iter->second.id, iter->first.data(), iter->second.microseconds);
		f_out << iter->second.id << " " << iter->first.data() << " " << iter->second.microseconds << "\n";
	}
	//f_out << "\n";
	f_out.close();
	//callcudahook(2);

	std::vector<std::future<void>> vec;
	std::getline (std::cin, line1);
	vec.push_back(std::async(std::launch::async,exec,line1.data()));

	std::getline (std::cin, line2);
	vec.push_back(std::async(std::launch::async,exec,line2.data()));

	printf("começoooouuuu\n");
	bool test = callcudahook(2, 2);
	printf("acaboooouuuu\n");

	vec[0].get();
	vec[1].get();
	vec[2].get();



	bip::shared_memory_object::remove("shared_memory");

	return 0;
}*/


/*std::vector<std::future<void>> vec;

	std::string line1 = "";
	std::string line2 = "";
	std::string line3 = "";
	std::string line4 = "";
	std::string line5 = "";
	std::string line6 = "";
	std::string line7 = "";
	std::string line8 = "";

	std::getline (std::cin, line1);
	std::getline (std::cin, line2);
	std::getline (std::cin, line3);
	std::getline (std::cin, line4);


	std::vector<char*> commandVector;
	commandVector.push_back(const_cast<char*>(line2.data()));
	commandVector.push_back(const_cast<char*>(line3.data()));
	commandVector.push_back(const_cast<char*>(line4.data()));
	commandVector.push_back(NULL);
	//const int status = execvp(commandVector[0], &commandVector[0]);
	//exec(commandVector[0], commandVector);
	myclass a(commandVector[0], commandVector);

	std::vector<char*> commandVector2;
	//commandVector2.push_back(const_cast<char*>(line1.data()));
	std::getline (std::cin, line1);
	std::getline (std::cin, line2);
	std::getline (std::cin, line3);
	std::getline (std::cin, line4);
	std::getline (std::cin, line5);
	std::getline (std::cin, line6);
	std::getline (std::cin, line7);
	std::getline (std::cin, line8);

	commandVector2.push_back(const_cast<char*>(line2.data()));
	commandVector2.push_back(const_cast<char*>(line3.data()));
	commandVector2.push_back(const_cast<char*>(line4.data()));
	commandVector2.push_back(const_cast<char*>(line5.data()));
	commandVector2.push_back(const_cast<char*>(line6.data()));
	commandVector2.push_back(const_cast<char*>(line7.data()));
	commandVector2.push_back(const_cast<char*>(line8.data()));
	commandVector2.push_back(NULL);
	//const int status = execvp(commandVector[0], &commandVector[0]);
	//exec(commandVector2[0], commandVector2);
	myclass b(commandVector2[0], commandVector2);*/

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
#include <cuda_profiler_api.h>
#include <iostream>
#include <vector>
#include <future>

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

void exec(const char* s){
	system(s);
}

class Scheduler {
	std::vector<std::string> programs;
	std::vector<int> map;
	int i=0;

public:
	Scheduler(){
	}

	void programCall(std::string str) {
		//std::cout << str << "\n";
		programs.push_back(str);
		map.push_back(-1);
	}

	void schedule(){

	}

	void execute(){
		//cudaProfilerStart();
		for(auto f : programs){
			//std::async(std::launch::async, exec, f.data());
			std::thread t1(exec, f.data());
			t1.join();
		}

		/*for(auto f : programs){
			//std::async(std::launch::async, exec, f.data());
			t1(exec, f.data());
		}*/
		//cudaProfilerStop();
	}
};


int main(int argc, char **argv) {

	getDeviceInformation();

	Scheduler s;

	std::string line = "";
	while(line != " ") {
		std::getline (std::cin, line);
		//std::cout << line << "\n";
		//std::string str = argv[i];//"./hotspot 1024 2 2 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.out";
		s.programCall(line);
	//	s.schedule();
	}
	s.execute();

	return 0;
}

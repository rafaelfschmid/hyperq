#include <stdio.h>
#include <dlfcn.h>
#include <cassert>
#include <list>
#include <cuda.h>
#include <vector_types.h>
#include <vector>

typedef struct {
	const char* entry;
	int numOfBlocks;
	int numOfThreads;
	int numOfRegisters;
	int sharedMemory;
	int computationalTime;
	bool start = false;
	std::list<void *> args;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
	static kernelInfo_t _kernelInfo;
	return _kernelInfo;
}

std::vector<kernelInfo_t> &kernels() {
	static std::vector<kernelInfo_t> _kernels;
	return _kernels;
}

typedef struct {
	int numOfSMs;
	int numOfRegister; // register per SM
	int maxThreads;    // max threads per SM
	int sharedMemory;  // sharedMemory per SM
} deviceInfo_t;

deviceInfo_t &deviceInfo() {
	static deviceInfo_t _deviceInfo;
	return _deviceInfo;
}

std::vector<deviceInfo_t> &devices() {
	static std::vector<deviceInfo_t> _devices;
	return _devices;
}

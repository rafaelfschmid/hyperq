#include <thread>
#include <chrono>

#include "cudahook.h"

#define DEBUG 0

typedef cudaError_enum (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes *,	const void *);
static cudaFuncGetAttributes_t realCudaFuncGetAttributes = NULL;

extern "C" cudaError_enum cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {

	if (realCudaFuncGetAttributes == NULL)
		realCudaFuncGetAttributes = (cudaFuncGetAttributes_t) dlsym(RTLD_NEXT,
				"cudaFuncGetAttributes");

	assert(realCudaFuncGetAttributes != NULL && "cudaFuncGetAttributes is null");

	return realCudaFuncGetAttributes(attr, func);
}

typedef cudaError_enum (*cudaGetDeviceProperties_t)(struct cudaDeviceProp *prop, int device);
static cudaGetDeviceProperties_t realCudaGetDeviceProperties = NULL;

extern "C" cudaError_enum cudaGetDeviceProperties(struct cudaDeviceProp *prop,	int device) {

	if (realCudaGetDeviceProperties == NULL)
		realCudaGetDeviceProperties = (cudaGetDeviceProperties_t) dlsym(RTLD_NEXT, "cudaGetDeviceProperties");

	assert(realCudaGetDeviceProperties != NULL && "cudaGetDeviceProperties is null");

	auto ret = realCudaGetDeviceProperties(prop, device);

	deviceInfo().numOfSMs = prop->multiProcessorCount;
	deviceInfo().numOfRegister = prop->regsPerMultiprocessor;
	deviceInfo().sharedMemory = prop->sharedMemPerMultiprocessor;
	deviceInfo().maxThreads = prop->maxThreadsPerMultiProcessor;
	devices().push_back(deviceInfo());

	return ret;
}

void printDevices() {
	for(auto d : devices()) {
		printf("##################################################\n");
		printf("numOfSMs=%d\n", d.numOfSMs);
		printf("numOfRegister=%d\n", d.numOfRegister);
		printf("sharedMemory=%d\n", d.sharedMemory);
		printf("maxThreads=%d\n", d.maxThreads);
		printf("##################################################\n");
	}
}

typedef cudaError_enum (*cudaConfigureCall_t)(dim3, dim3, size_t, CUstream_st*);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_enum cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st* stream = 0) {
	if(DEBUG)
		printf("TESTE 1\n");

	kernelInfo().sharedDynamicMemory = sharedMem;
	kernelInfo().numOfThreads = blockDim.x * blockDim.y * blockDim.z;
	kernelInfo().numOfBlocks = gridDim.x * gridDim.y * gridDim.z;

	//std::this_thread::sleep_for(std::chrono::seconds(2));
	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, stream);

}

typedef cudaError_enum (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_enum cudaLaunch(const char *entry) {

	printf("Testando - CudaLaunch\n");
	bip::managed_shared_memory segment(bip::open_only, "shared_memory");

	int* 				index 	= segment.find<int>("Index").first;
	int* 				numKernels 	= segment.find<int>("Max").first;

	if(*index < ((*numKernels) - 1)) {
		printf("index=%d --- numKernels=%d\n", *index, (*numKernels)-1);

		SharedVector* 		kernels2 = segment.find<SharedVector>("Kernels2").first;

		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, (void*) entry);

		kernelInfo_t k;
		k.sharedDynamicMemory = kernelInfo().sharedDynamicMemory;
		k.numOfThreads = kernelInfo().numOfThreads;
		k.numOfBlocks = kernelInfo().numOfBlocks;
		k.numOfRegisters = attr.numRegs;
		k.sharedStaticMemory = attr.sharedSizeBytes;
		k.start = false;
		k.id = *index = (*index) + 1;

		kernels2->push_back(k);
	}

	if (realCudaLaunch == NULL) {
		realCudaLaunch = (cudaLaunch_t) dlsym(RTLD_NEXT, "cudaLaunch");
	}
	assert(realCudaLaunch != NULL && "cudaLaunch is null");

	//auto start = std::chrono::high_resolution_clock::now();
	cudaError_enum ret = realCudaLaunch(entry);


	return ret;
}

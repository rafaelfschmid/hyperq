#include "cudahook.h"
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <atomic>
#include <iostream>

typedef cudaError_t (*cudaFuncGetAttributes_t)(struct cudaFuncAttributes *,	const void *);
static cudaFuncGetAttributes_t realCudaFuncGetAttributes = NULL;

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {

	if (realCudaFuncGetAttributes == NULL)
		realCudaFuncGetAttributes = (cudaFuncGetAttributes_t) dlsym(RTLD_NEXT,
				"cudaFuncGetAttributes");

	assert(realCudaFuncGetAttributes != NULL && "cudaFuncGetAttributes is null");

	return realCudaFuncGetAttributes(attr, func);
}

typedef cudaError_t (*cudaGetDeviceProperties_t)(struct cudaDeviceProp *prop, int device);
static cudaGetDeviceProperties_t realCudaGetDeviceProperties = NULL;

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,	int device) {

	if (realCudaGetDeviceProperties == NULL)
		realCudaGetDeviceProperties = (cudaGetDeviceProperties_t) dlsym(RTLD_NEXT, "cudaGetDeviceProperties");

	assert(realCudaGetDeviceProperties != NULL && "cudaGetDeviceProperties is null");

	auto ret = realCudaGetDeviceProperties(prop, device);

	deviceInfo().numOfSMs = prop->multiProcessorCount;
	deviceInfo().numOfRegister = prop->regsPerMultiprocessor;
	deviceInfo().sharedMemory = prop->sharedMemPerMultiprocessor;
	deviceInfo().maxThreads = prop->maxThreadsPerMultiProcessor;

	return ret;
}

void printDevices() {
	for(auto d : devices()) {
		printf("##################################################\n");
		printf("numOfSMs=%s\n", d.numOfSMs);
		printf("numOfRegister=%s\n", d.numOfRegister);
		printf("sharedMemory=%s\n", d.sharedMemory);
		printf("maxThreads=%s\n", d.maxThreads);
		printf("##################################################\n");
	}
}

void printKernels() {
	for(auto k : kernels()) {
		printf("##################################################\n");
		printf("entry=%d\n", k.entry);
		printf("numOfBlocks=%d\n", k.numOfBlocks);
		printf("numOfThreads=%d\n", k.numOfThreads);
		printf("numOfRegisters=%d\n", k.numOfRegisters);
		printf("sharedMemory=%d\n", k.sharedMemory);
		printf("computationalTime=%d\n", k.computationalTime);
		printf("##################################################\n");
	}
}

std::condition_variable cvm;
std::mutex cv_m;

std::condition_variable cvx;
std::mutex cv_x;

extern "C" bool scheduleKernels(int n) {
	{
		std::unique_lock<std::mutex> lkg(cv_x);
		printf("n=%d\n", n);
		printf("kernels().size()=%d\n", kernels().size());
		cvx.wait(lkg, [&n](){ return kernels().size() == n; });
		printf("kernels().size()=%d\n", kernels().size());
	}

	bool a = true;
	for(auto &k : kernels()) {
		std::lock_guard<std::mutex> lk(cv_m);
		k.start = a;
		a = !a;
		cvm.notify_all();
	}

	return true;
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0) {
	printf("TESTE 1\n");

	kernelInfo().sharedMemory = sharedMem;
	kernelInfo().numOfThreads = blockDim.x * blockDim.y * blockDim.z;
	kernelInfo().numOfBlocks = gridDim.x * gridDim.y * gridDim.z;

	if (realCudaConfigureCall == NULL)
		realCudaConfigureCall = (cudaConfigureCall_t) dlsym(RTLD_NEXT, "cudaConfigureCall");

	assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
	return realCudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

typedef cudaError_t (*cudaLaunch_t)(const char *);
static cudaLaunch_t realCudaLaunch = NULL;

extern "C" cudaError_t cudaLaunch(const char *entry) {

	cudaFuncAttributes attr;
	//cudaFuncGetAttributes(&attr, kernels()[entry]);
	cudaFuncGetAttributes(&attr, (void*) entry);

	kernelInfo().entry = entry;
	kernelInfo().numOfRegisters = attr.numRegs;
	kernelInfo().sharedMemory += attr.sharedSizeBytes;

	int i;
	{
		printf("testando\n");
		std::lock_guard<std::mutex> lkg(cv_x);
		kernels().push_back(kernelInfo());
		i = kernels().size() - 1;
		cvx.notify_all();
	}

	{
		std::unique_lock<std::mutex> lk(cv_m);
		printf("Waiting... \n");
		cvm.wait(lk, [&i](){return kernels()[i].start;});
		printf("%d...finished waiting.\n", i);
	}

	if (realCudaLaunch == NULL) {
		realCudaLaunch = (cudaLaunch_t) dlsym(RTLD_NEXT, "cudaLaunch");
	}
	assert(realCudaLaunch != NULL && "cudaLaunch is null");

	return realCudaLaunch(entry);
	//return (cudaError_t)0; //success == 0
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
		const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
		const char *hostFun, char *deviceFun, const char *deviceName,
		int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
		int *wSize) {

	printf("TESTE 0\n");

	if (realCudaRegisterFunction == NULL) {
		realCudaRegisterFunction = (cudaRegisterFunction_t) dlsym(RTLD_NEXT,
				"__cudaRegisterFunction");
	}
	assert(realCudaRegisterFunction != NULL && "cudaRegisterFunction is null");

	realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
			thread_limit, tid, bid, bDim, gDim, wSize);
}

typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
static cudaSetupArgument_t realCudaSetupArgument = NULL;

extern "C" cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	printf("TESTE 2\n");

	kernelInfo().args.push_back(const_cast<void *>(arg));
	if (realCudaSetupArgument == NULL) {
		realCudaSetupArgument = (cudaSetupArgument_t) dlsym(RTLD_NEXT,
				"cudaSetupArgument");
	}
	assert(realCudaSetupArgument != NULL);
	return realCudaSetupArgument(arg, size, offset);
}

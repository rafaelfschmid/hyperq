
#include <stdio.h>
#include <stdlib.h>  /* exit */

#include <list>

#include <cuda.h>
#include <driver_types.h>

#include <vector_types.h>
#include <vector>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#include <boost/unordered_map.hpp>

#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

#include <dlfcn.h>
#include <cassert>

#include <semaphore.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <unistd.h>  /* _exit, fork */
#include <errno.h>   /* errno */
#include <signal.h>
#include <string.h>

/*
#ifdef CUDA10
#define cudaStream_t CUstream_st;
#define cudaError_t cudaError_enum;
#define cudaEventCreate_t cuEventCreate;
#define cudaEventRecord_t cuEventRecord;
#define cudaEventSynchronize_t cuEventSynchronize;
#define cudaEventElapsedTime_t cuEventElapsedTime;
#define cudaFuncGetAttributes_t cuFuncGetAttribute;
#define cudaGetDeviceProperties_t cuDeviceGetProperties;
#define cudaStreamCreate_t cuStreamCreate;
#define cudaFree_t cuMemFree_v2;
#define cudaEvent_t cuEvent;
// typedef X cudaConfigureCall_t;
#define cuLaunch cudaLaunch_t;
//typedef Y cudaSetupArgument_t;
#endif
*/
namespace bip = boost::interprocess;

typedef struct {
	char* entry;
	int id = -1;
	//dim3 gridDim;
	//dim3 blockDim;
	int numOfBlocks;
	int numOfThreads;
	int numOfRegisters;
	int sharedDynamicMemory;
	int sharedStaticMemory;
	CUstream_st* stream;
	float milliseconds;
	int computation;
	bool start = false;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
	static kernelInfo_t _kernelInfo;
	return _kernelInfo;
}

std::map<const char *, char *> &kernelsMap() {
  static std::map<const char*, char*> _kernels;
  return _kernels;
}

/*std::vector<kernelInfo_t> &kernels() {
	static std::vector<kernelInfo_t> _kernels;
	return _kernels;
}*/

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

typedef bip::allocator<char, bip::managed_shared_memory::segment_manager> CharAllocator;
typedef bip::basic_string<char, std::char_traits<char>, CharAllocator> ShmemString;


typedef ShmemString MapKey;
typedef kernelInfo_t MapValue;

typedef std::pair< MapKey, MapValue> ValueType;

//allocator of for the map.
typedef bip::allocator<ValueType, bip::managed_shared_memory::segment_manager> ShMemAllocator;
typedef boost::unordered_map< MapKey, MapValue, boost::hash<MapKey>, std::equal_to<MapKey>, ShMemAllocator > SharedMap;

typedef bip::allocator<kernelInfo_t, bip::managed_shared_memory::segment_manager> KernelAllocator;
typedef bip::vector< kernelInfo_t, KernelAllocator > SharedVector;




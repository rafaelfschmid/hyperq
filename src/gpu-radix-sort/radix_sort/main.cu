#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>

#include "sort.h"
#include "utils.h"

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 1
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

void print(uint* host_data, uint n, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << host_data[i * m + j] << " ";
		}
		std::cout << "\n";
	}

}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, int m, unsigned int *results_h)
{
    for (int i = 0 ; i < n ; ++i) {
    	for (uint j = 1; j < m; j++) {
    		if (results_h[i*m +j -1] > results_h[i*m +j])
			{
				std::cout << "Invalid item[" << j-1 << "]: " << results_h[i*m +j -1] << " greater than " << results_h[i*m +j] << std::endl;
				exit(EXIT_FAILURE);
			}
    	}
    }

    std::cout << "OK" << std::endl;
}

int main(int argc, char** argv) {

	uint num_of_segments;
	uint num_of_elements;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (int i = 0; i < num_of_segments+1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	for (int i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_out, *d_vec;

	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	int nstreams = NUM_STREAMS;
	if (NUM_STREAMS > num_of_segments)
		nstreams = num_of_segments;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_out, mem_size_vec));

	for (uint j = 0; j < EXECUTIONS; j++) {

		// copy host memory to device
		//cudaTest(cudaMemcpy(d_out, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
		for(int i = 0; i < num_of_segments; i+=nstreams) {
			for (int s = 0; s < nstreams; s++) {
				radix_sort(d_out + h_seg[i+s], d_vec + h_seg[i+s], num_of_elements/num_of_segments, streams[s]);
			}
		}
		cudaEventRecord(stop);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	cudaTest(cudaMemcpy(h_vec, d_out, mem_size_vec, cudaMemcpyDeviceToHost));

	cudaFree(d_vec);
	cudaFree(d_out);

	if (ELAPSED_TIME != 1) {
		//print(h_vec, num_of_segments, num_of_elements/num_of_segments);
		check_results(num_of_segments, num_of_elements/num_of_segments, h_vec);
	}

	free(h_seg);
	free(h_vec);

	return 0;
}



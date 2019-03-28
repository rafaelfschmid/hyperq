#include "kernels.h"

__global__ void kernel(uint *d_vec, uint N){

}

extern "C" void kernel1(uint *d_vec, uint N){
	uint num_of_elements=1024;

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(int) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint *h_value = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}
	kernel<<<>>>();
}

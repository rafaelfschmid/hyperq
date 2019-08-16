CUDA_PATH ?= "/usr/local/cuda-8.0"
NVCC := $(CUDA_PATH)/bin/nvcc
#NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -m64

radixsortstreams: main.cu scan.o sort.o Makefile
	for STREAMS in 1 8 16 32 ; do \
		$(NVCC) -o radixsort$${STREAMS}.exe main.cu sort.o scan.o $(NVCC_OPTS) -DEXECUTIONS=$(EXECS) -std=c++11 --expt-extended-lambda -lcuda -DNUM_STREAMS=$${STREAMS} ; \
	done

radixsort: main.cu scan.o sort.o Makefile
	$(NVCC) -o radixsort main.cu sort.o scan.o $(NVCC_OPTS) -DEXECUTIONS=$(EXECS) -std=c++11 --expt-extended-lambda -lcuda

sort.o: sort.cu
	$(NVCC) -c sort.cu $(NVCC_OPTS)

scan.o: scan.cu
	$(NVCC) -c scan.cu $(NVCC_OPTS)

clean:
	rm -f *.o radix_sort

#TIME=1
#BITS=12
#EXECS=1

arch=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60
COMMONFLAGS=-Wall -fPIC -shared -ldl

all: build

build: libcudahook.so main.exe maingen.exe

libcudahook.so: cudahook.cpp
	g++ -I$(CUDA_HOME)/include $(COMMONFLAGS) -o libcudahook.so cudahook.cpp -std=c++11

kernels.o: kernels.cu
	nvcc -dc $(arch) -o $@ -c $< -std=c++11

main.o: main.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp -dc $(arch) -o $@ -c $< -std=c++11

maingen.o: maingen.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp -dc $(arch) -o $@ -c $< -std=c++11

maingen.exe: kernels.o maingen.o
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch)  $+ -o $@ -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -lcudart -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

main.exe: kernels.o main.o
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch)  $+ -o $@ -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -lcudart -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -lcuda

clean:
	rm *.o *.exe *.so

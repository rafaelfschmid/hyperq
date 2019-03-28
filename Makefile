#TIME=1
#BITS=12
#EXECS=1

arch=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60

all: mergeseg radixseg fixcub fixthrust nthrust fixseq mseq

main.exe: main.cu
	nvcc $(arch) main.cu -o $@ -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)



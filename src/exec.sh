export LD_PRELOAD=/home/rafael/cuda-workspace/hyperq/src/libcudahook.so
nvprof -f -o main.nvvp ./main.exe

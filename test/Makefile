NVCC := /usr/local/cuda/bin/nvcc -arch=sm_20

main.o: test_BiconnectedComponents.cu testlib.cu generator.cu serialBCC.cu
	$(NVCC) -c $(DEBUG) test_BiconnectedComponents.cu -o main.o


NVCC := /usr/local/cuda/bin/nvcc

class: 
	$(NVCC) -arch=sm_20 BiconnectedComponents.cu -o class

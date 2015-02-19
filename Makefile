NVCC := /usr/local/cuda/bin/nvcc

class:
	$(NVCC) -arch=sm_20 BiconnectedComponents.cu -o class

test_preprocessEdges:
	$(NVCC) -arch=sm_20 test/computeTreeTester.cu -o test_preprocessEdges
	./test_preprocessEdges
	rm testpreprocessEdge

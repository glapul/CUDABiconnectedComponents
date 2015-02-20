NVCC := /usr/local/cuda/bin/nvcc

class:
	$(NVCC) -arch=sm_20 BiconnectedComponents.cu -o class

test_preprocessEdges:
	$(NVCC) -arch=sm_20 test/test_preprocessEdges.cu -o test_preprocessEdges
	./test_preprocessEdges
	rm testpreprocessEdge

test_BFS:
	$(NVCC) -arch=sm_20 test/test_BFS.cu -o test_BFS
	./test_BFS
	rm test_BFS


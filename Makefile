NVCC := /usr/local/cuda/bin/nvcc

class:
	$(NVCC) -arch=sm_20 BiconnectedComponents.cu -o class

test_preprocessEdges:
	$(NVCC) -arch=sm_20 test/test_preprocessEdges.cu -o test_preprocessEdges
	./test_preprocessEdges
	rm test_preprocessEdge

test_BFS:
	$(NVCC) -arch=sm_20 test/test_BFS.cu -o test_BFS
	./test_BFS
	rm test_BFS

test_computeTreeFunctions:
	$(NVCC) -arch=sm_20 $(FLAGS) test/test_computeTreeFunctions.cu -o test_computeTreeFunctions
	./test_computeTreeFunctions
	rm test_computeTreeFunctions
tmp:
	$(NVCC) -arch=sm_20 tmp.cu -o tmp

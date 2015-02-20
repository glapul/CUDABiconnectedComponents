NVCC := /usr/local/cuda/bin/nvcc -arch=sm_20
CXX := g++
LINK := $(NVCC)

HEADERS := $(wildcard src/*.h)
SOURCES := $(wildcard src/*.cpp) $(wildcard src/*.cu)
OBJS := $(patsubst src/%, bin/%.o, $(SOURCES))

bin/%.cpp.o: src/%.cpp $(HEADERS)
	mkdir -p bin
	$(CXX) -c $< -o $@

bin/%.cu.o: src/%.cu $(HEADERS)
	mkdir -p bin
	$(NVCC) -c $< -o $@

.PHONY: test

test: $(OBJS) 
	cd test/ && make main.o
	$(LINK) test/main.o $(OBJS) -o bin/test
	bin/test

test_preprocessEdges:
	$(NVCC) -arch=sm_20 test/test_preprocessEdges.cu -o test_preprocessEdges
	./test_preprocessEdges
	rm testpreprocessEdge

test_BFS:
	$(NVCC) -arch=sm_20 test/test_BFS.cu -o test_BFS
	./test_BFS
	rm test_BFS

clean:
	rm bin/*

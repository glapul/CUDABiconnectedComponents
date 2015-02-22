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

test_connectedComponents: test/test_connectedComponents.cu src/computeConnectedComponents.cu
	$(NVCC) test/test_connectedComponents.cu src/computeConnectedComponents.cu -o test_connectedComponents.test
	./test_connectedComponents.test

test_computeTreeFunctions: test/test_computeTreeFunctions.cu src/computeTreeFunctions.cu
	$(NVCC) $(DEBUG) test/test_computeTreeFunctions.cu src/computeTreeFunctions.cu -o test_computeTreeFunctions.test
	./test_computeTreeFunctions.test
	rm test_computeTreeFunctions.test

clean:
	rm bin/*

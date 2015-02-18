#include "BiconnectedComponents.h"

const int BLOCK_SIZE = 32,
          NUM_OF_BLOCKS = 32,
          THREADS =  BLOCK_SIZE * NUM_OF_BLOCKS;

void BiconnectedComponents::computeSpanningTree(
        const Graph & graph,
        device_vector<bool> & inSpanningTree) const {

    device_vector<int> visited = device_vector<int>(graph.vertexCount);

    int thread_interval_length = (graph.edges.size() + THREADS - 1) / THREADS;
    void ** args = {&graph.edges, &inSpanningTree, &visited, &thread_interval_length, &graph.edges.size()};

    CUfunction _kernel_computeSpanningTree;
    _cu_exec(cuModuleGetFunction(&_kernel_computeSpanningTree,
                _cu_module, "_kernel_computeSpanningTree" ));

    _cu_exec(cuLaunchKernel(_kernel_computeSpanningTree,
                                NUM_OF_BLOCKS, 1, 1,
                                BLOCK_SIZE, 1, 1,
                                0, 0, args, 0));
    _cu_exec(cuCtxSynchronize());
}

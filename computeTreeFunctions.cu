#include "BiconnectedComponents.h"
using thrust::host_vector;
using thrust::device_vector;
const int BLOCK_SIZE = 1024,
          INF        = 1 << 30; 

/* edge preprocessing */
__global__ void _kernel_extractEdges(
            const Edge * edges,
            int * extractedEdges,
            int * edgeListStart,
            int num_of_edges,
            int num_of_vertices) {

        int id = threadIdx.x + blockIdx.x * blockDim.x;
        if(id >= num_of_edges)
            return;
        extractedEdges[id] = edges[id].to;
        if (id < num_of_edges - 1 && edges[id].from != edges[id + 1].from)
            edgeListStart[edges[id + 1].from] = id + 1;
        if(id == num_of_edges - 1)
            edgeListStart[num_of_vertices] = num_of_edges;
}

void BiconnectedComponents::preprocessEdges(
        const Graph & graph,
        device_vector<int> & extractedEdges,
        device_vector<int> & edgeListStart) const {

    extractedEdges = device_vector<int>(graph.edges.size());
    edgeListStart = device_vector<int>(graph.vertexCount + 1);

    int blocks = (graph.edges.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(blocks);
    dim3 blockDim(BLOCK_SIZE);

    _kernel_extractEdges<<<gridDim, blockDim>>>(
            pointer(graph.edges),
            pointer(extractedEdges),
            pointer(edgeListStart),
            (int)graph.edges.size(),
            graph.vertexCount);

}

/* BFS */
__global__ void _kernel_BFS(
        int * extractedEdges, 
        int * edgeListStart, 
        int * parent, 
        int * distance, 
        int vertexCount, 
        int curr_level) {
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= vertexCount || distance[id] != curr_level) 
        return;

    for(int i = edgeListStart[id], i < edgeListStart[id + 1]; i++)
        if(distance[i] == INF) {
            distance[i] = curr_level + 1;
            parent[i] = id;
        }
}

void BiconnectedComponents::BFS(
        const Graph & graph,
        const device_vector<int> & extractedEdges,
        const device_vector<int> & edgeListStart,
        device_vector<int> & parent,
        device_vector<int> & distance) const {

        parent = device_vector<int>(graph.vertexCount);
        distance = device_vector<int>(graph.vertexCount, INF);
        parent[0] = -1;
        distance[0] = 0;

        device_vector<bool> finished(1, false);

        for(int curr = 0; !finished[0]; curr++)
            _kernel_BFS<<<gridDim, blockDim>>>(
                    pointer(extractedEdges),
                    pointer(edgeListStart),
                    pointer(parent),
                    pointer(distance),
                    graph.vertexCount,
                    curr);
}


void BiconnectedComponents::computeTreeFunctions(
        const Graph & graph,
        device_vector<int> & preorder,
        device_vector<int> & descendantsCount,
        device_vector<int> & low,
        device_vector<int> & high,
        device_vector<int> & parent) const {}

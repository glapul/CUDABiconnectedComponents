#pragma once
#include "BiconnectedComponents.h"
const int BLOCK_SIZE = 1024,
          INF        = 1<<30;

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
        device_vector<int> & edgeListStart) {

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
        const int * edges,
        const int * edgeListStart,
        int * parent,
        int * distance,
        bool * finished,
        int vertexCount,
        int curr_level) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= vertexCount || distance[id] != curr_level)
        return;

    for(int i = edgeListStart[id]; i < edgeListStart[id + 1]; i++)
        if(distance[edges[i]] == INF) {
            distance[edges[i]] = curr_level + 1;
            parent[edges[i]] = id;
            *finished = false;
        }
}

int BiconnectedComponents::BFS(
        const Graph & graph,
        const device_vector<int> & extractedEdges,
        const device_vector<int> & edgeListStart,
        device_vector<int> & parent,
        device_vector<int> & distance) {

    parent = device_vector<int>(graph.vertexCount);
    distance = device_vector<int>(graph.vertexCount, INF);
    parent[0] = -1;
    distance[0] = 0;

    device_vector<bool> finished(1, false);

    dim3 gridDim((graph.vertexCount + BLOCK_SIZE - 1)/BLOCK_SIZE),
         blockDim((BLOCK_SIZE));

    int curr = 0;
    while(!finished[0]) {
        finished[0] = true;
        _kernel_BFS<<<gridDim, blockDim>>>(
                pointer(extractedEdges),
                pointer(edgeListStart),
                pointer(parent),
                pointer(distance),
                pointer(finished),
                graph.vertexCount,
                curr);
        curr++;
    }
    return curr - 1;
}
/* computeDescendantsCount */
__global__ void _kernel_computeDescendantsCount(
        const int * edges,
        const int * edgeListStart,
        const int * parent,
        const int * distance,
        int * descendantsCount,
        int vertexCount,
        int curr_level) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= vertexCount || distance[id] != curr_level)
        return;

    descendantsCount[id] = 1;
    for(int i = edgeListStart[id]; i < edgeListStart[id + 1]; i++)
        if(distance[edges[i]] == curr_level + 1 && parent[edges[i]] == id)
            descendantsCount[id] += descendantsCount[edges[i]];
}
void BiconnectedComponents::computeDescendantsCount(
        const Graph & graph,
        const device_vector<int> & extractedEdges,
        const device_vector<int> & edgeListStart,
        const device_vector<int> & parent,
        const device_vector<int> & distance,
        device_vector<int> & descendantsCount,
        int maxDistance) {

    dim3 gridDim((graph.vertexCount + BLOCK_SIZE - 1)/BLOCK_SIZE),
         blockDim((BLOCK_SIZE));

    descendantsCount = device_vector<int>(graph.vertexCount);
    for(int curr = maxDistance; curr >= 0; curr--)
            _kernel_computeDescendantsCount<<<gridDim, blockDim>>>(
                pointer(extractedEdges),
                pointer(edgeListStart),
                pointer(parent),
                pointer(distance),
                pointer(descendantsCount),
                graph.vertexCount,
                curr);
}
/*  compute Preorder */
__global__ void _kernel_computePreorder(
        const int * edges,
        const int * edgeListStart,
        const int * parent,
        const int * distance,
        const int * descendantsCount,
        int * preorder,
        int vertexCount,
        int curr_level) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= vertexCount || distance[id] != curr_level)
        return;
    int offset = preorder[id];
    for(int i = edgeListStart[id]; i < edgeListStart[id + 1]; i++)
        if(distance[edges[i]] == curr_level + 1 && parent[edges[i]] == id) {
            preorder[edges[i]] = offset;
            offset += descendantsCount[edges[i]];
        }
}
void BiconnectedComponents::computePreorder(
        const Graph & graph,
        const device_vector<int> & extractedEdges,
        const device_vector<int> & edgeListStart,
        const device_vector<int> & parent,
        const device_vector<int> & distance,
        const device_vector<int> & descendantsCount,
        device_vector<int> & preorder,
        int maxDistance) {

    dim3 gridDim((graph.vertexCount + BLOCK_SIZE - 1)/BLOCK_SIZE),
         blockDim((BLOCK_SIZE));

    preorder = device_vector<int>(graph.vertexCount);
    for(int curr = 0; curr <= maxDistance; curr++)
            _kernel_computePreorder<<<gridDim, blockDim>>>(
                pointer(extractedEdges),
                pointer(edgeListStart),
                pointer(parent),
                pointer(distance),
                pointer(descendantsCount),
                pointer(preorder),
                graph.vertexCount,
                curr);
}
#define maxi(x, y) {if(y > x) x = y;}
#define mini(x, y) {if(y < x) x = y;}

/* compute Low and High */
__global__ void _kernel_computeLowHigh(
        const int * edges,
        const int * edgeListStart,
        const int * parent,
        const int * distance,
        const int * descendantsCount,
        const int * preorder,
        int * low,
        int * high,
        int vertexCount,
        int curr_level) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= vertexCount || distance[id] != curr_level)
        return;
    low[id] = high[id] = preorder[id];
    for(int i = edgeListStart[id]; i < edgeListStart[id + 1]; i++) {
        if(distance[edges[i]] == curr_level + 1 && parent[edges[i]] == id) {
            mini(low[id], low[edges[i]]);
            maxi(high[id], high[edges[i]]);
        }
        else {
            mini(low[id], preorder[edges[i]]);
            maxi(high[id], preorder[edges[i]]);
        }
    }
}
void BiconnectedComponents::computeLowHigh(
        const Graph & graph,
        const device_vector<int> & extractedEdges,
        const device_vector<int> & edgeListStart,
        const device_vector<int> & parent,
        const device_vector<int> & distance,
        const device_vector<int> & descendantsCount,
        const device_vector<int> & preorder,
        device_vector<int> & low,
        device_vector<int> & high,
        int maxDistance) {

    dim3 gridDim((graph.vertexCount + BLOCK_SIZE - 1)/BLOCK_SIZE),
         blockDim((BLOCK_SIZE));

    low = device_vector<int>(graph.vertexCount);
    high = device_vector<int>(graph.vertexCount);
    for(int curr = maxDistance; curr >=0; curr--)
            _kernel_computeLowHigh<<<gridDim, blockDim>>>(
                pointer(extractedEdges),
                pointer(edgeListStart),
                pointer(parent),
                pointer(distance),
                pointer(descendantsCount),
                pointer(preorder),
                pointer(low),
                pointer(high),
                graph.vertexCount,
                curr);
}
void BiconnectedComponents::computeTreeFunctions(
        const Graph & graph,
        device_vector<int> & preorder,
        device_vector<int> & descendantsCount,
        device_vector<int> & low,
        device_vector<int> & high,
        device_vector<int> & parent) {

    device_vector<int> extractedEdges,
                       edgeListStart,
                       distance;

    preprocessEdges(graph, extractedEdges, edgeListStart);

    int maxDistance = BFS(
            graph,
            extractedEdges,
            edgeListStart,
            parent,
            distance);

    computeDescendantsCount(
            graph,
            extractedEdges,
            edgeListStart,
            parent,
            distance,
            descendantsCount,
            maxDistance);

    computePreorder(
            graph,
            extractedEdges,
            edgeListStart,
            parent,
            distance,
            descendantsCount,
            preorder,
            maxDistance);

    computeLowHigh(
            graph,
            extractedEdges,
            edgeListStart,
            parent,
            distance,
            descendantsCount,
            preorder,
            low,
            high,
            maxDistance);
}

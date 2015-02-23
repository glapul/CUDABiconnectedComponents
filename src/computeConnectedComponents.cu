#include "config.h"
#ifdef computeConnectedComponents_IMPLEMENTED
#include "BiconnectedComponents.h"

using thrust::host_vector;
using thrust::device_vector;

#define pointer(x) (thrust::raw_pointer_cast(& ((x) [0])))
#define ceilDiv(x, y) ((x) + (y) - 1) / (y)

namespace {

const int NUM_THREADS = 1024;

__global__
void initParents(int* parent, int n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        parent[i] = i;
    }
}

__global__
void hooking(const Edge* edges, int* parent, bool* change, int ed, int mode) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < ed) {
        int u = parent[edges[i].from];
        int v = parent[edges[i].to];
        if (u != v) {
            int max = v * (v > u) + u * (u >= v);
            int min = v * (v < u) + u * (u <= v);
            parent[mode * max + (1 - mode) * min] = mode * min + (1 - mode) * max;
            *change = true;
        }
    }
}

__global__
void jumping(int* parent, bool* change, int vert) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < vert && parent[i] != parent[parent[i]]) {
        parent[i] = parent[parent[i]];
        *change = true;
    }
}

}

void BiconnectedComponents::computeConnectedComponents(
        const Graph& graph,
        device_vector<int>& components) {
    int vert = graph.vertexCount;
    int ed = graph.edgeCount();

    components = device_vector<int>(vert);
    initParents<<<ceilDiv(vert, NUM_THREADS), NUM_THREADS>>>(
        pointer(components),
        vert);
    cudaDeviceSynchronize();

    device_vector<bool> marked(ed, false);
    device_vector<bool> change(1, true);

    int mode = 1;

    while(change[0]) {
        change[0] = false;
        mode = 1 - mode;

        hooking<<<ceilDiv(ed, NUM_THREADS), NUM_THREADS>>>(
            pointer(graph.edges),
            pointer(components),
            pointer(change),
            ed,
            mode);
        cudaDeviceSynchronize();

        device_vector<bool> jumpChange(1, true);
        while(jumpChange[0]) {
            jumpChange[0] = false;

            jumping<<<ceilDiv(vert, NUM_THREADS), NUM_THREADS>>>(
                pointer(components),
                pointer(jumpChange),
                vert);
            cudaDeviceSynchronize();
        }
    }
}

#endif

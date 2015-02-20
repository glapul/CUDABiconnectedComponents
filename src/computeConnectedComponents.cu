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
void hooking(const Edge* edges, int* parent, bool* marked, bool* change, int ed, int mode) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < ed && !marked[i]) {
        int u = parent[edges[i].from];
        int v = parent[edges[i].to];
        if (u != v) {
            int max = v * (v > u) + u * (u >= v);
            int min = v * (v < u) + u * (u <= v);
            if (mode == 0) {
                parent[min] = max;
            } else {
                parent[max] = min;
            }
            *change = true;
        } else {
            marked[i] = true;
        }
    }
}

__global__
void jumping(int* parent, bool* change, int vert) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < vert) {
        *change |= (parent[i] != parent[parent[i]]);
        parent[i] = parent[parent[i]];
    }
}

}

void BiconnectedComponents::computeConnectedComponents(
        const Graph& graph,
        device_vector<int>& components) {
    int vert = graph.vertexCount;
    int ed = graph.edgeCount();

    device_vector<int> parent(vert);
    initParents<<<ceilDiv(vert, NUM_THREADS), NUM_THREADS>>>(
        pointer(parent),
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
            pointer(parent),
            pointer(marked),
            pointer(change),
            ed,
            mode);
        cudaDeviceSynchronize();

        device_vector<bool> jumpChange(1, true);
        while(jumpChange[0]) {
            jumpChange[0] = false;

            jumping<<<ceilDiv(vert, NUM_THREADS), NUM_THREADS>>>(
                pointer(parent),
                pointer(jumpChange),
                vert);
        }
    }

    components = parent;
}

#endif

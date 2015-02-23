#include "config.h"
#ifdef extendRelation_IMPLEMENTED

#include "BiconnectedComponents.h"
#include "Graph.h"

#define pointer(x) (thrust::raw_pointer_cast(& ((x) [0])))
#define ceilDiv(x, y) ((x) + (y) - 1) / (y)

const int NUM_THREADS = 1024;

namespace {

__global__
void edgeKernel(const Edge * edges, int edgeCount, const int * preorder,
		const int * partial, int * components) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	register int w;
	if (preorder[edges[i].to] < preorder[edges[i].from])
		w = edges[i].from;
	else
		w = edges[i].to;

	components[i] = partial[w];
}

} // namespace

void BiconnectedComponents::extendRelation(
		const Graph & graph,
		const device_vector<int> & preorder,
		const device_vector<int> & partial,
		device_vector<int> & components) {

	int edgeCount = graph.edgeCount();
	components = device_vector<int>(edgeCount);
	
	edgeKernel<<<ceilDiv(edgeCount, NUM_THREADS), NUM_THREADS>>>(
            pointer(graph.edges),
			edgeCount,
            pointer(preorder),
            pointer(partial),
            pointer(components));
	cudaDeviceSynchronize();
}


#endif // IMPLEMENTED

#include "config.h"
#ifdef createAuxiliaryGraph_IMPLEMENTED

#include <thrust/scan.h>
#include "BiconnectedComponents.h"
#include "Graph.h"

#define UNRELATED(x,y) (preorder[x] + nd[x] <= preorder[y])
#define JOINS(x,y) (preorder[x] != 1 && (low[y] < preorder[x] || high[y] >= preorder[x] + nd[x]))

#define pointer(x) (thrust::raw_pointer_cast(& ((x) [0])))
#define ceilDiv(x, y) ((x) + (y) - 1) / (y)

const int NUM_THREADS = 1024;

__global__
void initKernel(int * active, const Edge * edges, int edgeCount, const int * parent,
        const int * preorder, const int * nd, const int * low, const int * high) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < edgeCount) {
		register int lower, upper;
		if (preorder[edges[i].to] < preorder[edges[i].from]) {
			upper = edges[i].to;
			lower = edges[i].from;
		}
		else {
			upper = edges[i].from;
			lower = edges[i].to;
		}
		if (parent[lower] == upper) {
			if (JOINS(upper, lower))
				active[i+1] = 1;
			else
				active[i+1] = 0;
		}
		else {
			if (UNRELATED(upper, lower))
				active[i+1] = 1;
			else
				active[i+1] = 0;
		}
	}
}

__global__
void finalKernel(const int * prefs, const Edge * edges, int edgeCount, Edge * newEdges) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < edgeCount) {
		if (prefs[i+1] > prefs[i])
			newEdges[prefs[i]] = edges[i];
	}
}

void BiconnectedComponents::createAuxiliaryGraph(
		const Graph & graph,
		const device_vector<int> & parent,
		const device_vector<int> & preorder,
		const device_vector<int> & nd,
		const device_vector<int> & low,
		const device_vector<int> & high,
		Graph & auxiliaryGraph) {
	
	int edgeCount = graph.edgeCount();
	device_vector<int> active = device_vector<int>(edgeCount + 1);
	initKernel<<<ceilDiv(edgeCount, NUM_THREADS), NUM_THREADS>>>(
			pointer(active),
			pointer(graph.edges),
			edgeCount,
			pointer(parent),
			pointer(preorder),
			pointer(nd),
			pointer(low),
			pointer(high)
			);

	cudaDeviceSynchronize();
	// prefsum
	active[0] = 0;
    thrust::inclusive_scan(active.begin(), active.end(), active.begin());

	int newEdgeCount = active[edgeCount];
	device_vector<Edge> newEdges = device_vector<Edge>(newEdgeCount);

	finalKernel<<<ceilDiv(edgeCount, NUM_THREADS), NUM_THREADS>>>(
			pointer(active),
			pointer(graph.edges),
			edgeCount,
			pointer(newEdges)
			);
	cudaDeviceSynchronize();
	
	auxiliaryGraph = Graph(graph.vertexCount, newEdges);
}

#endif // IMPLEMENTED

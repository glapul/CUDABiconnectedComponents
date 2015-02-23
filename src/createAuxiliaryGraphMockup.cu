#include "config.h"
#ifndef createAuxiliaryGraph_IMPLEMENTED

#include "BiconnectedComponents.h"
#include "Graph.h"

#define UNRELATED(x,y) (preorder[x] + nd[x] <= preorder[y])
#define JOINS(x,y) (preorder[x] != 1 && (low[y] < preorder[x] || high[y] >= preorder[x] + nd[x]))

void BiconnectedComponents::createAuxiliaryGraph(
		const Graph & graph,
		const device_vector<int> & parent,
		const device_vector<int> & preorder,
		const device_vector<int> & nd,
		const device_vector<int> & low,
		const device_vector<int> & high,
		Graph & auxiliaryGraph) {
	
	int edgeCount = graph.edgeCount();
	host_vector<int> active = host_vector<int>(edgeCount + 1);
	for (int i = 0; i < edgeCount; ++i) {
		const Edge & e = graph.edges[i];
		// is e a tree edge?
		int lower, upper;
		if (preorder[e.to] < preorder[e.from]) {
			upper = e.to;
			lower = e.from;
		}
		else {
			upper = e.from;
			lower = e.to;
		}
		if (parent[lower] == upper) {
			// e is a tree edge
			// upper -> lower
			if (JOINS(upper, lower))
				active[i+1] = 1;
			else
				active[i+1] = 0;
		}
		else {
			// e isn't a tree edge
			if (UNRELATED(upper, lower))
				active[i+1] = 1; // +1 is important!
			else
				active[i+1] = 0;
		}
	}

	// prefsum
	active[0] = 0;
	for (int i = 1; i <= edgeCount; ++i)
		active[i] += active[i-1];

	int newEdgeCount = active[edgeCount];
	host_vector<Edge> newEdges = host_vector<Edge>(newEdgeCount);

	for (int i = 0; i < edgeCount; ++i) {
		if (active[i+1] > active[i])
			newEdges(active[i]) = graph.edges[i];
	}
	
	auxiliaryGraph = Graph(graph.vertexCount, newEdges);
}

#endif // IMPLEMENTED

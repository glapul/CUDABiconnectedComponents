#include "config.h"
#ifndef createAuxiliaryGraph_IMPLEMENTED

#include "BiconnectedComponents.h"
#include "Graph.h"

#define UNRELATED(x,y) (x >= 0 && y >= 0 && preorder[x] + nd[x] <= preorder[y])
#define JOINS(x,y) (low[y] < preorder[x] || high[y] >= preorder[x] + nd[x])

void BiconnectedComponents::createAuxiliaryGraph(
		const Graph & graph,
		const device_vector<int> & preorder,
		const device_vector<int> & parent,
		const device_vector<int> & nd,
		const device_vector<int> & low,
		const device_vector<int> & high,
		Graph & auxiliaryGraph,
		device_vector<std::pair<int, int>> & mapping) {
	
	int edgeCount = graph.edgeCount();
	host_vector<Edge> newEdges = host_vector<Edge>(edgeCount);
	for (int i = 0; i < edgeCount; ++i) {
		Edge & e = graph.edges[i];
		// is e a tree edge?
		bool down_tree = (e.from == parent[e.to]);
		bool up_tree = (e.to == parent[e.from]);
		if (!(down_tree || up_tree)) {
			if (UNRELATED(parent[e.to], parent[e.from]))
				newEdges[i] = Edge(from, to);
		}
		if (down_tree) {
			if (parent[from] >= 0 && JOINS(from, to))
				newEdges[i] = Edge(from, to);
		}
		if (up_tree) {
			if (parent[to] >= 0 && JOINS(to, from))
				newEdges[i] = Edge(to, from);
		}
	}

	int counter = 0;
	for (int i = 0; i < edgeCount; ++i) {
		if (!newEdges[i].is_zero())
			++counter;
	}
	host_vector<Edge> result = host_vector<Edge>(counter);
	int ptr = 0;
	for (int i = 0; i < edgeCount; ++i) {
		if (!newEdges[i].is_zero())
			result[ptr++] = newEdges[i];
	}
	auxiliaryGraph = Graph(graph.vertexCount, result);
}

#endif // IMPLEMENTED

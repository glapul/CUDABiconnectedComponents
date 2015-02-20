#ifndef GRAPH_H
#define GRAPH_H

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct Edge {
	int from, to;
	int rev;
    Edge(){}
    Edge(int from, int to, int rev)
        :from(from), to(to), rev(rev){}
};

struct Graph {
	Graph(int vertexCount)
        : vertexCount(vertexCount) {}
    Graph(int vertexCount, thrust::host_vector<Edge> _edges)
        : vertexCount(vertexCount) {
        edges = _edges;
    }

	int vertexCount;
	thrust::device_vector<Edge> edges;
};
#endif

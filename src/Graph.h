#pragma once

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using thrust::device_vector;
using thrust::host_vector;

struct Edge {
	int from, to;
	int rev;
    Edge() {}
    Edge(int from, int to, int rev)
        : from(from), 
        to(to), 
        rev(rev) {}
};

struct Graph {
	Graph(int vertexCount)
        : vertexCount(vertexCount) {}
        
    Graph(int vertexCount, host_vector<Edge> edges)
        : vertexCount(vertexCount),
    	edges(edges) {}

    int edgeCount() const {
    	return edges.size();
    }

	int vertexCount;
	device_vector<Edge> edges;
};

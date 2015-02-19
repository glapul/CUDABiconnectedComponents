#pragma once

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct Edge {
	int from, to;
	int rev;
};

struct Graph {
	Graph(int vertexCount)
        : vertexCount(vertexCount) {}
        
    Graph(int vertexCount, thrust::host_vector<Edge> edges)
        : vertexCount(vertexCount),
    	edges(edges) {}

	int vertexCount;
	thrust::device_vector<Edge> edges;
};

#pragma once

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using thrust::device_vector;
using thrust::host_vector;

struct Edge {
	int from, to;
	int rev;
    Edge() 
	: from(0),
	to(0),
	rev(0) {}
	
    Edge(int from, int to, int rev = 0)
        : from(from), 
        to(to), 
        rev(rev) {}

	bool is_zero() const {
		return from == 0 && to == 0;
	}
};

struct Graph {
	Graph(int vertexCount)
        : vertexCount(vertexCount) {}
        
    Graph(int vertexCount, device_vector<Edge> edges)
        : vertexCount(vertexCount),
    	edges(edges) {}

    int edgeCount() const {
    	return edges.size();
    }

	int vertexCount;
	device_vector<Edge> edges;
};

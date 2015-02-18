#include "BiconnectedComponents.h"
#include<bits/stdc++.h>
using namespace std;
struct FindJoin {
    vector<int> parent;
    FindJoin(int n)
        : parent(vector<int>(n)) {
        for(int i = 0; i < parent.size(); i++)
            parent[i] = i;
    }
    int find(int x) {
        return x == parent[x] ? x : parent[x] = find(parent[x]);
    }
    void join(int x, int y) {
        parent[find(x)] = find(y);
    }
};
void BiconnectedComponents::computeSpanningTree(
        const Graph & graph,
        device_vector<bool> & inSpanningTree) const {

    FindJoin fu = FindJoin(graph.vertexCount);
    host_vector<Edge> host_edges = graph.edges;
    host_vector<bool> host_inSpanningTree = inSpanningTree;

    for(int i = 0; i < host_edges.size(); i++) {
        if(fu.find(host_edges[i].from) != fu.find(host_edges[i].to)) {
            fu.join(host_edges[i].from, host_edges[i].to);
            host_inSpanningTree[i] = host_inSpanningTree[host_edges[i].rev] = true;
        }
    }
    inSpanningTree = host_inSpanningTree;
}

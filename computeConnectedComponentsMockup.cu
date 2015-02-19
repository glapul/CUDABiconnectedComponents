
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

private void BiconnectedComponents::computeConnectedComponents(
        const Graph & graph,
        device_vector<int> & components) const {

    FindJoin fu = FindJoin(graph.vertexCount);
    host_vector<Edge> host_edges = graph.edges;

    int sccs = graph.vertexCount;
    host_vector<int> host_components = host_vector<int>(sccs);

    for(int i = 0; i < host_edges.size(); i++)
        if(fu.find(host_edges[i].from) != fu.find(host_edges[i].to)) {
            fu.join(host_edges[i].from, host_edges[i].to);
            sccs--;
        }
    int wsk = 0;
    unordered_map<int,int> m;
    for(int i = 0; i < graph.vertexCount; i++) {
        if(m.count(fu.find(i)) == 0)
            m[fu.find(i)] = wsk++;
        host_components[i] = m[fu.find(i)];
    }
    components = host_components;
}

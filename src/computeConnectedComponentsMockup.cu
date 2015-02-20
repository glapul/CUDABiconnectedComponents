#include "config.h"
#ifndef computeConnectedComponents_IMPLEMENTED

#include<bits/stdc++.h>
#include "BiconnectedComponents.h"
using namespace std;
#include "helper.h"
void BiconnectedComponents::computeConnectedComponents(
        const Graph & graph,
        device_vector<int> & components) {

    FindJoin fu = FindJoin(graph.vertexCount);
    thrust::host_vector<Edge> host_edges = graph.edges;

    int sccs = graph.vertexCount;
    thrust::host_vector<int> host_components = thrust::host_vector<int>(sccs);

    for(int i = 0; i < host_edges.size(); i++)
        if(fu.find(host_edges[i].from) != fu.find(host_edges[i].to)) {
            fu.join(host_edges[i].from, host_edges[i].to);
            sccs--;
        }
    int wsk = 0;
    map<int,int> m;
    for(int i = 0; i < graph.vertexCount; i++) {
        if(m.count(fu.find(i)) == 0)
            m[fu.find(i)] = wsk++;
        host_components[i] = m[fu.find(i)];
    }
    components = host_components;
}

#endif // IMPLEMENTED

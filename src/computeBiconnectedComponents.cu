#include <cstdio>

#include "BiconnectedComponents.h"
#include "Graph.h"

void BiconnectedComponents::computeBiconnectedComponents(
        const Graph & graph,
        device_vector<int> & components) {
    
    device_vector<int> parent;
    device_vector<int> preorder;
    device_vector<int> low;
    device_vector<int> high;
    device_vector<int> nd;
    device_vector<int> partial;
    Graph auxiliaryGraph(0);
    printf("jeb 1\n");

    computeTreeFunctions(
            graph,
            preorder,
            nd,
            low,
            high,
            parent);
    printf("jeb 2\n");
    createAuxiliaryGraph(
            graph,
            parent,
            preorder,
            nd,
            low,
            high,
            auxiliaryGraph);
    printf("jeb 3\n");
    computeConnectedComponents(
            auxiliaryGraph,
            partial);
    printf("jeb 4\n");
    extendRelation(
            graph,
            preorder,
            partial,
            components);
    printf("jeb 5\n");
    // wtf have I just written
}

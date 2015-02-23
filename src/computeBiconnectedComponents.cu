
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

    computeTreeFunctions(
            graph,
            preorder,
            nd,
            low,
            high,
            parent);
    createAuxiliaryGraph(
            graph,
            parent,
            preorder,
            nd,
            low,
            high,
            auxiliaryGraph);
    computeConnectedComponents(
            auxiliaryGraph,
            partial);
    extendRelation(
            graph,
            parent,
            preorder,
            partial,
            components);
    // wtf have I just written
}

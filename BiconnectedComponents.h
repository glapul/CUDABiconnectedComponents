#ifndef BICONNNECTED_COMPONENTS_H
#define BICONNNECTED_COMPONENTS_H
#include "Graph.h"

using thrust::device_vector;

class BiconnectedComponents {
public:
    static void computeTreeFunctions(
        const Graph & graph,
        device_vector<int> & preorder,
        device_vector<int> & descendantsCount,
        device_vector<int> & low,
        device_vector<int> & high,
        device_vector<int> & parent);

	static void createAuxiliaryGraph(
		const Graph& graph,
		const device_vector<int>& preorder,
		const device_vector<int>& nd,
		const device_vector<int>& low,
		const device_vector<int>& high,
		Graph& auxiliaryGraph,
		device_vector<std::pair<int, int> >& mapping);

	static void computeConnectedComponents(
		const Graph& graph,
		device_vector<int>& components);

	static void computeBiconnectedComponents(
		const Graph& graph,
		device_vector<int>& components);
private:
	BiconnectedComponents(); // private constuctor - all methods are static
	~BiconnectedComponents();

    static void preprocessEdges(
            const Graph & graph,
            device_vector<int> & extractedEdges,
            device_vector<int> & edgeListStart);

    static int BFS(
            const Graph & graph,
            const device_vector<int> & extractedEdges,
            const device_vector<int> & edgeListStart,
            device_vector<int> & parent,
            device_vector<int> & distance);

    static void computeDescendantsCount(
            const Graph & graph,
            const device_vector<int> & extractedEdges,
            const device_vector<int> & edgeListStart,
            const device_vector<int> & parent,
            const device_vector<int> & distance,
            device_vector<int> & descendantsCount,
            int maxDistance);

    static void computePreorder(
            const Graph & graph,
            const device_vector<int> & extractedEdges,
            const device_vector<int> & edgeListStart,
            const device_vector<int> & parent,
            const device_vector<int> & distance,
            const device_vector<int> & descendantsCount,
            device_vector<int> & preorder,
            int maxDistance);

    static void computeLowHigh(
            const Graph & graph,
            const device_vector<int> & extractedEdges,
            const device_vector<int> & edgeListStart,
            const device_vector<int> & parent,
            const device_vector<int> & distance,
            const device_vector<int> & descendantsCount,
            const device_vector<int> & preorder,
            device_vector<int> & low,
            device_vector<int> & high,
            int maxDistance);
};
#endif

#include "Graph.h"
#include "BiconnectedComponents.h"

#include "computeTreeFunctions.cu"

#define computeConnectedComponentsMockup
#ifdef computeSpanningTreeMockup
#include "computeConnectedComponentsMockup.cu"
#else
#include "computeConnectedComponents.cu"
#endif

void BiconnectedComponents::computeBiconnectedComponents(
        const Graph & graph,
        device_vector<int> & components) const {}

void BiconnectedComponents::createAuxiliaryGraph(
		const Graph& graph,
		const device_vector<int>& preorder,
		const device_vector<int>& nd,
		const device_vector<int>& low,
		const device_vector<int>& high,
		Graph& auxiliaryGraph,
		device_vector<std::pair<int, int> >& mapping) const {}

int main() {
	return 0;
}

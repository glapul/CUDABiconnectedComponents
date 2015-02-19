#pragma once

#include "Graph.h"

using thrust::device_vector;

class BiconnectedComponents {
public:
	BiconnectedComponents();
	~BiconnectedComponents();

    void computeTreeFunctions(
        const Graph & graph,
        device_vector<int> & preorder,
        device_vector<int> & descendantsCount,
        device_vector<int> & low,
        device_vector<int> & high,
        device_vector<int> & parent) const;

	void createAuxiliaryGraph(
		const Graph& graph,
		const device_vector<int>& preorder,
		const device_vector<int>& nd,
		const device_vector<int>& low,
		const device_vector<int>& high,
		Graph& auxiliaryGraph,
		device_vector<std::pair<int, int>>& mapping) const;

	void computeConnectedComponents(
		const Graph& graph,
		device_vector<int>& components) const;

	void computeBiconnectedComponents(
		const Graph& graph,
		device_vector<int>& components) const;
};

#pragma once

#include "Graph.h"

using thrust::device_vector;

class BiconnectedComponents {
public:
	BiconnectedComponents();
	~BiconnectedComponents();

	void computeSpanningTree(
		const Graph& graph,
		device_vector<bool>& inSpanningTree) const;

	void computeTreeFunctions(
		const Graph& graph,
		const device_vector<bool>& inSpanningTree,
		device_vector<int>& preorder,
		device_vector<int>& descendantsCount) const;

	void computeLowHigh(
		const Graph& graph,
		const device_vector<bool>& inSpanningTree,
		const device_vector<int>& preorder,
		device_vector<int>& low,
		device_vector<int>& high) const;

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
private:
    //should be set to right values in constructor
    CUresult _cu_res;
    CUdevice _cu_device;
    CUcontext _cu_context;
    CUmodule _cu_module;
};
// a macro for executing things on cuda
#define _cu_exec(x) _cu_res = x; if (_cu_res != CUDA_SUCCESS) {puts(#x); exit(1);}
#define _cu_exec_msg(x,msg) _cu_res = x; if ( _cu_res != CUDA_SUCCESS) {printf(msg);printf("\n"); exit(1);}

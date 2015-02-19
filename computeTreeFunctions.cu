const int BLOCK_SIZE = 32,
          INF = 1<<30;

#define pointer(x) (thrust::raw_pointer_cast(& ((x) [0])))

__global__ void _kernel_extractEdges(
            Edge * edges,
            int * extractedEdges,
            int * edgeListStart,
            int num_of_edges,
            int num_of_vertices) {

        int id = threadIdx + blockIdx * blockDim.x;
        if(id >= num_of_edges)
            return;
        extractedEdges[id] = edges[id].to;
        else if (edges[id].from != edges[id + 1].from)
            edgeListStart[edges[id + 1].from] = id + 1;
        if(id == num_of_edges - 1)
            edgeListStart[num_of_vertices] = num_of_edges;
}

private void BiconnectedComponents::preprocessEdges(
        const Graph & graph,
        device_vector<int> & extractedEdges,
        device_vector<int> & edgeListStart) const {

    extractedEdges = device_vector<int>(graph.edges.size());
    edgeListStart = device_vector<int>(graph.vertexCount + 1);

    int blocks = (graph.edges.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(blocks);
    dim3 blockDim(BLOCK_SIZE);

    _kernel_extractEdges<<<gridDim, blockDim>>(
            pointer(graph.edges),
            pointer(extractedEdges),
            pointer(edgeListStart),
            (int)graph.edges.size(),
            graph.vertexCount);

}

public void BiconnectedComponents::computeTreeFunctions(
        const Graph & graph,
        device_vector<int> & preorder,
        device_vector<int> & descendantsCount,
        device_vector<int> & low,
        device_vector<int> & high,
        device_vector<int> & parent) const {}

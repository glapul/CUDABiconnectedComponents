/* Do zrobienia :
 * sleep jesli vis_count == 0
 * skakanie nad rozpatrzonymi krawedziami
 */
extern "C++" {
    __global__
        void _kernel_computeSpanningTree (
                device_vector<Edge> & edges,
                device_vector<bool> & inSpanningTree,
                device_vector<bool> & visited,
                int thread_interval_length,
                int num_of_edges ) {

            int nr    = threadIdx.x  + blockIdx.x * blockDim.x,
                first = nr * thread_interval_length,
                last  = min(first + thread_interval_length, num_of_edges);

            if(nr == 0)
                    visited[0] = true;
            int vis_count = 0;
            while(vis_count < last - first) {
                vis_count = 0;
                for(int i = first; i < last; i++) {
                    if(visited[edges[i].from] && !visited[edges[i].to]) {
                        visited[edges[i].to] = i + 1;
                        if(visited[edges[i].to] == i + 1 ) // make sure the write actually occured
                            inSpanningTree[i] = inSpanningTree[edges[i].rev] = true;
                    }
                    vis_count += visited[edges[i].to] != 0;
                }
            }
        }
}

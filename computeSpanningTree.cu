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

            ////////////////////////////////////////////////
            //skipping data structure
            int skipped = 0;
            int * _skip = new int [last - first],
                * _prev = new int [last - first];
            for(int i = 0; i < last-first; i++) {
                _skip[i] = 1;
                _prev[i] = -1;
            }
            _prev[0] = last-first - 1;
            _skip[last - first - 1] = - (last - first - 1);
            #define skip(i) (i + _skip[i-first])
            #define prev(i) (i + _prev[i-first])
            #define del(x) {if(skipped < last -first - 1){_prev[skip(x)] += _prev[x]; _skip[prev(x)] += _skip[x]; _skip[x] = _prev[x] = -1<<30;skipped++} else {break;}}
            #define clean {delete _skip; delete_prev;}
            ///////////////////////////////////////////////

            int i = first;
            while(true) {
                if(visited[edges[i].from] && !visited[edges[i].to]) {
                    visited[edges[i].to] = i + 1;
                    if(visited[edges[i].to] == i + 1 ) // make sure the write actually occured
                        inSpanningTree[i] = inSpanningTree[edges[i].rev] = true;
                    int last = i;
                    i = skip(i);
                    del(last);
                    continue;
                }
                if(visited[edges[i].to]) {
                    i = skip(i);
                    del(last);
                    continue;
                }
                i = skip(i);
            }
            clean;
        }
}

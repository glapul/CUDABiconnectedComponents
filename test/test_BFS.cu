#define private public
#include "../BiconnectedComponents.cu"
#include "../generator.cu"

int main() {
    Graph G = generate_random_tree(5);
    cout << G << endl;
    BiconnectedComponents BC;
    thrust::device_vector<int> extractedEdges, edgeListStart, parent, distance;
    BC.preprocessEdges(G, extractedEdges, edgeListStart);
    BC.BFS(G, extractedEdges, edgeListStart, parent, distance);
    cout << "panret\n" << parent  <<"\n";
    cout << "distance\n" << distance << "\n";
}

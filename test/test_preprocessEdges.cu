#define private public
#include "../BiconnectedComponents.cu"
#include "../generator.cu"


int main() {
    Graph G = generate_random_tree(10);
    cout << G << endl;
    BiconnectedComponents BC;
    thrust::device_vector<int> extractedEdges, edgeListStart;
    BC.preprocessEdges(G, extractedEdges, edgeListStart);
    cout << "extracted_edges \n"<<extractedEdges;
    cout <<"edgeListStart \n" << edgeListStart;
}

#include <cstdio>
#include <iostream>
using namespace std;

#include "../src/BiconnectedComponents.h"

Edge edges[] = {Edge(0,1,1), Edge(1,0,0)};

void test1() {
    device_vector<Edge> vec_edges = device_vector<Edge>(2);
    for (int i = 0; i < 2; ++i)
        vec_edges[i] = edges[i];
    Graph g = Graph(2, vec_edges);
    device_vector<int> result;
    printf("hop\n");
    BiconnectedComponents::computeBiconnectedComponents(g, result);
    printf("i kurwa!\n");
    for (int i = 0; i < 2; ++i)
        cout << result[i] << endl;
}

int main() {
    test1();
}

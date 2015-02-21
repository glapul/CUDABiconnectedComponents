#include "../src/BiconnectedComponents.h"
#include "generator.cu"

#include <iostream>

int main() {
	Graph graph = generate_random_graph(10, 10);
	device_vector<int> result;
	BiconnectedComponents::computeConnectedComponents(graph, result);

	cout << graph << endl;
	cout << result << endl;
}
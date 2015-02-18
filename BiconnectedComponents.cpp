#include "Graph.h"
#include "BiconnectedComponents.h"

#include "constructorAndDestructor.cpp"

//Mockup - wylaczyc
#define computeSpanningTreeMockup
#ifdef computeSpanningTreeMockup
#include "computeSpanningTreeMockup.cpp"
#else
#include "computeSpanningTree.cpp"
#endif

#define computeConnectedComponentsMockup
#ifdef computeSpanningTreeMockup
#include "computeConnectedComponentsMockup.cpp"
#else
#include "computeConnectedComponents.cpp"
#endif
int main() {
	return 0;
}

#include "Graph.h"
#include "BiconnectedComponents.h"


//Mockup - wylaczyc
#define computeSpanningTreeMockup

#ifdef computeSpanningTreeMockup
#include "computeSpanningTreeMockup.cpp"
#else
#include "computeSpanningTree.cpp"
#endif

int main() {
	return 0;
}

#include <thrust/sort.h>

#include "../src/BiconnectedComponents.h"
#include "generator.cu"
#include "testlib.cu"
#include "../src/helper.h"

#include <iostream>
#include <boost/lexical_cast.hpp>

struct TestConnectedComponents : public Test {
    string name;
    int vert;
    int ed;
    TestResult res;

    TestConnectedComponents(string name, int vert, int ed)
        : name(name), vert(vert), ed(ed), res(NOT_CHECKED){}

    bool play() {
        // cout << graph << endl;
        Graph graph = generate_random_graph(vert, ed);

        device_vector<int> comps;
        BiconnectedComponents::computeConnectedComponents(graph, comps);
        host_vector<int> parRes = comps;

        device_vector<int> thrash, beginning;
        BiconnectedComponents::preprocessEdges(graph, thrash, beginning);

        host_vector<Edge> hEdges = graph.edges;
        host_vector<int> hBegin = beginning;
        host_vector<int> seqRes(graph.vertexCount, -1);
        seqComps(hEdges, hBegin, seqRes);

        // cout << comps << endl;

        // cout << seqRes << endl;

        for (int i = 0; i < graph.vertexCount; i++) {
            CHECK(seqRes[i] == seqRes[parRes[i]]);
            CHECK(parRes[i] == parRes[seqRes[i]]);
        }

        res = PASS;
        return true;
    }

    void report() {
        standard_report(name, res);
    }

    private:
        void dfs(
            int x, 
            int comp, 
            const host_vector<Edge>& edges,
            const host_vector<int>& beginning,
            host_vector<int>& result) {
        result[x] = comp;
        for (int i = beginning[x]; i < edges.size() && edges[i].from == x; i++) {
            if (result[edges[i].to] == -1) {
                dfs(edges[i].to, comp, edges, beginning, result);
            }
        }
    }

    void seqComps(
            const host_vector<Edge>& edges,
            const host_vector<int>& beginning,
            host_vector<int>& result) {
        for (int i = 0; i < result.size(); i++) {
            if (result[i] == -1) {
                dfs(i, i, edges, beginning, result);
            }
        }
    }
};

void test_connectedComponents() {
    srand(345);
    TestSuite testSuite = TestSuite("connected components tests");

    for (int i = 0; i < 5; i++) {
        testSuite.addTest(new TestConnectedComponents("n = 10, m = 10", 10, 10));
        testSuite.addTest(new TestConnectedComponents("n = 20, m = 20", 20, 20));
        testSuite.addTest(new TestConnectedComponents("n = 100, m = 100", 100, 100));
        testSuite.addTest(new TestConnectedComponents("n = 1000, m = 1000", 1000, 1000));
        testSuite.addTest(new TestConnectedComponents("n = 10000, m = 10000", 10000, 10000));
        testSuite.addTest(new TestConnectedComponents("n = 1000000, m = 1000000", 1000000, 1000000));
    }

    testSuite.play();
}

int main() {
	test_connectedComponents();
	return 0;
}

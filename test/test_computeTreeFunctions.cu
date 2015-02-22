#include "testlib.cu"
#include "../src/BiconnectedComponents.h"
#include "generator.cu"

class TestPreprocessEdges : public Test {
    public:
    string name;
    Graph g;
    TestResult res;

    TestPreprocessEdges(string name, const Graph & g)
        :name(name), g(g), res(NOT_CHECKED){}

    bool play() {
        VALIDATE(validate_connected_graph(g));

        device_vector<int> extractedEdges,
                           edgeListStart;
        BiconnectedComponents::preprocessEdges(g, extractedEdges, edgeListStart);

        CHECK(extractedEdges.size() == g.edges.size());
        CHECK(edgeListStart.size() == g.vertexCount + 1);
        host_vector<int> ee  = extractedEdges,
                         els = edgeListStart;
        host_vector<Edge> edges = g.edges;

        for(int i = 0; i < edges.size(); i++)
            CHECK(edges[i].to == ee[i]);
        for(int i = 1; i < g.vertexCount; i++) {
            CHECK(edges[els[i]].from == i);
            CHECK(edges[els[i]-1].from < i);
        }
        CHECK(els[0] == 0);
        CHECK(els[g.vertexCount] == edges.size());
        res = PASS;
        return true;
    }
    void report() {
        standard_report(name, res);
    }
};

const int INF = 1<<30;

class TestBFS : public Test {
    public:
    string name;
    TestResult res;
    std::pair<int,int> generator_args;
    
    TestBFS(string name, pair<int,int> args)
        :name(name), res(NOT_CHECKED), generator_args(args){}

    bool play() {
        Graph graph = generate_random_connected_graph(generator_args.first, generator_args.second);
        VALIDATE(validate_connected_graph(graph));

        device_vector<int> extractedEdges,
                           edgeListStart,
                           distance,
                           parent;
        BiconnectedComponents::preprocessEdges(graph, extractedEdges, edgeListStart);
        BiconnectedComponents::BFS(graph, extractedEdges, edgeListStart, parent, distance);
        vector<int> serial_distance = serialBFS(graph, extractedEdges, edgeListStart);
        host_vector<int> parallel_distance = distance,
                         parallel_parent   = parent;
        
        for(int i = 0; i < graph.vertexCount; i++) {
            CHECK(serial_distance[i] == parallel_distance[i]);
            CHECK(i == 0 || (parallel_parent[i] != i && parallel_parent[i] >= 0 && parallel_parent[i] < graph.vertexCount));
            CHECK(i == 0 || parallel_distance[parallel_parent[i]] + 1 == parallel_distance[i]);
        }
        res = PASS;
        return true;
    }
    void report() {
        standard_report(name, res);
    }
    
    private:
    vector<int> serialBFS(
            const Graph & g,
            const device_vector<int> & extractedEdges,
            const device_vector<int> & edgeListStart) {
        int n = g.vertexCount;
        host_vector<int> ee  = extractedEdges,
                         els = edgeListStart;
        vector<int> d(n, INF);
        d[0] = 0;
        queue<int> q;
        q.push(0);
        while(!q.empty()) {
            int akt = q.front();
            q.pop();
            for(int i = els[akt]; i < els[akt + 1]; i++) {
                int succ = ee[i];
                if(d[succ] == INF) {
                    d[succ] = d[akt] + 1;
                    q.push(succ);
                }
            }
        }
        return d;
    }
        
};

void test_preprocessEdges() {
    TestSuite preprocessEdgesTestSuite = TestSuite("preprocessEdges tests");
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("one-vertex tree", generate_random_tree(1)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("two-vertex tree", generate_random_tree(2)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("three-vertex tree", generate_random_tree(3)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("four-vertex tree", generate_random_tree(4)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("ten-vertex tree", generate_random_tree(10)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("100-vertex tree", generate_random_tree(100)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("1000-vertex tree", generate_random_tree(1000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("10000-vertex tree", generate_random_tree(10000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("four-vertex graph", generate_random_connected_graph(4, 6)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("ten-vertex graph", generate_random_connected_graph(10, 25)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("100-vertex graph", generate_random_connected_graph(100, 3000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("1000-vertex graph", generate_random_connected_graph(1000, 3000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("10000-vertex graph", generate_random_connected_graph(10000, 20000)));
    /*preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("100000-vertex tree", generate_random_tree(100000)));*/
    preprocessEdgesTestSuite.play();
}
void test_BFS() {
    TestSuite bfsTestSuite = TestSuite("BFS tests");
    bfsTestSuite.addTest(new TestBFS("one-vertex tree", make_pair(1, 0)));
    bfsTestSuite.addTest(new TestBFS("two-vertex tree", make_pair(2, 1)));
    bfsTestSuite.addTest(new TestBFS("four-vertex tree", make_pair(4, 3)));
    bfsTestSuite.addTest(new TestBFS("ten-vertex tree", make_pair(10, 9)));
    bfsTestSuite.addTest(new TestBFS("ten-vertex graph", make_pair(10, 20)));
    bfsTestSuite.addTest(new TestBFS("100 vertex 200 edge graph", make_pair(100, 200)));
    bfsTestSuite.addTest(new TestBFS("100 vertex 2k edge graph", make_pair(100, 2000)));
    bfsTestSuite.addTest(new TestBFS("1k vertex 20k edge graph", make_pair(1000, 20000)));
    bfsTestSuite.addTest(new TestBFS("10k vertex 200k edge graph", make_pair(10000, 200000)));
    bfsTestSuite.play();
}
int main() {
    test_preprocessEdges();
    test_BFS();
}

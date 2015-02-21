#include "testlib.cu"
#include "../BiconnectedComponents.cu"
#include "../generator.cu"

class TestPreprocessEdges : public Test {
    public:
    string name;
    Graph g;
    string message;
    TestResult res;

    TestPreprocessEdges()
        :g(Graph(0)), res(NOT_CHECKED){}
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


int main() {
    TestSuite preprocessEdgesTestSuite = TestSuite("preprocessEdges tests");
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("one-vertex tree", generate_random_tree(1)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("two-vertex tree", generate_random_tree(2)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("three-vertex tree", generate_random_tree(3)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("four-vertex tree", generate_random_tree(4)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("ten-vertex tree", generate_random_tree(10)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("100-vertex tree", generate_random_tree(100)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("1000-vertex tree", generate_random_tree(1000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("10000-vertex tree", generate_random_tree(10000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("four-vertex graph", generate_random_graph(4, 6)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("ten-vertex graph", generate_random_graph(10, 25)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("100-vertex graph", generate_random_graph(100, 3000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("1000-vertex graph", generate_random_graph(1000, 3000)));
    preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("10000-vertex graph", generate_random_graph(10000, 20000)));
    /*preprocessEdgesTestSuite.addTest(new TestPreprocessEdges("100000-vertex tree", generate_random_tree(100000)));*/
    preprocessEdgesTestSuite.play();
    return 0;
}

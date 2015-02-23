#include "testlib.cu"
#include "../src/BiconnectedComponents.h"
#include "generator.cu"
#include "../src/helper.h"
#include "serialBCC.cu"

class TestBiconnectedComponents : public Test {
    private:
        string name;
        GraphGenerator * generator;
        TestResult res;
        double duration;

        bool compareOutputs(
                const host_vector<int> & parallelComponents,
                const vector<int> & serialComponents) {

            map<int, vector<int> > components;
            for(int i = 0; i < serialComponents.size(); i++)
                components[serialComponents[i]].push_back(i);

            FOREACH(c, components) {
                vector<int> & numbers = c->second;
                for(int i = 1; i < numbers.size(); i++)
                    CHECK(parallelComponents[numbers[i]] == parallelComponents[numbers[0]]);
            }
            return true;
        }

    public:
        TestBiconnectedComponents(string name, GraphGenerator * generator)
            :name(name), generator(generator), res(NOT_CHECKED) {}
        ~TestBiconnectedComponents() {
            delete generator;
        }
        bool play() {
            Graph g = generator -> generate();
            VALIDATE(validate_connected_graph(g));

            device_vector<int> parallelBCCs;

            timestamp_t beg = get_timestamp();

            BiconnectedComponents::computeBiconnectedComponents(g, parallelBCCs);

            timestamp_t end = get_timestamp();
            duration = compare_timestamps(beg, end);

            host_vector<int> host_parallelBCCs = parallelBCCs;

            vector<int> serialBCCs = serialBiconnectedComponents(g);

            CHECK(compareOutputs(host_parallelBCCs, serialBCCs));

            res = PASS;
            return true;
        }
        void report() {
            time_report(name, res, duration);
        }
};

void run_tests() {
    TestSuite t("BiconnectedComponents tests");
    srand(1410);
    for(int i = 0; i < 10; i++)
        t.addTest(new TestBiconnectedComponents(
                    "test graph",
                    new RandomGraphGeneratorMemoryEfficient(7+i, 10+i)));
    /*t.addTest(new TestBiconnectedComponents(*/
    /*"10-vertex 20-edge graph",*/
    /*new RandomGraphGenerator(10, 20)));*/
    /*t.addTest(new TestBiconnectedComponents(*/
    /*"50-vertex 1000-edge graph",*/
    /*new RandomGraphGenerator(50, 1000)));*/
    /*t.addTest(new TestBiconnectedComponents(*/
    /*"200-vertex 2000-edge graph",*/
    /*new RandomGraphGenerator(200, 2000)));*/
    /*t.addTest(new TestBiconnectedComponents(*/
    /*"2000-vertex 20000-edge graph",*/
    /*new RandomGraphGenerator(2000, 20000)));*/
    /*t.addTest(new TestBiconnectedComponents(*/
    /*"20k-vertex 1kk-edge graph",*/
    /*new RandomGraphGenerator(20000, 1000000)));*/
    /*t.addTest(new TestBiconnectedComponents(*/
    /*"test graph",*/
    /*new RandomGraphGeneratorMemoryEfficient(150, 300)));*/
    t.play();
}
int main() {
    run_tests();
}

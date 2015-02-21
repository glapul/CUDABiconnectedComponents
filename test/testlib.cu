/* Miniframework do testowania 

   Testy warto pisać by były małymi obiektami (czyt - nie mają zawierać grafów, a jedynie generatory grafów)
*/
#pragma once

#include<vector>
#include<iostream>
#include "../helper.h"
using namespace std;


#define private public

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#ifndef EBUG
#define CHECK(x) if(!(x)) {res = FAIL; return false;}
#define VALIDATE(x) if(!(x)) {res = INVALID; return false;}
#else
#include<cassert>
#define CHECK(x) assert(x)
#define VALIDATE(x) assert(x)
#endif


enum TestResult {PASS, FAIL, INVALID, NOT_CHECKED};
class Test {
public:
    virtual bool play()   = 0; //return 1 if test passed
    virtual void report() = 0;
};

class TestSuite {
    private:
    string name;
    vector<Test*> tests;
    public:
    TestSuite(){} //delete?

    TestSuite(string name)
        :name(name) {}

    TestSuite(string name, vector<Test*> tests)
        :name(name), tests(tests){}

    void addTest(Test * ptr) {
        tests.push_back(ptr);
    }
    void play() {
        cerr << "----------------------------------------------\n";
        cerr << "Playing test suite '" << name <<"'\n";
        cerr << "----------------------------------------------\n";
        int passed = 0;
        FOREACH(i, tests) {
            passed += (*i) -> play();
            (*i) -> report();
        }
        cerr << KNRM << "----------------------------------------------\n";
        cerr << (passed == tests.size() ? KGRN : KRED)  << passed << " / " << tests.size() <<" tests passed\n";
        cerr <<KNRM <<"----------------------------------------------\n";
        cerr << endl;
    }
    ~TestSuite() {
        FOREACH(i, tests)
            delete *i;
    }
};
void standard_report(string name, TestResult res) {
    string msg = "";
    if(res == PASS) msg += (string)KGRN + (string)"PASSED ";
    if(res == FAIL) msg += (string)KRED + "FAILED ";
    if(res == INVALID) msg += (string)KYEL + "INVALID ";
    if(res == NOT_CHECKED) msg += (string)KWHT + "NOT CHECKED ";
    msg += name;
    cerr << msg << endl;
}

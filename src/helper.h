#pragma once

#include "Graph.h"
#include<iostream>
#include<sstream>
using namespace std;
#define FOREACH(i, c) for(__typeof((c).begin()) i = (c).begin(); i!=(c).end();i++)
#define pointer(x) (thrust::raw_pointer_cast(& ((x) [0])))
#define ASSERT(x) if(!(x)) return false;
//#define ASSERT(x) assert(x); //for debugging
inline ostream & operator<<(ostream & stream, Edge & e) {
    stream << "{ from = " << e.from << " to = " << e.to << " rev = " << e.rev << " }\n";
    return stream;
}
template<typename T>
inline ostream & operator<<(ostream & stream, thrust::device_vector<T> & vec) {
    thrust::host_vector<T> local = vec;
    stream << "SIZE = " <<local.size() << "\n";
    FOREACH(i, local)
        stream << *i << " ";
    stream << "\n";
    return stream;
}
template<typename T>
inline ostream & operator<<(ostream & stream, thrust::host_vector<T> & local) {
    stream << "SIZE = " <<local.size() << "\n";
    FOREACH(i, local)
        stream << *i << " ";
    stream << "\n";
    return stream;
}
inline ostream & operator<<(ostream & stream, Graph & g) {
    stream << "vertexCount = " << g.vertexCount << " edges :" << endl;
    stream << g.edges;
    return stream;
}

struct FindJoin {
    std::vector<int> parent;
    FindJoin(int n)
        : parent(std::vector<int>(n)) {
        for(int i = 0; i < parent.size(); i++)
            parent[i] = i;
    }
    int find(int x) {
        return x == parent[x] ? x : parent[x] = find(parent[x]);
    }
    void join(int x, int y) {
        parent[find(x)] = find(y);
    }
};

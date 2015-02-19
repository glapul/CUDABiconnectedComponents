#pragma once

#include "Graph.h"
#include<iostream>
#include<sstream>
using namespace std;
#define FOREACH(i, c) for(__typeof((c).begin()) i = (c).begin(); i!=(c).end();i++)
#define pointer(x) (thrust::raw_pointer_cast(& ((x) [0])))
ostream & operator<<(ostream & stream, Edge & e) {
    stream << "{ from = " << e.from << " to = " << e.to << " rev = " << e.rev << " }\n";
    return stream;
}
template<typename T>
ostream & operator<<(ostream & stream, thrust::device_vector<T> & vec) {
    thrust::host_vector<T> local = vec;
    stream << "SIZE = " <<local.size() << "\n";
    FOREACH(i, local)
        stream << *i << " ";
    stream << "\n";
    return stream;
}
ostream & operator<<(ostream & stream, Graph & g) {
    stream << "vertexCount = " << g.vertexCount << " edges :" << endl;
    stream << g.edges;
    return stream;
}

#include<bits/stdc++.h>
#include "Graph.h"

using namespace std;
vector<pair<int,int> > generate_random_undirected_edges(int n, int m) {
    assert(m >= n - 1);
    assert(m <= (n * (n - 1)) / 2);

    srand(time(NULL));
    vector<vector<bool> > used(n, vector<bool>(n));
    vector<pair<int,int> > undirected_edges;
    for(int i = 1; i < n; i++) {
        int parent = rand()%i;
        undirected_edges.push_back(make_pair(i, parent));
        used[i][parent] = used[parent][i] = true;
    }
    vector<pair<int,int> > possible_edges;
    for(int i = 0; i < n; i++)
        for(int j = i + 1; j < n; j++)
            if(!used[i][j])
                possible_edges.push_back(make_pair(i, j));
    random_shuffle(possible_edges.begin(), possible_edges.end());
    for(int i = 0; i <  m - n + 1; i++)
        undirected_edges.push_back(possible_edges[i]);
    return undirected_edges;
}

thrust::host_vector<Edge> direct_edges(vector<pair<int,int> > undirected_edges) {
    thrust::host_vector<Edge> result;
    for(int i = 0; i < undirected_edges.size(); i++) {
        auto e = undirected_edges[i];
        result.push_back(Edge(e.first, e.second, 2*i));
        result.push_back(Edge(e.second, e.first, 2*i+1));
    }
    return result;
}

Graph generate_random_graph(int n, int m) {
    return Graph(n, direct_edges(generate_random_undirected_edges(n, m)));
}
Graph generate_random_tree(int n) {
    return Graph(n, direct_edges(generate_random_undirected_edges(n, n - 1)));
}

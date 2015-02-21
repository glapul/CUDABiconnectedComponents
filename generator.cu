#pragma once
#include<bits/stdc++.h>
#include "Graph.h"
#include "helper.h"
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

    vector<pair<int,int> > directed = undirected_edges;
    FOREACH(i, undirected_edges)
        directed.push_back(make_pair(i->second, i-> first));
    sort(directed.begin(), directed.end());

    map<pair<int,int>, int> where;
    for(int i = 0; i < directed.size(); i++)
        where[directed[i]] = i;

    thrust::host_vector<Edge> result;
    FOREACH(i, directed)
        result.push_back(Edge(i->first, i->second, where[make_pair(i->second, i->first)]));
    return result;
}

Graph generate_random_graph(int n, int m) {
    return Graph(n, direct_edges(generate_random_undirected_edges(n, m)));
}
Graph generate_random_tree(int n) {
    return Graph(n, direct_edges(generate_random_undirected_edges(n, n - 1)));
}

bool validate_graph(const Graph & graph) {
    thrust::host_vector<Edge> edges = graph.edges;
    set<int> seen_vertices;
    for(int i = 0; i < edges.size();i++) {
        Edge curr = edges[i];
        ASSERT(curr.from >= 0 && curr.from < graph.vertexCount);
        ASSERT(curr.to >= 0 && curr.to < graph.vertexCount);
        ASSERT(curr.from != curr.to);
        ASSERT(curr.rev >= 0  && curr.rev < edges.size());
        Edge rev = edges[curr.rev];
        ASSERT(rev.from == curr.to && curr.from == rev.to);
        seen_vertices.insert(edges[i].from);
    }
    ASSERT(graph.vertexCount == 1 || seen_vertices.size() == graph.vertexCount);
    return true;
}

bool validate_connected_graph(const Graph & graph) {
    ASSERT(validate_graph(graph));
    FindJoin fu = FindJoin(graph.vertexCount);
    int sccs = graph.vertexCount;
    thrust::host_vector<Edge> edges = graph.edges;
    for(int i = 0; i < edges.size(); i++) {
        int u = edges[i].from;
        int v = edges[i].to;
        if(fu.find(u) != fu.find(v)) {
            fu.join(u, v);
            sccs--;
        }
    }
    ASSERT(sccs == 1);
    return true;
}

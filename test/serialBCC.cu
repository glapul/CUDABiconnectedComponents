// Implementacja sekwencyjna. Wywolywac

#include<bits/stdc++.h>
using namespace std;
namespace serialBCC {

    // Dwuspójne składowe itd. - mosty traktowane jak dwuspójne!
    // Linie oznaczone [D], [M], [A], [L] potrzebne tylko, jeśli szukamy
    //     numeracji [D]wuspójnych, [M]ostów, p.[A]rtykulacji, funkcji [L]ow
    // Maciek Wawro

    const int MAXN = 5000005;

    struct _Edge{ // W momencie rozpoczęcia algorytmu musi być bcc = -1 i bridge = 0
        _Edge* rev;
        int dest;
        int bcc;        //OUT: Numer komponentu
        bool bridge;    //OUT: Czy most                         /*M*/
        _Edge(int v) : dest(v), bcc(-1) {
            bridge = false;                                     /*M*/
        };
    };

    int N;                  //IN: Liczba wierzchołków
    list<_Edge>  adj[MAXN];  //IN: Listy sąsiedztwa
    int  visit[MAXN];
    bool  artp[MAXN];  //OUT: True, jeśli dany wierzchołek jest p.art. /*A*/
    int      bcc_num;  //OUT: Liczba komponentów                /*D*/
    int    low[MAXN];                                           /*L*/

    stack<_Edge*> _stack;                                        /*D*/
    int        _dfsTime;
    int bccDFS(int v, bool root = false) {
        int lo = visit[v] = ++_dfsTime;
        FOREACH(it, adj[v]) {
            if(it -> bcc != -1) continue;
            _stack.push(&*it);                                  /*D*/
            it->rev->bcc = -2;
            if(!visit[it->dest]) {
                int ulo = bccDFS(it->dest);
                lo = min(ulo, lo);
                it->bridge = it->rev->bridge = (ulo > visit[v]);/*M*/
                if(ulo >= visit[v]) {                           /*AD*/
                    artp[v] = !root; root = false;              /*A*/
                    _Edge* edge;                                 /*D*/
                    do {
                        edge = _stack.top();                    /*D*/
                        _stack.pop();                           /*D*/
                        edge->bcc = edge->rev->bcc = bcc_num;   /*D*/
                    } while(edge != &*it);                      /*D*/
                    ++bcc_num;                                  /*D*/
                }                                               /*AD*/
            } else lo = min(lo, visit[it->dest]);
        }
        low[v] = lo;                                            /*L*/
        return lo;
    }

    void computeBCC(){
        while(!_stack.empty())
            _stack.pop();
        fill(artp, artp+N, false);                              /*A*/
        fill(visit, visit+N, false);
        _dfsTime = 1;
        bcc_num = 0;                                            /*D*/
        for(int i = 0; i < N; i++)
            if(!visit[i]) 
                bccDFS(i, true);
    }
};

vector<int> serialBiconnectedComponents(const Graph & g, host_vector<Edge> & host_edges) {
    serialBCC::N = g.vertexCount;
    vector<serialBCC::_Edge * > pointers(host_edges.size());

    for(int i = 0; i < (int) host_edges.size(); i++) {

        Edge & e = host_edges[i];
        serialBCC::adj[e.from].push_back(serialBCC::_Edge(e.to));
        pointers[i] = &(serialBCC::adj[e.from].back());

        if(e.rev < i) {
            pointers[e.rev] -> rev = pointers[i];
            pointers[i] -> rev = pointers[e.rev];
        }
    }

    serialBCC::computeBCC();
    
    vector<int> res(host_edges.size());
    for(int i = 0; i < (int)host_edges.size(); i++)
        res[i] = pointers[i] -> bcc;

    for(int i = 0; i < serialBCC::N; i++)
        serialBCC::adj[i].clear();

    return res;
}

#define private public
#include "../BiconnectedComponents.cu"
#include "../generator.cu"

#include<vector>
int main() {
    std::vector<int> v;
    v.push_back(1);
    cout<<v.size();
}

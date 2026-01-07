#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    auto start = chrono::high_resolution_clock::now();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 1000);
    
    vector<int> list(20);
    for (int i = 0; i < 20; i++) list[i] = dis(gen);
    
    sort(list.begin(), list.end());
    
    cout << "Sorted: ";
    for (int val : list) cout << val << " ";
    cout << endl;
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    
    cout << duration.count() << endl;
    
    return 0;
}

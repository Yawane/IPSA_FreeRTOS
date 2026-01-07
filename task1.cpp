#include <iostream>
#include <chrono>

using namespace std;

int main() {
    auto start = chrono::high_resolution_clock::now();
    
    cout << "Working - System Normal" << endl;
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    
    cout << duration.count() << endl;
    
    return 0;
}
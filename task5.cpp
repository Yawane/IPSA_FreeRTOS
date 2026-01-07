#include <iostream>
#include <chrono>
#include <random>

using namespace std;

int main() {
    auto start = chrono::high_resolution_clock::now();
    
    static int reset_param = 0;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);
    
    int input = dis(gen);
    
    if (input == 1) {
        reset_param = 1;
        cout << "RESET received: " << reset_param << endl;
    } else {
        cout << "Waiting for RESET: " << reset_param << endl;
    }
    
    // Reset parameter for next execution
    reset_param = 0;
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    
    cout << duration.count() << endl;
    
    return 0;
}

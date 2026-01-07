#include <iostream>
#include <chrono>
#include <random>

using namespace std;

int main() {
    auto start = chrono::high_resolution_clock::now();
    
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<long long> dis(1000000000LL, 9999999999LL);
    
    long long num1 = dis(gen);
    long long num2 = dis(gen);
    long long result = num1 * num2;
    
    // show multiplication in console
    cout << num1 << " * " << num2 << " = " << result << endl;
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    
    // show execution time in console
    cout << duration.count() << endl;
    
    return 0;
}

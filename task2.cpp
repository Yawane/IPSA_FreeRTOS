#include <iostream>
#include <chrono>
#include <random>

using namespace std;

int main() {
    auto start = chrono::high_resolution_clock::now();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-50.0, 150.0);
    
    double fahrenheit = dis(gen);
    double celsius = (fahrenheit - 32.0) * 5.0 / 9.0; // conversion
    
    // show the initial and converted value in console
    cout << "F: " << fahrenheit << " -> C: " << celsius << endl;
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    
    // show execution time in console
    cout << duration.count() << endl;
    
    return 0;
}

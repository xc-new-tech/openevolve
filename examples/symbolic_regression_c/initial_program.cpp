#include <iostream>
#include <cmath>

// Target function: f(x) = x^2 + x
// Initial approximation: f(x) = x
double evaluate_function(double x) {
    return x;  // Start with a simple linear function
}

int main() {
    // Test the function with some sample inputs
    double test_values[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < num_tests; i++) {
        double x = test_values[i];
        double result = evaluate_function(x);
        std::cout << "f(" << x << ") = " << result << std::endl;
    }
    
    return 0;
} 
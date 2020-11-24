#include <iostream>
#include "linear.h"
#include "matrix.h"
#include "models.h"

using namespace std;

int main() {

    // So that it doesn't affect the input of the matrix
    Matrix max_actions = Matrix(1, 1, [](int row, int column){return 1;});

    Critic a = Critic(max_actions);

    Matrix x;
    Matrix actions;
    Matrix out;
    Matrix expected;

    // Learns to add three numbers and multiply the result by two. pretty neat.
    for (int i = 1; i < 10000; ++i) {
        int val1 = rand() % 50;
        int val2 = rand() % 50;
        int val3 = rand() % 50;
        x = Matrix(1, 2, [&](int i) { return i == 0 ? val1 : val2; });
        actions = Matrix(1, 1, [&]() { return val3; });
        out = a.forward(x, actions);
        Matrix expected = Matrix(1, 1, [&](){ return (val1 + val2 + val3) * 2; });
        if (i % 100 == 0) {
            cout << "2 * (" << val1 << " + " << val2 << " + " << val3 << ") = " << (val1 + val2 + val3) * 2 << " =? " << out;
        }
        a.backprop(expected, out);
    }
}
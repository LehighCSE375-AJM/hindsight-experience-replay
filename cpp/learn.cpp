#include <iostream>
#include <chrono>
#include "tensor.h"
#include "models.h"
#include "adam.h"


using namespace std;

int main() {

    // So that it doesn't affect the input of the tensor
    Tensor max_actions(1, 1, [](){ return 1;});

    Critic a = Critic(max_actions);

    Tensor x;
    Tensor actions;
    Tensor out;
    Tensor expected;

    auto start = chrono::high_resolution_clock::now();

    // Learns to add three numbers and multiply the result by two. pretty neat.
    for (int i = 1; i < 10000; ++i) {
        int val1 = rand() % 50;
        int val2 = rand() % 50;
        int val3 = rand() % 50;
        x = Tensor(1, 2, [&](int i) { return i == 0 ? val1 : val2; });
        actions = Tensor(1, 1, [&]() { return val3; });
        out = a.forward(x, actions);
        expected = Tensor(1, 1, [&](){ return (val1 + val2 + val3) * 2; });
        if (i % 100 == 0) {
            cout << "2 * (" << val1 << " + " << val2 << " + " << val3 << ") = " << (val1 + val2 + val3) * 2 << " =? " << out;
        }
        a.backprop(expected, out);
    }

    auto end = chrono::high_resolution_clock::now();
    // 3.56722e+07 with no optimizations!
    // 7.40075e+06 with optimizations made. 
    chrono::duration<float, micro> duration = end - start;
    cout << "Duration: " << duration.count() << endl;
}

// g++ -shared -Wl,-soname,learn_wrapper -o learn_wrapper.so -fPIC learn_wrapper.cpp -I /usr/include/x86_64-linux-gnu -lopenblas

#include <iostream>
#include <chrono>
#include "linear.h"
#include "matrix.h"
#include "models.h"

using namespace std;

class wrapper_class
{
public:
	Matrix max_actions;
	Critic* a;

	Matrix x;
	Matrix actions;
	Matrix out;
	Matrix expected;

	int val1;
	int val2;
	int val3;

	wrapper_class()
	{
		max_actions = Matrix(1, 1, [](){ return 1;});
		a = new Critic(max_actions);
	}

	~wrapper_class() = default;
};

extern "C"
{
	wrapper_class* setup()
	{
		wrapper_class* w = new wrapper_class();
		return w;
	}

	int run(wrapper_class* w)
	{
		auto start = chrono::high_resolution_clock::now();

		// Learns to add three numbers and multiply the result by two. pretty neat.
		for (int i = 1; i < 10000; ++i) {
			// cout << "Iteration " << i << endl;
			w->val1 = rand() % 50;
			w->val2 = rand() % 50;
			w->val3 = rand() % 50;
			w->x = Matrix(1, 2, [&](int i) { return i == 0 ? w->val1 : w->val2; });
			w->actions = Matrix(1, 1, [&]() { return w->val3; });
			w->out = w->a->forward(w->x, w->actions);
			w->expected = Matrix(1, 1, [&](){ return (w->val1 + w->val2 + w->val3) * 2; });
			if (i % 100 == 0) {
				cout << "2 * (" << w->val1 << " + " << w->val2 << " + " << w->val3 << ") = " << (w->val1 + w->val2 + w->val3) * 2 << " =? " << w->out;
			}
			w->a->backprop(w->expected, w->out);
		}

		auto end = chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>
										(end - start).count();
	}

	int run_all()
	{
		// So that it doesn't affect the input of the matrix
		Matrix max_actions(1, 1, [](){ return 1;});

		Critic a = Critic(max_actions);

		Matrix x;
		Matrix actions;
		Matrix out;
		Matrix expected;

		auto start = chrono::high_resolution_clock::now();

		// Learns to add three numbers and multiply the result by two. pretty neat.
		for (int i = 1; i < 10000; ++i) {
			int val1 = rand() % 50;
			int val2 = rand() % 50;
			int val3 = rand() % 50;
			x = Matrix(1, 2, [&](int i) { return i == 0 ? val1 : val2; });
			actions = Matrix(1, 1, [&]() { return val3; });
			out = a.forward(x, actions);
			expected = Matrix(1, 1, [&](){ return (val1 + val2 + val3) * 2; });
			if (i % 100 == 0) {
				cout << "2 * (" << val1 << " + " << val2 << " + " << val3 << ") = " << (val1 + val2 + val3) * 2 << " =? " << out;
			}
			a.backprop(expected, out);
		}

		auto end = chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>
										(end - start).count();
	}
}
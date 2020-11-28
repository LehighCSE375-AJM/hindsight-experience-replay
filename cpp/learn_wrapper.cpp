// g++ -shared -Wl,-soname,wrapper -o wrapper.so -fPIC -I /usr/include/x86_64-linux-gnu -lopenblas learn_wrapper.cpp

#include <iostream>
#include "matrix.h"

using namespace std;

class matrix_storage
{
public:
	Matrix m;

	matrix_storage(int height, int width, double* inputs)
	{
		m = Matrix(height, width, inputs);
		// cout << m << endl;
	}

	~matrix_storage() = default;
};

extern "C"
{
	void* new_matrix(double* inputs)
	{
		// Matrix* m = new Matrix(2, 2, inputs);
		matrix_storage* mat = new matrix_storage(1, 1, inputs);
		return (void *)mat;
	}

	void multiply(void* a, double b)
	{
		((Matrix *)a)->mul_(b);
	}

	void print_matrix(void* mat)
	{
		cout << ((matrix_storage *)mat)->m << endl;
	}

	void test_matrix(double* inputs)
	{
		cout << "Starting..." << endl;
		for (int i = 0; i < 4; i++)
		{
			cout << inputs[i] << endl;
		}
		// Matrix m = Matrix(2, 2, inputs);
	}
}
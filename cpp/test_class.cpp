// g++ -shared -Wl,-soname,test_class -o test_class.so -fPIC test_class.cpp -I /usr/include/x86_64-linux-gnu -lopenblas

#include <iostream>
#include "matrix.h"

class test_class
{
public:
	int storage;
	double* array;
	int size;
	Matrix m;

	test_class(int s, double* a, int size)
	{
		storage = s;
		array = new double[size]();
		for (int i = 0; i < size; i++)
		{
			array[i] = a[i];
		}
		this->size = size;
		m = Matrix(2, 2, a);

		std::cout << m << std::endl;
	}

	void print()
	{
		for (int i = 0; i < size; i++)
		{
			std::cout << array[i] << std::endl;
		}
	}


	~test_class() = default;
};

extern "C"
{
	void* test_set(int s, double* a, int size)
	{
		test_class* t = new test_class(s, a, size);
		return (void *)t;
	}

	int test_get(void* t)
	{
		return (((test_class *)t)->storage);
	}

	void test_print(void* t)
	{
		((test_class *)t)->print();
	}
}
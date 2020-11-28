// g++ -shared -Wl,-soname,wrapper-o wrapper.so -fPIC functions.cpp

#include <iostream>

int main()
{
  std::cout << "main\n";
}

extern "C"
{
	int notmain()
	{
	  std::cout << "notmain\n";
	}

	float cmult(float x, float y)
	{
		return x * y;
	}

	float add_all(float* a, size_t size)
	{
		float total = 0;
		for (int i = 0; i < size; i++)
		{
			total += a[i];
		}
		return total;
	}

	void mult_scalar(double* a, size_t size, double b)
	{
		for (int i = 0; i < size; i++)
		{
			a[i] *= b;
		}
	}

	void ref(double* a)
	{
		a[0] += 1;
	}
}
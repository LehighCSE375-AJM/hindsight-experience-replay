#include <iostream>
#include "tensor.h"

using namespace std;

__global__ void matrixAddOne(double* m) {
	int i = threadIdx.x;
	m[i] = m[i] + 1;
}

int main() {
	Tensor t(2, 2, [](int i) {return i;});
	cout << t << endl;
	matrixAddOne<<1, 4>>(t.values);
	cout << t << endl;
}
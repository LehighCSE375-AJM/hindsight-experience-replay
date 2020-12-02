#include <iostream>
#include "tensor.h"

using namespace std;

// This is pretty neat! (brings me back to my 109 days)
// From here: https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matrixAddOne(double* m) {
	// Very helpful that we can print from a kernel
	printf("hi\n");
	int i = threadIdx.x;
	m[i] = m[i] + 1;
}

int main() {
	Tensor t(2, 2, [](int i) {return i;});
	cout << t << endl;
	// We should probably standardize that device variables will start with d_
	double *d_values;
	gpuErrchk(cudaMalloc((void **) &d_values, sizeof(double) * 4));
	gpuErrchk(cudaMemcpy(d_values, t.values, sizeof(double) * 4, cudaMemcpyHostToDevice));
	matrixAddOne<<<1, 4>>>(d_values);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(t.values, d_values, sizeof(double) * 4, cudaMemcpyDeviceToHost));
	cout << t << endl;
}

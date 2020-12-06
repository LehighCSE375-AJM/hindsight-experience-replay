#include <iostream>
#include "tensor.h"
#include "cuda_utils.h"

using namespace std;

__global__ void matrixAddOne(Tensor m, Tensor out) {
	// m.tanh(out); and m.copy(out); also work
	m.relu(out);
	// Something that we have to think about is doing a __syncthreads() before we do not 
	// elementwise operations such as matrix multiplication (not needed for element-wise 
	// operations since they always act on the same value as the last element-wise operation
	// so they will be consistent). There is a chance that one warp is still finishing the 
	// previous relu, while another thread is doing matrix multiplication with that value. 
	// (something interesting is that newer gpu's allow different threads in the same warp to
	// run operations simultaneously so in theory this could be an issue within a single war
	// too)

	// Another thing to consider if multiple element operations are done in a row then it 
	// could make sense to do it like this: bring subset of matrix into shared memory, 
	// perform operations, write back to global memory, then operate on next subset of 
	// matrix. Would be instead of: read all of matrix from global memory, operate on, then 
	// write back to global memory then do this for the next operation. Ends up as 1 
	// read/write from global memory for all of the operations instead of 1 read/write per 
	// operation from global memory. This may be significantly faster since reading from 
	// shared memory is ~10x faster than global even with coalescense. 

	// Very helpful that we can print from a kernel
	out.print();
}

int main() {
	Tensor t(10, 10, [](int i) {return i - 20;});
	cout << t << endl;
	t.cudafy();
	Tensor out(10, 5);
	out.cudafy();
	matrixAddOne<<<1, 4>>>(t, out);
	gpuErrchk(cudaDeviceSynchronize());
	// This uncudafy doesn't work since the cuda memory address was updated by the kernel
	// (pass-by-value so not updated on the host. Can't be pass-by-reference since then the 
	// kernel would be reading host memory) It doesn't really matter that it doesn't work btw
	// since most (all?) of the tensors will stay within the kernel. (for that same reason 
	// the cudafy method probably isn't needed either)
	out.uncudafy();
}

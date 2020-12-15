#include <iostream>
#include "models.h"
#include "tensor.h"
#include "cuda_utils.h"
#include <curand_kernel.h>
#include "linear.h"

using namespace std;

__global__ void matrixAddOne(unsigned long seed) {
	curandState rand_state;
        curand_init(seed, threadIdx.x, 0, &rand_state);
	Tensor max_action = Tensor(1, 1);
	max_action.values[0] = 1;

	Critic c(1, 1, 1, max_action, rand_state);
	Tensor in = Tensor(1, 2);
	Tensor actions = Tensor(1, 1);
	Tensor expected = Tensor(1, 1);
	Tensor out;
	for (int i = 1; i < 100; ++i) {
		if (threadIdx.x == 0) {
			int val1 = curand(&rand_state) % 50;
			int val2 = curand(&rand_state) % 50;
			int val3 = curand(&rand_state) % 50;
			in.values[0] = val1;
			in.values[1] = val2;
			actions.values[0] = val3;
			expected.values[0] = 2 * (val1 + val2 + val3);
		}
		__syncthreads();
		out = c.forward(in, actions);
		if (threadIdx.x == 0) {
			if (i % 10 == 0) {
				printf("2 * (%g + %g + %g) = %g =? %g\n", in.values[0], in.values[1], actions.values[0], expected.values[0], out.values[0]);
			}
		}
		c.backprop(expected, out);
	}
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
	// This above idea might not be super helpful when we have to store the intermediate matrixes
	// but I would guess it will be good for the adam optimizer. (not really sure though, could 
	// just make sense to do everything in one call, idk)

	// Very helpful that we can print from a kernel
}

int main() {
	// 256 megabyte heap size (probably overkill). Otherwise we run out and the new operater just return null
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
	matrixAddOne<<<1, THREADS>>>(1234);
	gpuErrchk(cudaDeviceSynchronize());
	// This uncudafy doesn't work since the cuda memory address was updated by the kernel
	// (pass-by-value so not updated on the host. Can't be pass-by-reference since then the 
	// kernel would be reading host memory) It doesn't really matter that it doesn't work btw
	// since most (all?) of the tensors will stay within the kernel. (for that same reason 
	// the cudafy method probably isn't needed either)
}

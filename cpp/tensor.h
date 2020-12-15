#pragma once

#include <iostream>
#include <functional>
#include <cmath>
#include <cassert>
#include <utility>
#include "cblas.h"
#include <curand_kernel.h>
#include "cuda_utils.h"
#include "math.h"

#define THREADS 32

using namespace std;

class Tensor {
private:
	void initialize(int height, int width, function<double(int, int)> initializer) {
		this->height = height;
		this->width = width;
		cout << "Updated output tensor" << endl;
		this->values = new double[height * width];
		for (int i = 0; i < height * width; ++i) {
			int row = i / width;
			int column = i % width;
			values[i] = initializer(row, column);
		}
	};

	// This initializes the d_values ptr properly based on the currently set height. 
	__device__ void initialize_d_values() {
		// This is some fancy code to make sure all the threads point to the same matrix values
		__shared__ double *new_d_ptr;
		if (threadIdx.x == 0) {
			printf("Updated output tensor\n");
			new_d_ptr = new double[height * width];
			// May want to zero values? idk
		}
		__syncthreads();
		this->values = new_d_ptr;
		// __syncthreads();
	}
	
public:
	int height = 0;
	int width = 0;
	double* values = NULL;
	Tensor* gradient = NULL;

	Tensor() = default;

	// Creates a new device tensor in accordance with pytorch random initialization for weight/bias matrices
	__device__ Tensor(int height, int width, curandState &rand_state, int in_features) {
		this->height = height;
		this->width = width;
		initialize_d_values();
		int i = threadIdx.x;
                // Warp divergence (although I don't think this is all that bad)
                while (i < height * width) {
			// curand_uniform generates a uniformly distributed random number betweeen 0 and 1
			this->values[i] = (curand_uniform(&rand_state) - 0.5) * 2 * sqrt(1./in_features);
                        i += blockDim.x;
                }

	}

	__host__ __device__ Tensor(int height, int width) {
#ifdef __CUDA_ARCH__
		this->height = height;
		this->width = width;
		initialize_d_values();
#else
		initialize(height, width, [&](int _1, int _2){return 0;});
#endif
	}

	Tensor(int height, int width, function<double()> initializer) {
		initialize(height, width, [&](int _1, int _2){return initializer();});
	}

	Tensor(int height, int width, function<double(int)> initializer) {
		initialize(height, width, [&](int row, int column) {
			int index = row * width + column;
			return initializer(index);
		});
	}

	Tensor(int height, int width, function<double(int, int)> initializer) {
		initialize(height, width, std::move(initializer));
	}

	__host__ __device__ Tensor& grad() {
		if (this->gradient == NULL) {
#ifdef __CUDA_ARCH__
			//__syncthreads();
			//__shared__ Tensor *new_t_ptr;
			//if (threadIdx.x == 0) {
			//	new_t_ptr = NULL;
			//	new_t_ptr = new Tensor(height, width);
			//	// May want to zero values? idk
			// }
			// __syncthreads();
			this->gradient = new Tensor(height, width);
			// I'm not 100% sure this is required, it depends on how shared memory functions
			// If the new_d_ptr is overwritten when grad() is called again then this is required
			// (not sure if shared memory is more akin to local variables though)
			//__syncthreads();
#else
			this->gradient = new Tensor(height, width, [&]() {
				return 0.;
			});
#endif
		}
		return *this->gradient;
	}

	__host__ __device__ Tensor& operator=(const Tensor &m) {
		if (this==&m) return *this; 
		m.copy(*this);
		return (*this);
	}

	__host__ __device__ ~Tensor() {
#ifdef __CUDA_ARCH__
		if (threadIdx.x == 0) {
			// Might be good at some point to do this (although it might not matter since a tensor should only be destructed when the kernel exits)
			// printf("deleting %p\n", this->values);
			// delete[] this->values;
		}
#else
		delete[] values;
		delete gradient;
#endif
	}

	// Sets up the tensor to output the from a tensor operation onto. 
	// This function is advantageous since it will only cause the tensor to allocate a new double array if the tensor is not the proper size. 
	__host__ __device__ static void setup_output_tensor(int height, int width, Tensor &m) {
#ifdef __CUDA_ARCH__
		// This is a static function so no possible colision with this->height, this->width
		if (height * width != m.height * m.width) {
			m.height = height;
			m.width = width;
			m.initialize_d_values();
		}
#else
		if (height * width != m.height * m.width) {
			// This should only be printed during the first pass of training. 
			m.height = height;
			m.width = width;
			cout << "Updating output tensor" << endl;
			m.values = new double[height * width];
			// cblas way of zeroeing a tensor.
			m.mul_(0);
		}
#endif
	}

	// Calculates the relu function and writies the output to the passed in tensor
	// Means can reuse already alocated matricies. 
	__host__ __device__ void relu(Tensor &m) const {
		setup_output_tensor(height, width, m);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		// Warp divergence (although I don't think this is all that bad)
		while (i < height * width) {
			double in_val = values[i];
			// Not sure how this type of if statement works with cuda warp divergence. 
			m.values[i] = in_val < 0 ? 0 : in_val;
			i += blockDim.x;
		}
#else
		for (int i = 0; i < height * width; ++i) {
			m.values[i] = max(0., values[i]);
		}
#endif
	}

	// Computes the gradient of the relu function for each element in place
	__host__ __device__ Tensor& relu_gradient() {
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			values[i] = values[i] > 0 ? 1. : 0.;
			i += blockDim.x;
		}
#else
		for (int i = 0; i < height * width; ++i) {
			values[i] = values[i] > 0 ? 1. : 0.;
		}
#endif
		return *this;
	}

	// Calculates the tanh function and writies the output to the passed in tensor
	// This has the advantage that we don't need to allocate a whole bunch of new space each time we run this function. 
	__host__ __device__ void tanh(Tensor &m) const {
		setup_output_tensor(height, width, m);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		// Possible warp divergence
		while (i < height * width) {
			double in_val = values[i];
			// There may be an inaccurate fast version of tanh which we may want to look into
			m.values[i] = std::tanh(in_val);
			i += blockDim.x;
		}
#else
		for (int i = 0; i < height * width; ++i) {
			m.values[i] = std::tanh(values[i]);
		}
#endif
	}

	// Computes the gradient of the tanh function for each element in place. 
	__host__ __device__ Tensor& tanh_gradient() {
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			values[i] = 1 - pow(std::tanh(values[i]), 2);
			i += blockDim.x;
		}
#else
		for (int i = 0; i < height * width; ++i) {
			// This is another way to write sech^2. Not sure why cmath doesn't have a sech function. 
			values[i] = 1 - pow(std::tanh(values[i]), 2);
		}
#endif
		return *this;
	}

	// Copies the values from this tensor onto the passed in tensor. 
	__host__ __device__ void copy(Tensor &m) const {
		setup_output_tensor(height, width, m);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		// Warp divergence?
		while (i < height * width) {
			m.values[i] = this->values[i];
			i += blockDim.x;
		}
#else
		cblas_dcopy(height * width, values, 1, m.values, 1);
#endif
	}
	
	Tensor zeros() const {
		return Tensor(height, width, [&]() {
			return 0.;
		});
	}

	// This is the gradient when no activation function is used. 
	__host__ __device__ Tensor& ones() {
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			values[i] = 1;
			i += blockDim.x;
		}
#else
		for (int i = 0; i < height * width; ++i) {
			values[i] = 1;
		}
#endif
		return *this;
	}

	// Transposes this tensor onto the passed in tensor. 
	__host__ __device__ void transpose(Tensor &m) const {
		setup_output_tensor(width, height, m);
#ifdef __CUDA_ARCH__
		__syncthreads();
		int i = threadIdx.x;
		while (i < height * width) {
			int index2 = (i % width) * height + i / width;
			m.values[index2] = this->values[i];
			i += blockDim.x;
		}
		// There are not consistent elementwise accesses of m, so we have to syncthreads
		__syncthreads();
#else
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				int index1 = i * width + j;
				int index2 = j * height + i;
				m.values[index2] = values[index1];
			}
		}
#endif
	}

	__host__ __device__ Tensor& add_(double d) {
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] += d;
			i += blockDim.x;
		}
#else
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] += d;
		}
#endif
		return *this;
	}

	Tensor& add_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		cblas_daxpy(height * width, 1, m.values, 1, values, 1);
		return *this;
	}

	__host__ __device__ Tensor& sub_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] -= m.values[i];
			i += blockDim.x;
		}
#else
		cblas_daxpy(height * width, -1, m.values, 1, values, 1);
#endif
		return *this;
	}

	__host__ __device__ Tensor& mul_(const double& d) {
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		// Warp divergence?
		while (i < height * width) {
			this->values[i] *= d;
			i += blockDim.x;
		}
#else
		cblas_dscal(height * width, d, values, 1);
#endif
		return *this;
	}

	__host__ __device__ Tensor& mul_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		// Warp divergence?
		while (i < height * width) {
			this->values[i] *= m.values[i];
			i += blockDim.x;
		}
#else
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] *= m.values[i];
		}
#endif
		return *this;
	}

	__host__ __device__ Tensor& div_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] /= m.values[i];
			i += blockDim.x;
		}
#else
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] /= m.values[i];
		}
#endif
		return *this;
	}

	__host__ __device__ Tensor& div_(const double& d) {
		assert(d != 0.);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] /= d;
			i += blockDim.x;
		}
#else
		cblas_dscal(height * width, 1/d, values, 1);
#endif
		return *this;
	}

	__host__ __device__ Tensor& sqrt_() {
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] *= std::sqrt(this->values[i]);
			i += blockDim.x;
		}
#else
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] = std::sqrt(this->values[i]);
		}
#endif
		return *this;
	}

	// this = this + m * p
	__host__ __device__ Tensor& addmul_(const Tensor& m, const double& p) {
		assert(this->height == m.height);
		assert(this->width == m.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] += m.values[i] * p;
			i += blockDim.x;
		}
#else
		cblas_daxpby(height * width, p, m.values, 1, 1, values, 1);
#endif
		return *this;
	}

	// this = this - m * p
	__host__ __device__ Tensor& submul_(const Tensor& m, const double& p) {
		assert(this->height == m.height);
		assert(this->width == m.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			this->values[i] -= m.values[i] * p;
			i += blockDim.x;
		}
#else
		cblas_daxpby(height * width, -p, m.values, 1, 1, values, 1);
#endif
		return *this;
	}

	/**
	 * This may need to be put in layer.h
	 */
	__host__ __device__ Tensor& addcdiv_(const Tensor& m1, const Tensor& m2, const double& val) {
		assert(this->height == m1.height);
		assert(m1.height == m2.height);
		assert(this->width == m1.width);
		assert(m1.width == m2.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < height * width) {
			if (m2.values[i] != 0) {
				this->values[i] += val * m1.values[i] / m2.values[i];
			}
			i += blockDim.x;
		}
#else
		for (int i = 0; i < this->height * this->width; i++) {
			if (m2.values[i] != 0) {
				this->values[i] += val * m1.values[i] / m2.values[i];
			}
		}
#endif
		return *this;
	}

	/**
	 * This may need to be put in layer.h
	 */
	Tensor& addcmul_(const Tensor& m1, const Tensor& m2, const double& val) {
		assert(this->height == m1.height);
		assert(m1.height == m2.height);
		assert(this->width == m1.width);
		assert(m1.width == m2.width);
		for (int i = 0; i < this->height * this->width; i++) {
				this->values[i] += val * m1.values[i] * m2.values[i];
		}
		return *this;
	}

	// this = this + m^2 * val
	__host__ __device__ Tensor& addsquaremul_(const Tensor& m, const double& val) {
		assert(this->height == m.height);
		assert(this->width == m.width);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
                while (i < height * width) {
                        this->values[i] += std::pow(m.values[i], 2) * val;
                        i += blockDim.x;
                }
#else
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] += val * m.values[i] * m.values[i];
		}
#endif
		return *this;
	}

	// Computes m1*m2 + bias (m1 and m2 are transposed if their respective transpose bools are true)
	// Bias is overwritten with the result (just how blas works)
	__host__ __device__ static void matrix_multiply(const Tensor &m1, bool m1_transpose, const Tensor &m2, bool m2_transpose, Tensor &bias) {
		int m = m1_transpose ? m1.width : m1.height;
		int m1_k = m1_transpose ? m1.height : m1.width;
		int m2_k = m2_transpose ? m2.width : m2.height;
		int n = m2_transpose ? m2.height : m2.width;
		assert(m1_k == m2_k);
                setup_output_tensor(m, n, bias);
#ifdef __CUDA_ARCH__
		// These two multiples cause VERY bad memory access coalescense. (90% chance I spelt that wrong)
		// Its fine when the multiples are one, but otherwise it will significantly slow down performance!
		int m1_offset_multiple = m1_transpose ? m1.width : 1;
		int m2_offset_multiple = m2_transpose ? 1 : m2.width;
		// Possible race condition would exist without this __syncthreads()
		__syncthreads();
		// Currently I am assuming the m2_transpose is true and m1_transpose is false (obviously a poor assumption!)
		for (int i = 0; i < m; ++i) {
			for (int k = 0; k < n; ++k) {
				int m1_offset = i * (m1_transpose ? 1 : m1_k);
				int m2_offset = k * (m2_transpose ? m1_k : 1);
				__shared__ double prod_results[THREADS];
				int t = threadIdx.x;
				prod_results[t] = 0;
				// For most of our matrix multiplication half the threads will never enter this while loop, and end up doing no work which is not ideal...
				while (t < m1_k) {
					prod_results[threadIdx.x] += m1.values[m1_offset + t * m1_offset_multiple] * m2.values[m2_offset + t * m2_offset_multiple];
                        		t += blockDim.x;
				}
				// Reduce the shared array to one value (https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
				for (unsigned int s = 1; s < blockDim.x; s *= 2) {
					if (threadIdx.x % (2 * s) == 0) {
						prod_results[threadIdx.x] += prod_results[threadIdx.x + s];
					}
					__syncthreads();
				}
				if (threadIdx.x == 0) {
					int offset = i * n;
					bias.values[offset + k] += prod_results[0];
				}
			}
		}
		__syncthreads();
#else
		cblas_dgemm(CblasRowMajor, m1_transpose ? CblasTrans : CblasNoTrans, m2_transpose ? CblasTrans : CblasNoTrans,
								m, n, m1_k,
								1.0, m1.values, m1_transpose ? m : m1_k,
								m2.values, m2_transpose ? m1_k : n,
								1.0, bias.values, n);
#endif
	}

	// Concatonate to Tensor classes. They both MUST be vectors (Tensor with height 1)
	// Could make sense to extend this is the future so 2d tensor's are supported, however, that
	// isn't needed right now so I won't bother supporting it. 
	__host__ __device__ static void vector_concat_onto(const Tensor &m1, const Tensor &m2, Tensor &out) {
		assert(m1.height == 1);
		assert(m2.height == 1);
		setup_output_tensor(1, m1.width + m2.width, out);
#ifdef __CUDA_ARCH__
		int i = threadIdx.x;
		while (i < m1.width + m2.width) {
			out.values[i] = i >= m1.width ? m2.values[i - m1.width] : m1.values[i];
			i += blockDim.x;
		}
#else
		cblas_dcopy(m1.width, m1.values, 1, out.values, 1);
		cblas_dcopy(m2.width, m2.values, 1, &(out.values[m1.width]), 1);
#endif
	}
	
	// Cuda device code doesn't support the string data type interestingly enough
	__device__ void print(const char * header) const {
		if (threadIdx.x == 0) {
			printf("%s:\n", header);
			this->print();
		}
	}

	__device__ void print() const {
		if (threadIdx.x == 0) {
			for (int i = 0; i < height * width; ++i) {
				printf(" %g ", values[i]);
				if (i % width == width - 1) {
					printf("\n");
				}
			}
		}
	}
};

// Overriding << operator. How fun!
ostream& operator<<(ostream& os, const Tensor& m) {
	for (int i = 0; i < m.height * m.width; ++i) {
		os << " " << m.values[i] << " ";
		if (i % m.width == m.width - 1) {
			os << endl;
		}
	}
	return os;
}

// The below operators should be used sparingly as they are not done i place which means a new (possibly large) double array has to be allocated. 
// Calculates the product of each individual element (does NOT multiply the tensor in typical matrix multiplication fashion.)
// Resembles torch tensor * operator. 
Tensor operator*(const Tensor& m1, const Tensor& m2) {
	assert(m1.height == m2.height);
	assert(m1.width == m2.width);
	return Tensor(m1.height, m1.width, [&](int i) {
		return m1.values[i] * m2.values[i];
	});
}

// Multiplies each element by a constant. 
Tensor operator*(const Tensor& m1, const double& m2) {
	return Tensor(m1.height, m1.width, [&](int i) {
		return m1.values[i] * m2;
	});
}

// Calculates the m1 / m2 for each individual element
// Resembles torch tensor / operator. 
Tensor operator/(const Tensor& m1, const Tensor& m2) {
	assert(m1.height == m2.height);
	assert(m1.width == m2.width);
	return Tensor(m1.height, m1.width, [&](int i) {
		return m1.values[i] / m2.values[i];
	});
}

Tensor operator+(const Tensor& m1, const Tensor& m2) {
	assert(m1.height == m2.height);
	assert(m1.width == m2.width);
	return Tensor(m1.height, m1.width, [&](int i) {
		return m1.values[i] + m2.values[i];
	});
}

Tensor operator+(const Tensor& m1, const double& m2) {
	return Tensor(m1.height, m1.width, [&](int i) {
		return m1.values[i] + m2;
	});
}

Tensor operator-(const Tensor& m1, const Tensor& m2) {
	assert(m1.height == m2.height);
	assert(m1.width == m2.width);
	return Tensor(m1.height, m1.width, [&](int i) {
		return m1.values[i] - m2.values[i];
	});
}

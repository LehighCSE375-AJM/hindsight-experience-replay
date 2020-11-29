#pragma once

#include <iostream>
#include <functional>
#include <cmath>
#include <cassert>
#include <utility>
#include "cblas.h"

using namespace std;

class Tensor {
private:
	void initialize(int height, int width, function<double(int, int)> initializer) {
		this->height = height;
		this->width = width;
		this->values = new double[height * width];
		for (int i = 0; i < height * width; ++i) {
			int row = i / width;
			int column = i % width;
			values[i] = initializer(row, column);
		}
	};
	
public:
	int height = 0;
	int width = 0;
	double* values = NULL;
	Tensor* gradient = NULL;

	Tensor() = default;

	Tensor(int height, int width) {
		initialize(height, width, [&](int _1, int _2){return 0;});
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

	Tensor& grad() {
		if (this->gradient == NULL) {
			this->gradient = new Tensor(height, width, [&]() {
				return 0.;
			});
		}

		return *this->gradient;
	}

	Tensor& operator=(const Tensor &m) {
		if (this==&m) return *this; 
		if (this->height * this->width != m.height * m.width) {
			delete[] this->values;
			this->values = new double[m.height * m.width];
		}
		// Theres a whole lot of copying values which seems slow. 
		this->height = m.height;
		this->width = m.width;
		cblas_dcopy(height * width, m.values, 1, values, 1);
		return (*this);
	}

	~Tensor() {
		delete[] values;
		delete gradient;
	}

	// Sets up the tensor to output the from a tensor operation onto. 
	// This function is advantageous since it will only cause the tensor to allocate a new double array if the tensor is not the proper size. 
	static void setup_output_tensor(int height, int width, Tensor &m) {
		if (height * width != m.height * m.width) {
			// This should only be printed during the first pass of training. 
			cout << "Updating output tensor" << endl;
			m.height = height;
			m.width = width;
			m.values = new double[height * width];
			// cblas way of zeroeing a tensor.
			m.mul_(0);
		}
	}

	// Calculates the relu function and writies the output to the passed in tensor
	// Means can reuse already alocated matricies. 
	void relu(Tensor &m) const {
		setup_output_tensor(height, width, m);
		for (int i = 0; i < height * width; ++i) {
			m.values[i] = max(0., values[i]);
		}
	}

	// Computes the gradient of the relu function for each element in place
	Tensor& relu_gradient() {
		for (int i = 0; i < height * width; ++i) {
			values[i] = values[i] > 0 ? 1. : 0.;
		}
		return *this;
	}

	// Calculates the tanh function and writies the output to the passed in tensor
	// This has the advantage that we don't need to allocate a whole bunch of new space each time we run this function. 
	void tanh(Tensor &m) const {
		setup_output_tensor(height, width, m);
		for (int i = 0; i < height * width; ++i) {
			m.values[i] = std::tanh(values[i]);
		}
	}

	// Computes the gradient of the tanh function for each element in place. 
	Tensor& tanh_gradient() {
		for (int i = 0; i < height * width; ++i) {
			// This is another way to write sech^2. Not sure why cmath doesn't have a sech function. 
			values[i] = 1 - pow(std::tanh(values[i]), 2);
		}
		return *this;
	}

	// Copies the values from this tensor onto the passed in tensor. 
	void copy(Tensor &m) const {
		setup_output_tensor(height, width, m);
		cblas_dcopy(height * width, values, 1, m.values, 1);
	}
	
	Tensor zeros() const {
		return Tensor(height, width, [&]() {
			return 0.;
		});
	}

	// This is the gradient when no activation function is used. 
	Tensor& ones() {
		for (int i = 0; i < height * width; ++i) {
			values[i] = 1;
		}
		return *this;
	}

	// Transposes this tensor onto the passed in tensor. 
	void transpose(Tensor &m) const {
		setup_output_tensor(width, height, m);
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				int index1 = i * width + j;
				int index2 = j * height + i;
				m.values[index2] = values[index1];
			}
		}
	}

	Tensor& add_(const double& d) {
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] += d;
		}
		return *this;
	}

	Tensor& add_(const Tensor& m1, const double& d) {
		assert(this->height == m1.height);
		assert(this->width == m1.width);
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] += m1.values[i] + d;
		}
		return *this;
	}

	Tensor& sub_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		cblas_daxpby(height * width, -1, m.values, 1, 1, values, 1);
		return *this;
	}

	Tensor& mul_(const double& d) {
		cblas_dscal(height * width, d, values, 1);
		return *this;
	}

	Tensor& mul_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] *= m.values[i];
		}
		return *this;
	}

	Tensor& div_(const Tensor& m) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] /= m.values[i];
		}
		return *this;
	}

	Tensor& div_(const double& d) {
		assert(d != 0.);
		cblas_dscal(height * width, 1/d, values, 1);
		return *this;
	}

	Tensor& sqrt_() {
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] = std::sqrt(this->values[i]);
		}
		return *this;
	}

	// this = this + m * p
	Tensor& addmul_(const Tensor& m, const double& p) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		cblas_daxpby(height * width, p, m.values, 1, 1, values, 1);
		return *this;
	}

	// this = this - m * p
	Tensor& submul_(const Tensor& m, const double& p) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		cblas_daxpby(height * width, -p, m.values, 1, 1, values, 1);
		return *this;
	}

	/**
	 * This may need to be put in layer.h
	 */
	Tensor& addcdiv_(const Tensor& m1, const Tensor& m2, const double& val) {
		assert(this->height == m1.height);
		assert(m1.height == m2.height);
		assert(this->width == m1.width);
		assert(m1.width == m2.width);
		for (int i = 0; i < this->height * this->width; i++) {
			if (m2.values[i] != 0) {
				this->values[i] += val * m1.values[i] / m2.values[i];
			}
		}
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
	Tensor& addsquaremul_(const Tensor& m, const double& val) {
		assert(this->height == m.height);
		assert(this->width == m.width);
		for (int i = 0; i < this->height * this->width; i++) {
				this->values[i] += val * m.values[i] * m.values[i];
		}
		return *this;
	}

	// Computes m1*m2 + bias (m1 and m2 are transposed if their respective transpose bools are true)
	// Bias is overwritten with the result (just how blas works)
	static void matrix_multiply(const Tensor &m1, bool m1_transpose, const Tensor &m2, bool m2_transpose, Tensor &bias) {
		int m = m1_transpose ? m1.width : m1.height;
		int m1_k = m1_transpose ? m1.height : m1.width;
		int m2_k = m2_transpose ? m2.width : m2.height;
		int n = m2_transpose ? m2.height : m2.width;
		assert(m1_k == m2_k);
		setup_output_tensor(m, n, bias);
		assert(m == bias.height);
		assert(n == bias.width);
		cblas_dgemm(CblasRowMajor, m1_transpose ? CblasTrans : CblasNoTrans, m2_transpose ? CblasTrans : CblasNoTrans,
								m, n, m1_k,
								1.0, m1.values, m1_transpose ? m : m1_k,
								m2.values, m2_transpose ? m1_k : n,
								1.0, bias.values, n);
	}

	// Concatonate to Tensor classes. They both MUST be vectors (Tensor with height 1)
	// Could make sense to extend this is the future so 2d tensor's are supported, however, that
	// isn't needed right now so I won't bother supporting it. 
	static void vector_concat_onto(const Tensor &m1, const Tensor &m2, Tensor &out) {
		assert(m1.height == 1);
		assert(m2.height == 1);
		setup_output_tensor(1, m1.width + m2.width, out);
		cblas_dcopy(m1.width, m1.values, 1, out.values, 1);
		cblas_dcopy(m2.width, m2.values, 1, &(out.values[m1.width]), 1);
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
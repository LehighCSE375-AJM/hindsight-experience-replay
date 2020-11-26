#pragma once

#include <iostream>
#include <functional>
#include <cmath>
#include <assert.h>
#include "cblas.h"

using namespace std;

class Matrix {
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
  int height;
  int width;
  double* values = NULL;

  Matrix() = default;

  Matrix(int height, int width, function<double()> initializer) {
    initialize(height, width, [&](int _1, int _2){return initializer();});
  }

  Matrix(int height, int width, function<double(int)> initializer) {
    initialize(height, width, [&](int row, int column) {
      int index = row * width + column;
      return initializer(index);
    });
  }

  Matrix(int height, int width, function<double(int, int)> initializer) {
    initialize(height, width, initializer);
  }

  Matrix& operator=(const Matrix &m) {
    if (this->height * this->width != m.height * m.width) {
      if (this->values != NULL) {
        delete[] this->values;
      }
      this->values = new double[m.height * m.width];
    }
    // Theres a whole lot of copying values which seems slow. 
    this->height = m.height;
    this->width = m.width;
    cblas_dcopy(height * width, m.values, 1, values, 1);
    return (*this);
  }

  ~Matrix() {
    if (values != NULL) {
      delete[] values;
    }
  }

  // Calculates the relu function and writies the output to the passed in matrix
  // Means can reuse already alocated matricies. 
  void relu(Matrix &m) {
    assert(m.height == height);
    assert(m.width == width);
    for (int i = 0; i < height * width; ++i) {
      m.values[i] = max(0., values[i]);
    }
  }

  // Computes the gradient of the relu function for each element in place
  Matrix& relu_gradient() {
    for (int i = 0; i < height * width; ++i) {
      values[i] = values[i] > 0 ? 1. : 0.;
    }
    return *this;
  }

  // Calculates the tanh function and writies the output to the passed in matrix
  // This has the advantage that we don't need to allocate a whole bunch of new space each time we run this function. 
  void tanh(Matrix &m) {
    assert(m.height == height);
    assert(m.width == width);
    for (int i = 0; i < height * width; ++i) {
      m.values[i] = std::tanh(values[i]);
    }
  }

  // Computes the gradient of the tanh function for each element in place. 
  Matrix& tanh_gradient() {
    for (int i = 0; i < height * width; ++i) {
      // This is another way to write sech^2. Not sure why cmath doesn't have a sech function. 
      values[i] = 1 - pow(std::tanh(values[i]), 2);
    }
    return *this;
  }

  // Copies the values from this matrix onto the passed in matrix. 
  void copy(Matrix &m) {
    assert(m.height == height);
    assert(m.width == width);
    cblas_dcopy(height * width, values, 1, m.values, 1);
  }
	
  Matrix zeros() {
    return Matrix(height, width, [&]() {
      return 0.;
    });
  }

  // This is the gradient when no activation function is used. 
  Matrix& ones() {
    for (int i = 0; i < height * width; ++i) {
      values[i] = 1;
    }
    return *this;
  }

  // Transposes this matrix onto the passed in matrix. 
  void transpose(Matrix &m) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int index1 = i * width + j;
        int index2 = j * height + i;
        m.values[index2] = values[index1];
      }
    }
  }

	Matrix& add_(const double& d) {
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] += d;
		}
		return *this;
	}

	Matrix& add_(const Matrix& m1, const double& d) {
		assert(this->height == m1.height);
		assert(this->width == m1.width);
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] += m1.values[i] + d;
		}
		return *this;
	}

	Matrix& mul_(const double& d) {
    cblas_dscal(height * width, d, values, 1);
    return *this;
	}

  Matrix& mul_(const Matrix& m) {
    assert(this->height == m.height);
		assert(this->width == m.width);
		for (int i = 0; i < this->height * this->width; i++) {
			this->values[i] *= m.values[i];
		}
		return *this;
	}

  // this = this - m * p
  Matrix& submul_(const Matrix& m, const double& p) {
    assert(this->height == m.height);
		assert(this->width == m.width);
    cblas_daxpby(height * width, -p, m.values, 1, 1, values, 1);
    return *this;
  }

	/**
	 * This may need to be put in layer.h
	 */
	Matrix& addcdiv_(const Matrix& m1, const Matrix& m2, const double& val) {
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
	Matrix& addcmul_(const Matrix& m1, const Matrix& m2, const double& val) {
		assert(this->height == m1.height);
		assert(m1.height == m2.height);
		assert(this->width == m1.width);
		assert(m1.width == m2.width);
		for (int i = 0; i < this->height * this->width; i++) {
				this->values[i] += val * m1.values[i] * m2.values[i];
		}
		return *this;
	}

  // Creates a bias matrix of all zeroes and runs matrix_multiply. 
  // This can be inefficient when the output matrix is large as it has to allocate a new array of that size. 
  static Matrix matrix_multiply(Matrix &m1, bool m1_transpose, Matrix &m2, bool m2_transpose) {
    int out_height = m1_transpose ? m1.width : m1.height;
    int out_width = m2_transpose ? m2.height : m2.width;
    Matrix out = Matrix(out_height, out_width, [](){return 0.;});
    matrix_multiply(m1, m1_transpose, m2, m2_transpose, out);
    return out;
  }

  // Computes m1*m2 + bias (m1 and m2 are transposed if their respective transpose bools are true)
  // Bias is overwritten with the result (just how blas works)
  static void matrix_multiply(Matrix &m1, bool m1_transpose, Matrix &m2, bool m2_transpose, Matrix &bias) {
    int m = m1_transpose ? m1.width : m1.height;
    int m1_k = m1_transpose ? m1.height : m1.width;
    int m2_k = m2_transpose ? m2.width : m2.height;
    int n = m2_transpose ? m2.height : m2.width;
    assert(m1_k == m2_k);
    assert(m == bias.height);
    assert(n == bias.width);
    cblas_dgemm(CblasRowMajor, m1_transpose ? CblasTrans : CblasNoTrans, m2_transpose ? CblasTrans : CblasNoTrans,
                m, n, m1_k,
                1.0, m1.values, m1_transpose ? m : m1_k,
                m2.values, m2_transpose ? m1_k : n,
                1.0, bias.values, n);
  }

  // Concatonate to Matrix classes. They both MUST be vectors (Matrix with height 1)
  // Could make sense to extend this is the future so 2d matrix's are supported, however, that
  // isn't needed right now so I won't bother supporting it. 
  static Matrix vector_concat(Matrix &m1, Matrix &m2) {
    assert(m1.height == 1);
    assert(m2.height == 1);
    return Matrix(1, m1.width + m2.width, [&](int row, int column) {
      if (column < m1.width) {
        return m1.values[column];
      }
      return m2.values[column - m1.width];
    });
  }
};

// Overriding << operator. How fun!
ostream& operator<<(ostream& os, const Matrix& m) {
  for (int i = 0; i < m.height * m.width; ++i) {
    os << " " << m.values[i] << " ";
    if (i % m.width == m.width - 1) {
      os << endl;
    }
  }
  return os;
}

// The below operators should be used sparingly as they are not done in place which means a new (possibly large) double array has to be allocated. 
// Calculates the product of each individual element (does NOT multiply the matrix in typical matrix multiplication fashion.)
// Resembles torch tensor * operator. 
Matrix operator*(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] * m2.values[i];
  });
}

// Multiplies each element by a constant. 
Matrix operator*(const Matrix& m1, const double& m2) {
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] * m2;
  });
}

// Calculates the m1 / m2 for each individual element
// Resembles torch tensor / operator. 
Matrix operator/(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] / m2.values[i];
  });
}

Matrix operator+(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] + m2.values[i];
  });
}

Matrix operator+(const Matrix& m1, const double& m2) {
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] + m2;
  });
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] - m2.values[i];
  });
}
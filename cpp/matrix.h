#pragma once

#include <iostream>
#include <functional>
#include <cmath>
#include <assert.h>

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
    delete[] this->values;
    // Theres a whole lot of copying values which seems slow. 
    initialize(m.height, m.width, [&](int row, int column) {
      int index = row * width + column;
      return m.values[index];
    });
    return (*this);
  }

  ~Matrix() {
    delete[] values;
  }

  // Creates a new matrix (required for computing gradients for layer.h)
  Matrix relu() {
    return Matrix(height, width, [&](int i) {
      return max(0., values[i]);
    });
  }

  // Creates a new matrix (possibly can do it in place, but need to write everything first)
  // Computes the gradient of the relu function for each element. 
  Matrix relu_gradient() {
    return Matrix(height, width, [&](int i) {
      // Maybe should be >= although I doubt it matters
      return values[i] > 0 ? 1. : 0.;
    });
  }

  // Creates a new matrix (required for computing gradients for layer.h)
  Matrix tanh() {
    return Matrix(height, width, [&](int i) {
      return std::tanh(values[i]);
    });
  }

  // Creates a new matrix (possibly can do it in place, but need to write everything first)
  // Computes the gradient of the relu function for each element. 
  Matrix tanh_gradient() {
    return Matrix(height, width, [&](int i) {
      // This is another way to write sech^2. Not sure why cmath doesn't have a sech function. 
      return 1 - pow(std::tanh(values[i]), 2);
    });
  }

  Matrix copy() {
    return Matrix(height, width, [&](int i) {
      return values[i];
    });
  }
	
	// This is the gradient when no activation function is used. 
  Matrix zeros() {
    return Matrix(height, width, [&]() {
      return 0.;
    });
  }

  // This is the gradient when no activation function is used. 
  Matrix ones() {
    return Matrix(height, width, [&]() {
      return 1.;
    });
  }

  Matrix transpose() {
    // The column, row args are backwards which makes the transpose work. 
    return Matrix(width, height, [&](int column, int row) {
      int index = row * width + column;
      return values[index];
    });
  }

  // Creates a bias matrix of all zeroes and runs matrix_multiply. 
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

// TODO redo all the below operations using blas (assuming blas supports them)
// Calculates the product of each individual element (does NOT multiply the matrix in typical matrix multiplication fashion.)
// Resembles torch tensor * operator. 
Matrix operator*(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  // Possibly could be faster using blas, however the matricies that use
  // this operator are so small anyways its probably not worth it. 
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] * m2.values[i];
  });
}

// Multiplies each element by a constant. 
Matrix operator*(const Matrix& m1, const double& m2) {
  // Possibly could be faster using blas
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

Matrix operator-(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  return Matrix(m1.height, m1.width, [&](int i) {
    return m1.values[i] - m2.values[i];
  });
}
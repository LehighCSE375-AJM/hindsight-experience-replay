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
  double* values;

  Matrix(int height, int width, function<double()> initializer) {
    initialize(height, width, [&](int _1, int _2){return initializer();});
  }

  Matrix(int height, int width, function<double(int, int)> initializer) {
    initialize(height, width, initializer);
  }

  // Modifies matrix in-place
  void relu() {
    for (int i = 0; i < height * width; ++i) {
      this->values[i] = max(0., values[i]);
    }
  }

  // Modifies matrix in-place
  void tanh() {
    for (int i = 0; i < height * width; ++i) {
      this->values[i] = std::tanh(this->values[i]);
    }
  }

  // Concatonate to Matrix classes. They both MUST be vectors (Matrix with height 1)
  // Could make sense to extend this is the future so 2d matrix's are supported, however, that
  // isn't needed right now so I won't bother supporting it. 
  static Matrix* vector_concat(Matrix *m1, Matrix *m2) {
    assert(m1->height == 1);
    assert(m2->height == 1);
    return new Matrix(1, m1->width + m2->width, [&](int row, int column) {
      if (column < m1->width) {
        return m1->values[column];
      }
      return m2->values[column - m1->width];
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

// Calculates the product of each individual element (does NOT multiply the matrix in typical matrix multiplication fashion.)
// Resembles torch tensor * operator. 
Matrix* operator*(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  // Possibly could be faster using blas, however the matricies that use
  // this operator are so small anyways its probably not worth it. 
  Matrix* out = new Matrix(m1.height, m1.width, [&](int row, int column) {
    int index = row * m1.width + column;
    return m1.values[index] * m2.values[index];
  });
  return out;
}

// Calculates the m1 / m2 for each individual element
// Resembles torch tensor / operator. 
Matrix* operator/(const Matrix& m1, const Matrix& m2) {
  assert(m1.height == m2.height);
  assert(m1.width == m2.width);
  Matrix* out = new Matrix(m1.height, m1.width, [&](int row, int column) {
    int index = row * m1.width + column;
    return m1.values[index] / m2.values[index];
  });
  return out;
}
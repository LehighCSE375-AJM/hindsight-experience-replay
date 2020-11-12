#pragma once

#include <iostream>
#include <functional>

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

  Matrix* relu() {
    Matrix *out = new Matrix(height, width, [&](int row, int column) {
      int index = row * width + column;
      return max(0., values[index]);
    });
    return out;
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
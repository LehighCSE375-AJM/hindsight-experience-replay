#pragma once

#include "matrix.h"
#include <iostream>
#include <random>
#include <assert.h>

using namespace std;

// Based off of pytorch nn.Linear
class Linear {
private:
    int _in_features;
    int _out_features;
public:
    Matrix weights;
    Matrix bias;

    Linear(int in_features, int out_features) {
        random_device rd;
        mt19937 gen(rd());

        // Random-ness per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        uniform_real_distribution<double> dis(-sqrt(1./in_features), sqrt(1./in_features));
        this->_in_features = in_features;
        this->_out_features = out_features;
        
        weights = Matrix(out_features, in_features, [&](){return dis(gen);});
        bias = Matrix(1, out_features, [&](){return dis(gen);});
    }

    void multiply(Matrix &x, Matrix &out) {
        // I'm not really sure that passing the out matrix speeds up anything. (and it does make the code less neat)
        // In theory it should since we don't need to allocate a new double array, but I'm not sure its worth it. 
        assert(x.width == _in_features);
        assert(x.height = out.height);
        assert(_out_features == out.width);
        for (int r = 0; r < out.height; ++r) {
            for (int c = 0; c < out.width; ++c) {
                out.values[r * out.width + c] = bias.values[c];
            }
        }
        // Matrix out = Matrix(x.height, _out_features, [&](int row, int column) {
        //     return bias.values[column];
        // });
        Matrix::matrix_multiply(x, false, weights, true, out);
        // return out;
    }
};
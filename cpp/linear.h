#pragma once

#include "cblas.h"
#include "matrix.h"
#include <iostream>
#include <random>
#include <assert.h>

using namespace std;

// Based off of pytorch nn.Linear
class linear {
private:
    int _in_features;
    int _out_features;
public:
    Matrix weights;
    Matrix bias;

    linear(int in_features, int out_features) {
        random_device rd;
        mt19937 gen(rd());

        // Random-ness per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        uniform_real_distribution<double> dis(-sqrt(1./in_features), sqrt(1./in_features));
        this->_in_features = in_features;
        this->_out_features = out_features;
        
        weights = Matrix(out_features, in_features, [&](){return dis(gen);});
        bias = Matrix(1, out_features, [&](){return dis(gen);});
    }

    Matrix multiply(Matrix &x) {
        assert(x.width == _in_features);
        Matrix out = Matrix(x.height, _out_features, [&](int row, int column){
            return bias.values[column];
        });
        Matrix::matrix_multiply(x, false, weights, true, out);
        return out;
    }
};
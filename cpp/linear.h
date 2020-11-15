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
    Matrix* weights;
    Matrix* bias;

    int _in_features;
    int _out_features;
public:
    linear(int in_features, int out_features) {
        random_device rd;
        mt19937 gen(rd());

        // Random-ness per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        uniform_real_distribution<double> dis(-sqrt(1./in_features), sqrt(1./in_features));
        this->_in_features = in_features;
        this->_out_features = out_features;
        
        weights = new Matrix(out_features, in_features, [&](){return dis(gen);});
        bias = new Matrix(1, out_features, [&](){return dis(gen);});
    }

    ~linear() {
        delete weights, bias;
    }

    Matrix *multiply(Matrix *x) {
        assert(x->width == _in_features);
        Matrix *out = new Matrix(x->height, _out_features, [&](int row, int column){
            return bias->values[column];
        });
        
        // https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html
        // x * transpose(weight) + bias
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,  // Options
                    x->height, out->width, x->width,          // Matrix Dimensions
                    1.0, x->values, x->width,                 // A Matrix
                    weights->values, x->width,                // B Matrix
                    1.0, out->values, out->width);            // C Matrix
        return out;
    }
};
#pragma once

#include "tensor.h"
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
    Tensor weights;
    Tensor bias;

    Linear(int in_features, int out_features) {
        random_device rd;
        mt19937 gen(rd());

        // Random-ness per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        uniform_real_distribution<double> dis(-sqrt(1./in_features), sqrt(1./in_features));
        this->_in_features = in_features;
        this->_out_features = out_features;
        
        weights = Tensor(out_features, in_features, [&](){return dis(gen);});
        bias = Tensor(1, out_features, [&](){return dis(gen);});
    }

    void multiply(const Tensor &x, Tensor &out) const {
        assert(x.width == _in_features);
        Tensor::setup_output_tensor(x.height, _out_features, out);
        for (int r = 0; r < out.height; ++r) {
            for (int c = 0; c < out.width; ++c) {
                out.values[r * out.width + c] = bias.values[c];
            }
        }
        Tensor::tensor_multiply(x, false, weights, true, out);
    }
};
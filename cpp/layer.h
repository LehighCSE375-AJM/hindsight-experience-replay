#pragma once

#include "linear.h"

enum ActivationFunction {
    RELU, TANH, NONE
};

// If this is too high it'll give a bunch of nans. 
const double LEARNING_RATE = 0.00001; // This should not be a constant (also it shouldn't be defined in this file)

class layer {
private:
    // The input and output matrixes from the last forward call
    Matrix in;
    Matrix preactiv_out;

    linear *lin;
    // I enjoy calling variables fun (you could says I find it fun)
    ActivationFunction fun;
public:
    layer(int in_features, int out_features, ActivationFunction fun) {
        this->lin = new linear(in_features, out_features);
        this->fun = fun;
    }

    ~layer() {
        delete this->lin;
    }

    Matrix forward(Matrix &x) {
        // Copy may not be required.
        this->in = x;
        this->preactiv_out = lin->multiply(x);
        Matrix ret;
        switch (this->fun) {
            case RELU:
                ret = preactiv_out.relu();
                break;
            case TANH:
                ret = preactiv_out.tanh();
                break;
            case NONE:
                // What a boring activation function
                ret = preactiv_out.copy();
                break;
        }
        return ret;
    }

    // error_gradient is gradient of error with respect to the output of this function. (This function returns the previous layer's error gradient)
    Matrix compute_gradient(Matrix &error_gradient) {
        // This is one way to calculate the gradients. I don't use it since I'm pretty sure its wrong (doesn't do the backpropagation with respect to the weight matrix neurons properly)
        // preactivation_error_gradient = error_gradient x activation_gradient (x indicates element wise multiplication)
        // preactivation_error_gradient^T * input = weight gradient (* means actual matrix multiplication)
        // preactivation_error_gradient * weight = new error gradient
        // preactivation_error_gradient = bias gradient (nice and simple!)


        // Math in use:
        // preactivation_error_gradient = error_gradient x activation_gradient^T
        // preactivation_error_gradient * input = weight gradient
        // preactivation_error_gradient^T = bias gradient
        // weight^T * preactivation_error_gradient = new error gradient
        Matrix activation_gradient;
        switch (fun) {
            case RELU:
                activation_gradient = preactiv_out.relu_gradient();
                break;
            case TANH:
                activation_gradient = preactiv_out.tanh_gradient();
            case NONE:
                activation_gradient = preactiv_out.ones();
        }
        Matrix activation_gradient_transpose = activation_gradient.transpose();
        Matrix preactivation_error_gradient = error_gradient * activation_gradient_transpose;
        
        
        Matrix bias_gradient = preactivation_error_gradient.transpose();
        Matrix weight_gradient = Matrix::matrix_multiply(preactivation_error_gradient, false, in, false);
        Matrix new_error_gradient = Matrix::matrix_multiply(lin->weights, true, preactivation_error_gradient, false);

        // This updates the model. Will have to be updated to use Adam optimizer. 
        lin->weights = lin->weights + (weight_gradient * LEARNING_RATE);
        lin->bias = lin->bias + (bias_gradient * LEARNING_RATE);
        return new_error_gradient;
    }
};
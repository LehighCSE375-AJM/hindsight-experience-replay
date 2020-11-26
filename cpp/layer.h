#pragma once

#include "linear.h"

enum ActivationFunction {
    RELU, TANH, NONE
};

// If this is too high it'll give a bunch of nans. 
const double LEARNING_RATE = 0.00001; // This should not be a constant (also it shouldn't be defined in this file)

class layer {
private:
    linear *lin;

    Matrix weight_gradient;
    Matrix bias_gradient;
    Matrix out_error_gradient;
    Matrix activation_gradient_transpose;

    // The input and output matrixes from the last forward call
    Matrix in;
    Matrix preactiv_out;
    Matrix out;

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

    Matrix& grad() {
        return &this->out_error_gradient;
    }

    Matrix& forward(Matrix &x) {
        this->in = x;
        if (out.height != x.height || out.width != lin->bias.width) {
            preactiv_out = Matrix(x.height, lin->bias.width, [](){ return 0; });
        }

        lin->multiply(x, this->preactiv_out);
        if (out.height != preactiv_out.height || out.width != preactiv_out.width) {
            out = Matrix(preactiv_out.height, preactiv_out.width, [](){return 0.;});
        }
        switch (this->fun) {
            case RELU:
                preactiv_out.relu(out);
                break;
            case TANH:
                preactiv_out.tanh(out);
                break;
            case NONE:
                // What a boring activation function
                preactiv_out.copy(out);
                break;
            default:
                throw runtime_error("Unrecognized activation function");
        }
        return out;
    }

    // error_gradient is gradient of error with respect to the output of this function. (This function returns the previous layer's error gradient)
    Matrix& compute_gradient(Matrix &error_gradient) {
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
        switch (fun) {
            case RELU:
                preactiv_out.relu_gradient();
                break;
            case TANH:
                preactiv_out.tanh_gradient();
            case NONE:
                preactiv_out.ones();
        }
        if (activation_gradient_transpose.width != preactiv_out.height || activation_gradient_transpose.height != preactiv_out.width) {
            activation_gradient_transpose = Matrix(preactiv_out.width, preactiv_out.height, [](){ return 0; });
        }
        preactiv_out.transpose(activation_gradient_transpose);
        error_gradient.mul_(activation_gradient_transpose);
        
        // Also there's a chance this doesn't even do anything. (it should in theory help because there will be less mallocing, but running the code shows mixed results)
        // I think it might be small enough where it doesn't make a difference. 
        // I have a rather strong distaste for all these if statements. There definitely needs to be a better way to do this. 
        // Could do it within these different function calls, but I would suspect that would hide errors which would be bad. 
        // The constructor could also work, but then we'd need to pass in some information about the dimensions of the incoming error gradient. 
        if (bias_gradient.width != error_gradient.height || bias_gradient.height != error_gradient.width) {
            bias_gradient = Matrix(error_gradient.width, error_gradient.height, [](){ return 0; });
        }
        error_gradient.transpose(bias_gradient);
        // In theory this if statement and the one below only occur the first time this is run, so this could be written more efficiently.
        if (weight_gradient.width != lin->weights.width || weight_gradient.height != lin->weights.height) {
            weight_gradient = Matrix(lin->weights.height, lin->weights.width, [](){ return 0; });
        } else {
            weight_gradient.mul_(0);            
        }
        Matrix::matrix_multiply(error_gradient, false, in, false, weight_gradient);

        if (out_error_gradient.width != error_gradient.width || out_error_gradient.height != lin->weights.width) {
            out_error_gradient = Matrix(lin->weights.width, error_gradient.width, [](){ return 0; });
        } else {
            out_error_gradient.mul_(0);
        }
        Matrix::matrix_multiply(lin->weights, true, error_gradient, false, out_error_gradient);
        // This updates the model. Will have to be updated to use Adam optimizer.
        lin->weights.submul_(weight_gradient, LEARNING_RATE); 
        lin->bias.submul_(bias_gradient, LEARNING_RATE); 
        return out_error_gradient;
    }
};
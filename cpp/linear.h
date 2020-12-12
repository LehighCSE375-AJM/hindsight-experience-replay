#pragma once

#include "tensor.h"
#include <random>
#include <curand_kernel.h>

enum ActivationFunction {
    RELU, TANH, NONE
};

// If this is too high it'll give a bunch of nans. 
const double LEARNING_RATE = 0.00001; // This should not be a constant (also it shouldn't be defined in this file)

// Based off of pytorch nn.Linear
class Linear {
private:
    // This is generally the math I followed (there are more specifics later about how this works with matricies):
    // y = xW^T + b (x equals the z values of previous layers)
    // z = f(y) where f(y) is the activation function. 
    // E = (a - z)^2 for the last layer where a is the correct value
    // dE/dz = -2(a - z) for the last layer
    // dz/dy = f'(y) (the gradient of the activation function depends on the activation function)
    // dy/dw = x (just take the derivative of y = xW^T + b for these three gradients)
    // dy/db = 1
    // dy/dx = W
    // 
    // dE/dx = dE/dz * dz/dy * dy/dx (since x equals the z value of previous layers this equals dE/dz for previous layers)
    // dE/dw = dE/dz * dz/dy * dy/dw
    // dE/db = dE/dz * dz/dy * dy/db = dE/dz * dz/dy (since dy/db = 1)
    int _in_features;
    int _out_features;

    // Error gradient w.r.t. the weights
    // Tensor weight_gradient;
    // Error gradient w.r.t. the bias
    // Tensor bias_gradient;
    // Error gradient w.r.t. the input of the next layer (whats passed into the following compute_gradient function)
    Tensor out_error_gradient;
    // An intermediate tensor for computing the gradient (represents the gradient of the error w.r.t. the output of linear layer)
    Tensor _activation_gradient_transpose;

    // The previous input value for the last forward call (x in y = xW^T + b)
    Tensor in;
    // The output from before the activation function of the last forward call (y in y = xW^T + b)
    Tensor preactiv_out;
    // A tensor which represents the output of last forward call
    Tensor _out;

    // I enjoy calling variables fun (you could says I find it fun)
    ActivationFunction fun;
public:
    Tensor weights;
    Tensor bias;
    
    Linear(int in_features, int out_features, ActivationFunction fun) {
        random_device rd;
        mt19937 gen(rd());

        // Random-ness per https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        uniform_real_distribution<double> dis(-sqrt(1./in_features), sqrt(1./in_features));
        this->_in_features = in_features;
        this->_out_features = out_features;
        
        weights = Tensor(out_features, in_features, [&](){return dis(gen);});
        bias = Tensor(1, out_features, [&](){return dis(gen);});

        this->fun = fun;
    }

    __device__ Linear(int in_features, int out_features, ActivationFunction fun, curandState &rand_state) {
	this->_in_features = in_features;
	this->_out_features = out_features;

	weights = Tensor(out_features, in_features, rand_state, in_features);
	bias = Tensor(1, out_features, rand_state, in_features);

	// FUN!
	this->fun = fun;
    }

    Tensor& grad() {
        return this->out_error_gradient;
    }

    Tensor& out() {
        return this->_out;
    }

    __host__ __device__ void multiply(const Tensor &x, Tensor &out) const {
        assert(x.width == _in_features);
        Tensor::setup_output_tensor(x.height, _out_features, out);
#ifdef __CUDA_ARCH__
	int t = threadIdx.x;
	while (t < out.height * out.width) {
		out.d_values[t] = bias.d_values[t % out.width];
		t += blockDim.x;
	}
#else
	for (int r = 0; r < out.height; ++r) {
            for (int c = 0; c < out.width; ++c) {
                out.values[r * out.width + c] = bias.values[c];
            }
        }
#endif
        Tensor::matrix_multiply(x, false, weights, true, out);
    }

    __host__ __device__ Tensor& forward(const Tensor &x) {
        this->in = x;
        this->multiply(x, this->preactiv_out);
        switch (this->fun) {
            case RELU:
                preactiv_out.relu(_out);
                break;
            case TANH:
                preactiv_out.tanh(_out);
                break;
            case NONE:
                // What a boring activation function
                preactiv_out.copy(_out);
                break;
        }
        return _out;
    }

    // error_gradient is gradient of error with respect to the output of this function. (This function returns the previous layer's error gradient)
    Tensor& compute_gradient(const Tensor &error_gradient) {
        // This is one way to calculate the gradients. I don't use it since I'm pretty sure its wrong (doesn't do the backpropagation with respect to the weight tensor neurons properly)
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
        preactiv_out.transpose(_activation_gradient_transpose);
        _activation_gradient_transpose.mul_(error_gradient);
        
        _activation_gradient_transpose.transpose(this->bias.grad());
        this->weights.grad().mul_(0);
        Tensor::matrix_multiply(_activation_gradient_transpose, false, in, false, this->weights.grad());

        out_error_gradient.mul_(0);
        Tensor::matrix_multiply(this->weights, true, _activation_gradient_transpose, false, out_error_gradient);
        // This updates the model. Will have to be updated to use Adam optimizer.
        return out_error_gradient;
    }
};

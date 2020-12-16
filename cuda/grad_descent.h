#pragma once

#include "optimizer.h"
#include "tensor.h"
#include <vector>

/**
 * Optimizer class implementing the Gradient Descent algorithm
 * 
 */
class GradientDescent: public Optimizer {
private:
	Tensor** params_;
	double lr = 1E-3L;
	int param_len_;

public:
	/**
	 * Constructor for the Gradient Descent Optimizer
	 * 
	 * NOTE: For some reason the default learning rate will create a lot of NANs
	 * @param params is an iterator of tensors in the order [weight, bias, weight ... ]
	 * @param lr is the learning rate, default 0.001
	 */
	__host__ __device__ GradientDescent(Tensor** params, int param_len, double lr = 1E-3L) {
		this->params_ = params;
		this->lr = lr;
		this->param_len_ = param_len;
	}
	
	/**
	 * Destructor for gradient descent optimizer
	 */
	__host__ __device__ ~GradientDescent() {
		// Nothing to destruct?
	}

	/**
	 * Take one step by subtracting the gradient * the learning rate from params
	 */
	__host__ __device__ void step() {
		for (int i = 0; i < param_len_; ++i) {
			Tensor *param = this->params_[i];
			if (param->gradient == NULL) {
				continue;
			}

			param->submul_(param->grad(), lr);
		}
	}
};

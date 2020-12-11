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
	vector<Tensor*> params_;
	double lr = 1E-3L;

public:
	/**
	 * Constructor for the Gradient Descent Optimizer
	 * 
	 * NOTE: For some reason the default learning rate will create a lot of NANs
	 * @param params is an iterator of tensors in the order [weight, bias, weight ... ]
	 * @param lr is the learning rate, default 0.001
	 */
	GradientDescent(vector<Tensor*> params, double lr = 1E-3L) {
		this->params_ = params;
		this->lr = lr;
	}
	
	/**
	 * Destructor for gradient descent optimizer
	 */
	~GradientDescent() {
		// Nothing to destruct?
	}

	/**
	 * Take one step by subtracting the gradient * the learning rate from params
	 */
	void step() {
		for (auto it = this->params_.begin(); it != this->params_.end(); ++it) {
			Tensor *param = *it;
			// cout << "step" << endl;
			// if (param->get_size() < 20) cout << *param << endl;
			if (param->gradient == NULL) {
				continue;
			}

			param->submul_(param->grad(), lr);
			// if (param->get_size() < 20) cout << *param << endl;
		}
	}
};
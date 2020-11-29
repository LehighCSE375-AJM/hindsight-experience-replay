#pragma once

#include "optimizer.h"
#include "tensor.h"
#include <vector>
#include <iterator>
#include <cmath>

class GradientDescent: public Optimizer {
private:
	vector<Tensor*> params_;
	double lr = 1E-3L;

public:
	GradientDescent(vector<Tensor*> params, double lr = 1E-3L) {
		this->params_ = params;
		this->lr = lr;
	}
	
	~GradientDescent() {
		
	}

	/**
	 * Take one step
	 */
	void step() {
		for (auto it = this->params_.begin(); it != this->params_.end(); ++it) {
			Tensor *param = *it;
			if (param->gradient == NULL) {
				continue;
			}

			param->submul_(param->grad(), lr);
		}
	}
};
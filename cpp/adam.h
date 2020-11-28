#include "matrix.h"
#include "layer.h"
#include <vector>
#include <iterator>
#include <cmath>


/**
 * Class meant to follow semantics of torch.optim.Adam
 * found at https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
 * extends https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
 * 
 * cpp implementation here
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/optim/adam.h
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/optim/adam.cpp
 */
class Adam {
private:
	struct Adam_Param_State {
		int64_t step = 0;
		Matrix exp_avg;
		Matrix exp_avg_sq;
	};
	
	vector<Layer> layers_;
	vector<Adam_Param_State> state_;
	double lr = 1E-3L;
	double betas[2] = {0.9L, 0.999L};
	double eps = 1E-8L;
public:

	// @param lr is the learning rate, default 0.001
	Adam(vector<Layer> params, double lr = 1E-3L) {
		this->layers_ = params;
		this->lr = lr;
	}

	// @param lr is the learning rate, default 0.001
	Adam(Layer& params, double lr = 1E-3L) {
		this->layers_.push_back(params);
		this->lr = lr;
	}

	/**
	 * Take one Adam step
	 */
	void step()  {
		int index = 0;
		for (auto it = this->layers_.begin(); it != this->layers_.end(); ++it, ++index) {
			// State initialization
			if(index >= this->state_.size()) {
				Adam_Param_State state = {0, it->lin->weights.zeros(), it->lin->weights.zeros()};
				this->state_.push_back(state);
			}

			Adam_Param_State &state = this->state_.at(index);
			Matrix &grad = it->grad();

			state.step += 1;
			double beta1 = this->betas[0];
			double beta2 = this->betas[1];

			double bias_correction1 = 1 - std::pow(beta1, state.step);
			double bias_correction2 = 1 - std::pow(beta2, state.step);

			// Update biased first moment estimate 
			// m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t 
			state.exp_avg.mul_(beta1).add_(grad, 1 - beta1);

			// Update biased second raw moment estimate
			// m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t^2
			state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

			// Copy Matrix using cute in place operations
			Matrix denom = state.exp_avg_sq;

			// Compute v-hat
			denom.div_(bias_correction1).sqrt_().add_(this->eps);

			
			// Update parameters
			// theta_t = theta_(t-1) + (-(step_size / (1 - beta_1^t)) * m_t / (v-hat_t + epsilon))
			// We bake the bias_correction1 value for computing m-hat into the constant multiplier
			// Intead of allocating and calculating m-hat like they do in the pseudocode
			it->lin->weights.addcdiv_(state.exp_avg, denom, -this->lr / bias_correction1);
		}
	}
};
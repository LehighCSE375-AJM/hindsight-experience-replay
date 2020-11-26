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
		Matrix& exp_avg;
		Matrix& exp_avg_sq;
	};
	
	vector<Matrix&> params;
	vector<Adam_Param_State&> state_;
	double lr = 1E-3L;
	double betas[2] = {0.9L, 0.999L};
	double eps = 1E-8L;
public:

	// @param lr is the learning rate, default 0.001
	Adam(vector<Matrix&> params, double lr = 1E-3L) {
		this->params = params;
		this->lr = lr;
	}

	// @param lr is the learning rate, default 0.001
	Adam(Matrix& params, double lr = 1E-3L) {
		this->params.push_back[params];
		this->lr = lr;
	}

	/**
	 * Take one Adam step
	 */
	void step()  {
		int index = 0;
		for (auto it = this->params.begin(); it != container.end(); ++it, ++index) {
			// State initialization
			if(index >= this->state_->size()) {
				Adam_Param_State state;
				state->exp_avg = p->zeros();
				state->exp_avg_sq = p->zeros();
				this->state_.push_back(state);
			}

			Adam_Param_State &state = this->state_.at(index);
			Matrix &grad = ????;

			state.step += 1;
			auto beta1 = this->betas[0];
			auto beta2 = this->betas[1];

			auto bias_correction1 = 1 - std::pow(beta1, state.step);
			auto bias_correction2 = 1 - std::pow(beta2, state.step);

			// Decay the first and second moment running average coefficient
			state.exp_avg.mul_(beta1).add_(grad, 1 - beta1);
			state.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

			Matrix denom = Matrix(state.exp_avg_sq);
			denom.sqrt_().div_(std::sqrt(bias_correction2)).add_(this->eps);

			auto step_size = this->lr / bias_correction1;
			p->addcdiv_(state.exp_avg, denom, -step);
		}
	}
};
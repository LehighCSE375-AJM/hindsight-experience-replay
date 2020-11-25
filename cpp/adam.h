#include "matrix.h"
#include <vector>
#include <iterator>
#import <cmath>


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
		struct Adam_Param_State;
		struct Tensor {
			Matrix* data;
			Tensor* grad;
			Adam_Param_State* state;
			bool requires_grad = true;
			bool is_leaf = true;

			void detach_() { this->is_leaf = true; };
			void requires_grad_(bool requires_grad=true) { this->requires_grad=requires_grad; };
			void zero_() {
				for (size_t i = 0; i < data->height * data->width; i++) {
					data->values[i] = 0;
				}
			};
		};
		struct Adam_Param_State {
			int64_t step = 0;
			Tensor* exp_avg;
			Tensor* exp_avg_sq;
			Tensor* max_exp_avg_sq;
		};
		struct Adam_Group {
			vector<Tensor*> params;
			vector<Adam_Param_State> state_;
			double lr = 1E-3L;
			double betas[2] = {0.9L, 0.999L};
			double eps = 1E-8L;
			double weight_decay = 0;
			bool amsgrad = false;
		};
		vector<Adam_Group*> param_groups_;
		Adam_Group defaults_;
public:

		// @param lr is the learning rate, default 0.001
		Adam(vector<Matrix*> params, double lr = 1E-3L) {
			Adam_Group* group = new Adam_Group();
			group->lr = lr;

			for (auto& data : params) {
				Tensor *input = new Tensor();
				input->data = data;
				group->params.push_back(input);
			}

			this->param_groups_.push_back(group);
		}

		// @param lr is the learning rate, default 0.001
		Adam(Matrix* params, double lr = 1E-3L) {
			Adam_Group* group = new Adam_Group();
			group->lr = lr;

			Tensor *input = new Tensor();
			input->data = params;
			group->params.push_back(input);

			this->param_groups_.push_back(group);
		}

		/**
		 * Zero out the entire 
		 */
		void zero_grad(bool set_to_none = false) {
			for (auto& group : this->param_groups_) {
				for (auto& param : group->params) {
					if (set_to_none) {
						param->grad = nullptr;
					}
					else {
						param->grad->requires_grad_(false);
						param->grad->zero_();
					}
				}
			}
		}

		/**
		 * Take one Adam step
		 */
		void step()  {
			for (auto& group : this->param_groups_) {
				for (auto& p : group->params) {
					if (p->grad == nullptr) {
						continue;
					}

					// State initialization
					if(p->state == nullptr) {
						auto state = new Adam_Param_State();
						p->state->step = 0;
						// Exponential moving average of gradient values

						p->state->exp_avg = new Tensor();
						p->state->exp_avg->data = &p->data->zeros();
						p->state->exp_avg_sq->data = &p->data->zeros();

						if(group->amsgrad) {
							// Maintains max of all exp. moving avg. of sq. grad. values
							state->max_exp_avg_sq->data = &p->data->zeros();
						}
					}

					p->state->step += 1;
					auto beta1 = group->betas[0];
					auto beta2 = group->betas[1];

					auto bias_correction1 = 1 - std::pow(beta1, p->state->step);
					auto bias_correction2 = 1 - std::pow(beta2, p->state->step);

					// Decay the first and second moment running average coefficient
					p->state->exp_avg->data->mul_(beta1).add_(*(p->grad->data), 1 - beta1);
					p->state->exp_avg_sq->data->mul_(beta2).addcmul_(*(p->grad->data), *(p->grad->data), 1 - beta2);

					Matrix denom = Matrix(exp_avg_sq->data);
					denom.sqrt_().div_(std::sqrt(bias_correction2)).add_(options->eps);

					auto step_size = group->lr / bias_correction1;
					p->data->addcdiv_(p->state->exp_avg, denom, -step);
				}
			}
		}
};
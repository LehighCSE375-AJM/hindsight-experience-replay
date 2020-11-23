#include "matrix.h"
#include <list>


/**
 * Class meant to follow semantics of torch.optim.Adam
 * found at https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
 * extends https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
 */
class Adam {
private:
		list<Matrix*> param_groups;
		double lr = 0.01;
		double betas[2] = {0.9, 0.999};
		double eps = 0.00000008;
		double weight_decay = 0;
		bool amsgrad = false;
public:
		// @param params should be an iterable of tensors of dicts
		// @param lr is the learning rate, default 0.01
		Adam(list<Matrix*> params, double lr) {
			this->lr = lr;

			for (auto it = params.begin(); it != params.end(); ++it) {
				param_groups.push_back(*it);
			}
		}

		void zero_grad() {
			for (auto it = this->param_groups.begin(); it != this->param_groups.end(); ++it) {
				
			}
		}

		void step() {

		}
};
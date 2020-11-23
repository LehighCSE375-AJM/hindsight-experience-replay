#include "matrix.h"
#include <list>


/**
 * Class meant to follow semantics of torch.optim.Adam
 * found at https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
 * extends https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
 */
class Adam {
private:
		struct Tensor {
			Matrix* data;
			Tensor* grad;
			bool requires_grad = true;

			void detach_();
			void requires_grad_(bool requires_grad=true) {this->requires_grad=requires_grad};
			void zero_(); // Fill self with zeros
		};
		struct Adam_Group {
			list<Tensor*> params;
			double lr = 1E-3L;
			double betas[2] = {0.9L, 0.999L};
			double eps = 1E-8L;
			double weight_decay = 0;
			bool amsgrad = false;
		};
		list<Adam_Group*> param_groups;
		Adam_Group defaults;
public:

		// @param lr is the learning rate, default 0.01
		Adam(list<Matrix*> params, double lr = 1E-3L) {
			Adam_Group* group = new Adam_Group();
			group->lr = lr;

			for (auto it = params.begin(); it != params.end(); ++it) {
				Tensor *input = new Tensor();
				input->data = *it;
				group->params.push_back(input);
			}

			param_groups.push_back(group);
		}

		// @param lr is the learning rate, default 0.01
		Adam(Matrix* params, double lr = 1E-3L) {
			Adam_Group* group = new Adam_Group();
			group->lr = lr;

			Tensor *input = new Tensor();
			input->data = params;
			group->params.push_back(input);

			param_groups.push_back(group);
		}

		/**
		 * 
		 */
		void zero_grad(bool set_to_none = false) {
			for (auto param_group = this->param_groups.begin(); param_group != this->param_groups.end(); ++param_group) {
				/* for each param in param_group['params'] {
					if (set_to_none) {
						(*param)->grad = NULL;
					}
					else {
						if (param->grad.grad_fn != NULL) {
							param->grad.detach_();
						}
						else {
							param->grad.requires_grad_(false);
						}
						param->grad.zero_();
					}
				} */
			}
		}

		void step() {

		}
};
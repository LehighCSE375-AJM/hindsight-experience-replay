#include "matrix.h"
#include <list>


/**
 * Class meant to follow semantics of torch.optim.Adam
 * found at https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
 * extends https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
 */
class Adam {
private:
		struct Adam_Group {
			Matrix* params;
			double lr = 1E-3L;
			double betas[2] = {0.9L, 0.999L};
			double eps = 1E-8L;
			double weight_decay = 0;
			bool amsgrad = false;
		};
		list<Adam_Group*> param_groups;
public:
		// @param params should be an iterable of tensors of dicts
		// @param lr is the learning rate, default 0.01
		Adam(list<Matrix*> params, double lr = 1E-3L) {
			Adam_Group* group = new Adam_Group();
			group->lr = lr;

			for (auto it = params.begin(); it != params.end(); ++it) {
				Adam_Group* group = new Adam_Group();
				group->lr = lr;
				group->params = *it;
				param_groups.push_back(group);
			}
		}

		void zero_grad(bool set_to_none = false) {
			for (auto it = this->param_groups.begin(); it != this->param_groups.end(); ++it) {
				for () {
					if (set_to_none) {

					}
					else {
						  
					}
				}
			}
		}

		void step() {

		}
};
#pragma once

#include "optimizer.h"
#include "tensor.h"
#include <vector>
#include <cmath>

/**
 * Optimizer class implementing the Adam algorithm
 * 
 * Class is meant to follow semantics of torch.optim.Adam
 * found at https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
 * extends https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
 * 
 * cpp implementation here
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/optim/adam.h
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/optim/adam.cpp
 * 
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/optim/optimizer.h
 * https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/optim/optimizer.cpp
 */
class Adam: public Optimizer {
private:
	struct Adam_Param_State {
		int64_t step = 0;
		Tensor* exp_avg;
		Tensor* exp_avg_sq;
		Tensor* denom;
	};
	
	int param_len_;
	Tensor **params_;
	Adam_Param_State *state_;
	double lr = 1E-3L;
	double betas[2] = {0.9L, 0.999L};
	double eps = 1E-8L;
public:

	/**
	 * Constructor for the Adam optimizer
	 * 
	 * NOTE: We call the parameter learning rate but the Adam paper calls it step size
	 * @param params is an iterator of tensors in the order [weight, bias, weight ... ]
	 * @param lr is the learning rate, default 0.001
	 */
	__host__ __device__ Adam(Tensor **params, int param_len, double lr = 1E-3L) {
		this->params_ = params;
		this->lr = lr;
		this->param_len_ = param_len;

#ifdef __CUDA_ARCH__
		__shared__ Adam_Param_State *new_d_ptr;
		if (threadIdx.x == 0) {
			new_d_ptr = new Adam_Param_State[param_len];
		}
		__syncthreads();
		this->state_ = new_d_ptr;
#else
		this->state_ = new Adam_Param_State[param_len];
#endif
	}

	/**
	 * Destructor for Adam optimizer
	 */
	__host__ __device__ ~Adam() {
#ifdef __CUDA_ARCH__
		if (threadIdx.x == 0) {
#endif
		for (int i = 0; i < param_len_; ++i) {
			if (this->state_[i].exp_avg != NULL) {
				delete this->state_[i].exp_avg;
				delete this->state_[i].exp_avg_sq;
				delete this->state_[i].denom;
			}
		}
#ifdef __CUDA_ARCH__
		}
#endif
	}

	/**
	 * Take one Adam step
	 */
	__host__ __device__ void step()  {
		for (int i = 0; i < param_len_; ++i) {
			Tensor *param = this->params_[i];
			if (param->gradient == NULL) {
				continue;
			}
			Tensor *grad = param->gradient;

			if(this->state_[i].exp_avg == NULL) {
#ifdef __CUDA_ARCH__
				if (threadIdx.x == 0) {
					this->state_[i] = {0, new Tensor(param->height, param->width), new Tensor(param->height, param->width), new Tensor(param->height, param->width)};
				}
				__syncthreads();
#else
				this->state_[i] = {0, new Tensor(param->height, param->width), new Tensor(param->height, param->width), new Tensor(param->height, param->width)};
#endif
			}
			Adam_Param_State &state = this->state_[i];
			// TODO this can't be done by all the threads for cuda (but also needs to be read by all the threads shortly after)
#ifdef __CUDA_ARCH__
			if (threadIdx.x == 0) {
				state.step += 1;
			}
			__syncthreads();
#else
			state.step += 1;
#endif

			double beta1 = this->betas[0];
			double beta2 = this->betas[1];

			double bias_correction1 = 1 - std::pow(beta1, state.step);
			double bias_correction2 = 1 - std::pow(beta2, state.step);

			// Update biased first moment estimate 
			// m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t 
			// 1.8e+07 without
			state.exp_avg->mul_(beta1).addmul_(*grad, 1 - beta1);

			// Update biased second raw moment estimate
			// m_t = beta_1 * m_(t-1) + (1 - beta_1) * g_t^2
			// 1.68e+7 without
			state.exp_avg_sq->mul_(beta2).addsquaremul_(*grad, 1 - beta2);

			// Copy Tensor
			state.exp_avg_sq->copy(*(state.denom));

			// Compute v-hat
			// 2.3e+7 without
			state.denom->div_(bias_correction1).sqrt_().add_(this->eps);
			
			// Update parameters
			// theta_t = theta_(t-1) + (-(step_size / (1 - beta_1^t)) * m_t / (v-hat_t + epsilon))
			// We bake the bias_correction1 value for computing m-hat into the constant multiplier
			// Intead of allocating and calculating m-hat like they do in the pseudocode
			// 1.78e+7 without
			param->addcdiv_(*state.exp_avg, *state.denom, -this->lr / bias_correction1);
			// 2.45e+07 with everything in place.
		}
	}
};

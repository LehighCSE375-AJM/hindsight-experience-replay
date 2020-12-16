#pragma once

#include "linear.h"

#include "optimizer.h"
#include "adam.h"
#include "grad_descent.h"

#define OBSERVATION_DIM 1 // TODO
#define GOAL_DIM 1 // TODO
#define ACTION_DIM 1 // TODO
#define NEURONS 256

class Actor {
private:
	Linear fc1 = Linear(OBSERVATION_DIM + GOAL_DIM, NEURONS, RELU);
	Linear fc2 = Linear(NEURONS, NEURONS, RELU);
	Linear fc3 = Linear(NEURONS, NEURONS, RELU);
	Linear action_out = Linear(NEURONS, ACTION_DIM, TANH);

	Tensor max_action;

	Tensor out;

public:
	explicit Actor(Tensor &max_action) {
		this->max_action = max_action;
	};

	Tensor& forward(Tensor &x) {
		out = fc1.forward(x);
		out = fc2.forward(out);
		out = fc3.forward(out);
		out = action_out.forward(out);
		out.mul_(max_action);
		return out;
	}

	vector<Tensor*> parameters() {
		vector<Tensor*> result;
		result.push_back(&fc1.weights);
		result.push_back(&fc1.bias);
		
		result.push_back(&fc2.weights);
		result.push_back(&fc2.bias);
		
		result.push_back(&fc3.weights);
		result.push_back(&fc3.bias);
		
		result.push_back(&action_out.weights);
		result.push_back(&action_out.bias);

		return result;
	}
};


class Critic {
private:
	Linear fc1;
	Linear fc2;
	Linear fc3;
	Linear q_out;
	Tensor max_action;
	// Only an Adam optimizer since we don't have to use virtual functions this way (which I believe are fairly slow, but not certain)
	Adam* optim;

	// Miscelanious intermediate matricies. 
	Tensor _adjusted_actions;
	Tensor _loss_gradient;
	Tensor _adjusted_in;

public:

	__device__ Critic(int obs, int goal, int action, Tensor &max_action, curandState &rand_state) {
		this->max_action = max_action;
		
		fc1 = Linear(obs + goal + action, 256, RELU, rand_state);
		fc2 = Linear(256, 256, RELU, rand_state);
		fc3 = Linear(256, 256, RELU, rand_state);
		q_out = Linear(256, 1, NONE, rand_state);
		
		
		__shared__ Adam *new_d_ptr;
                if (threadIdx.x == 0) {
                        new_d_ptr = new Adam(this->parameters(), 8, 0.001);
			//new_d_ptr = new GradientDescent(this->parameters(), 8, 0.00001);
                }
                __syncthreads();
                this->optim = new_d_ptr;
		__syncthreads();
		// this->optim = new GradientDescent(this->parameters(), 0.00001);
	};

	Critic(int obs, int goal, int action, Tensor &max_action) {
                this->max_action = max_action;

		fc1 = Linear(obs + goal + action, 256, RELU);
		fc2 = Linear(256, 256, RELU);
		fc3 = Linear(256, 256, RELU);
		q_out = Linear(256, 1, NONE);

                this->optim = new Adam(this->parameters(), 8, 0.00001);
                //this->optim = new GradientDescent(this->parameters(), 8, 0.00001);
        };
	

	__host__ __device__ ~Critic() {
#ifdef __CUDA_ARCH__
		if (threadIdx.x == 0) {
			delete this->optim;
		}
#else
		delete this->optim;
#endif
	}

	__host__ __device__ Tensor& forward(const Tensor &x, const Tensor &actions) {
		actions.copy(_adjusted_actions);
		_adjusted_actions.div_(max_action);
		Tensor::vector_concat_onto(x, _adjusted_actions, _adjusted_in);
		fc1.forward(_adjusted_in);
		fc2.forward(fc1.out());
		fc3.forward(fc2.out());
		q_out.forward(fc3.out());
		return q_out.out();
	}

	__host__ __device__ void backprop(const Tensor &actual, const Tensor &predicted) {
		predicted.copy(_loss_gradient);
		_loss_gradient.sub_(actual);
		q_out.compute_gradient(_loss_gradient);
		fc3.compute_gradient(q_out.grad());
		fc2.compute_gradient(fc3.grad());
		fc1.compute_gradient(fc2.grad());
		optim->step();
	}

	// Only one thread calls this method from the gpu
	__host__ __device__ Tensor** parameters() {
		Tensor** out = new Tensor*[8];
		out[0] = &fc1.weights;
		out[1] = &fc1.bias;
		out[2] = &fc2.weights;
		out[3] = &fc2.bias;
		out[4] = &fc3.weights;
		out[5] = &fc3.bias;
		out[6] = &q_out.weights;
		out[7] = &q_out.bias;
		return out;
	}
};

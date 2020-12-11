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
	vector<Tensor *> params;

	explicit Actor(Tensor &max_action) {
		this->max_action = max_action;
	};

	explicit Actor(int obs, int goal, int action, Tensor &max_action) {
		this->max_action = max_action;
		fc1 = Linear(obs + goal, 256, RELU);
		fc2 = Linear(256, 256, RELU);
		fc3 = Linear(256, 256, RELU);
		action_out = Linear(256, action, TANH);

		params.push_back(&(fc1.weights));
		params.push_back(&(fc1.bias));
		params.push_back(&(fc2.weights));
		params.push_back(&(fc2.bias));
		params.push_back(&(fc3.weights));
		params.push_back(&(fc3.bias));
		params.push_back(&(action_out.weights));
		params.push_back(&(action_out.bias));
	};

	explicit Actor(const Actor &other) {
		this->max_action = other.max_action;
		fc1 = other.fc1;
		fc2 = other.fc2;
		fc3 = other.fc3;
		action_out = other.action_out;

		params.push_back(&(fc1.weights));
		params.push_back(&(fc1.bias));
		params.push_back(&(fc2.weights));
		params.push_back(&(fc2.bias));
		params.push_back(&(fc3.weights));
		params.push_back(&(fc3.bias));
		params.push_back(&(action_out.weights));
		params.push_back(&(action_out.bias));
	}

	Tensor& forward(Tensor &x) {
		out = fc1.forward(x);
		out = fc2.forward(out);
		out = fc3.forward(out);
		out = action_out.forward(out);
		// cout << out << endl;
		// out.mul_(max_action); // multiplying by 1?
		return out;
	}

	vector<Tensor *> parameters() {
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

	void copy_transform_params(Actor* source,
								function<void(Tensor *, Tensor *)> copy_transform) {
		vector<Tensor *> source_params = source->parameters();
		
		copy_transform(&(fc1.weights), (source_params[0]));
		copy_transform(&(fc1.bias), (source_params[1]));

		copy_transform(&(fc2.weights), (source_params[2]));
		copy_transform(&(fc2.bias), (source_params[3]));

		copy_transform(&(fc3.weights), (source_params[4]));
		copy_transform(&(fc3.bias), (source_params[5]));

		copy_transform(&(action_out.weights), (source_params[6]));
		copy_transform(&(action_out.bias), (source_params[7]));
	}
};


class Critic {
private:
	Linear fc1 = Linear(OBSERVATION_DIM + GOAL_DIM + ACTION_DIM, NEURONS, RELU);
	Linear fc2 = Linear(NEURONS, NEURONS, RELU);
	Linear fc3 = Linear(NEURONS, NEURONS, RELU);
	Linear q_out = Linear(NEURONS, 1, NONE);
	Tensor max_action;
	Optimizer* optim;

	// Miscelanious intermediate matricies. 
	Tensor _adjusted_actions;
	Tensor _loss_gradient;
	Tensor _adjusted_in;

public:

	explicit Critic(Tensor &max_action) {
		this->max_action = max_action;
		// this->optim = new Adam(this->parameters(), 0.00001);
		this->optim = new GradientDescent(this->parameters(), 0.00001);
	};

	explicit Critic(int obs, int goal, int action, Tensor& max_action)
	{
		this->max_action = max_action;
		fc1 = Linear(obs + goal + action, 256, RELU);
		fc2 = Linear(256, 256, RELU);
		fc3 = Linear(256, 256, RELU);
		q_out = Linear(256, 1, NONE);
		this->optim = new GradientDescent(this->parameters(), 0.00001);
	}

	explicit Critic(const Critic &other)
	{
		this->max_action = other.max_action;
		fc1 = other.fc1;
		fc2 = other.fc2;
		fc3 = other.fc3;
		q_out = other.q_out;
		this->optim = new GradientDescent(this->parameters(), 0.00001);
	}

	~Critic() {
		delete this->optim;
	}

	Tensor& forward(const Tensor &x, const Tensor &actions) {
		actions.copy(_adjusted_actions);
		// _adjusted_actions.div_(max_action);
		Tensor::vector_concat_onto(x, _adjusted_actions, _adjusted_in);
		fc1.forward(_adjusted_in);
		fc2.forward(fc1.out());
		fc3.forward(fc2.out());
		q_out.forward(fc3.out());
		return q_out.out();
	}

	void backprop(const Tensor &actual, const Tensor &predicted) {
		predicted.copy(_loss_gradient);
		_loss_gradient.sub_(actual);
		q_out.compute_gradient(_loss_gradient);
		fc3.compute_gradient(q_out.grad());
		fc2.compute_gradient(fc3.grad());
		fc1.compute_gradient(fc2.grad());
		optim->step();
	}

	void backprop(const double actual, const Tensor &predicted) {
		predicted.copy(_loss_gradient);
		// _loss_gradient.sub_(actual);
		_loss_gradient = _loss_gradient - actual;
		q_out.compute_gradient(_loss_gradient);
		fc3.compute_gradient(q_out.grad());
		fc2.compute_gradient(fc3.grad());
		fc1.compute_gradient(fc2.grad());
		optim->step();
	}

	vector<Tensor*> parameters() {
		vector<Tensor*> result;
		result.push_back(&fc1.weights);
		result.push_back(&fc1.bias);
		
		result.push_back(&fc2.weights);
		result.push_back(&fc2.bias);
		
		result.push_back(&fc3.weights);
		result.push_back(&fc3.bias);
		
		result.push_back(&q_out.weights);
		result.push_back(&q_out.bias);

		return result;
	}

	void copy_transform_params(Critic* source, vector<Tensor *> params,
								function<void(Tensor *, Tensor *)> copy_transform) {
		vector<Tensor *> source_params = source->parameters();
		
		for (size_t i = 0; i < source_params.size(); i++)
		{
			copy_transform(params[i], source_params[i]);
		}
	}
};
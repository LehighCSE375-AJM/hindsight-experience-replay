#pragma once

#include "layer.h"

#define OBSERVATION_DIM 1 // TODO
#define GOAL_DIM 1 // TODO
#define ACTION_DIM 1 // TODO
#define NEURONS 256

class Actor {
private:
    Layer fc1 = Layer(OBSERVATION_DIM + GOAL_DIM, NEURONS, RELU);
    Layer fc2 = Layer(NEURONS, NEURONS, RELU);
    Layer fc3 = Layer(NEURONS, NEURONS, RELU);
    Layer action_out = Layer(NEURONS, ACTION_DIM, TANH);

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
};


class Critic {
private:
    Layer fc1 = Layer(OBSERVATION_DIM + GOAL_DIM + ACTION_DIM, NEURONS, RELU);
    Layer fc2 = Layer(NEURONS, NEURONS, RELU);
    Layer fc3 = Layer(NEURONS, NEURONS, RELU);
    Layer q_out = Layer(NEURONS, 1, NONE);
    Tensor max_action;

    // Miscelanious intermediate matricies. 
    Tensor _adjusted_actions;
    Tensor _loss_gradient;
    Tensor _adjusted_in;

public:

    explicit Critic(Tensor &max_action) {
        this->max_action = max_action;
    };

    Tensor& forward(const Tensor &x, const Tensor &actions) {
        actions.copy(_adjusted_actions);
        _adjusted_actions.div_(max_action);
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
    }
};
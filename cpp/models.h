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

    Matrix max_action;

public:
    Actor(Matrix &max_action) {
        this->max_action = max_action;
    };

    Matrix forward(Matrix &x) {

        Matrix m1 = fc1.forward(x);
        Matrix m2 = fc2.forward(m1);
        Matrix m3 = fc3.forward(m2);
        Matrix out = action_out.forward(m3);
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
    Matrix max_action;

public:

    Critic(Matrix &max_action) {
        this->max_action = max_action;
    };

    Matrix& forward(Matrix &x, Matrix &actions) {
        Matrix adjusted_actions = actions / max_action;
        Matrix in = Matrix::vector_concat(x, adjusted_actions);
        Matrix &out = fc1.forward(in);
        out = fc2.forward(out);
        out = fc3.forward(out);
        out = q_out.forward(out);
        return out;
    }

    void backprop(Matrix &actual, Matrix &predicted) {
        Matrix loss_gradient = predicted - actual;
        loss_gradient = q_out.compute_gradient(loss_gradient);
        loss_gradient = fc3.compute_gradient(loss_gradient);
        loss_gradient = fc2.compute_gradient(loss_gradient);
        fc1.compute_gradient(loss_gradient);
    }
};
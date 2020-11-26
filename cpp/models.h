#pragma once

#include "layer.h"

#define OBSERVATION_DIM 1 // TODO
#define GOAL_DIM 1 // TODO
#define ACTION_DIM 1 // TODO
#define NEURONS 256

class Actor {
private:
    layer fc1 = layer(OBSERVATION_DIM + GOAL_DIM, NEURONS, RELU);
    layer fc2 = layer(NEURONS, NEURONS, RELU);
    layer fc3 = layer(NEURONS, NEURONS, RELU);
    layer action_out = layer(NEURONS, ACTION_DIM, TANH);

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
    layer fc1 = layer(OBSERVATION_DIM + GOAL_DIM + ACTION_DIM, NEURONS, RELU);
    layer fc2 = layer(NEURONS, NEURONS, RELU);
    layer fc3 = layer(NEURONS, NEURONS, RELU);
    layer q_out = layer(NEURONS, 1, NONE);
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
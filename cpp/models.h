#pragma once

#include "linear.h"

#define OBSERVATION_DIM 1 // TODO
#define GOAL_DIM 1 // TODO
#define ACTION_DIM 1 // TODO
#define NEURONS 10 // TODO 256

class Actor {
private:
    linear fc1 = linear(OBSERVATION_DIM + GOAL_DIM, NEURONS);
    linear fc2 = linear(NEURONS, NEURONS);
    linear fc3 = linear(NEURONS, NEURONS);
    linear action_out = linear(NEURONS, ACTION_DIM);

    Matrix *max_action;

public:
    Actor(Matrix *max_action) {
        this->max_action = max_action;
    };

    Matrix* forward(Matrix *x) {
        Matrix* m1 = fc1.multiply(x);
        m1->relu();
        Matrix* m2 = fc2.multiply(m1);
        m2->relu();
        Matrix* m3 = fc3.multiply(m2);
        m3->relu();
        Matrix* m4 = action_out.multiply(m3);
        m4->tanh();
        Matrix* out = (*max_action) * (*m4);
        delete m1, m2, m3, m4;
        return out;
    }
};


class Critic {
private:
    linear fc1 = linear(OBSERVATION_DIM + GOAL_DIM + ACTION_DIM, NEURONS);
    linear fc2 = linear(NEURONS, NEURONS);
    linear fc3 = linear(NEURONS, NEURONS);
    linear q_out = linear(NEURONS, 1);
    Matrix *max_action;

public:
    Critic(Matrix *max_action) {
        this->max_action = max_action;
    };

    Matrix* forward(Matrix *x, Matrix *actions) {
        Matrix *adjusted_actions = *actions / *max_action;
        Matrix *in = Matrix::vector_concat(x, adjusted_actions);
        Matrix* m1 = fc1.multiply(in);
        m1->relu();
        Matrix* m2 = fc2.multiply(m1);
        m2->relu();
        Matrix* m3 = fc3.multiply(m2);
        m3->relu();
        Matrix* m4 = q_out.multiply(m3);
        delete m1, m2, m3, adjusted_actions, in;
        return m4;
    }
};
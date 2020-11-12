#pragma once

#include "linear.h"

#define OBSERVATION_DIM 1 // TODO
#define GOAL_DIM 1 // TODO
#define ACTION_DIM 1 // TODO
#define NEURONS 10 // TODO 256

// TODO add critic
class Actor {
private:
    linear fc1 = linear(OBSERVATION_DIM + GOAL_DIM, NEURONS);
    linear fc2 = linear(NEURONS, NEURONS);
    linear fc3 = linear(NEURONS, NEURONS);
    linear action_out = linear(NEURONS, ACTION_DIM);
public:
    Actor() = default;

    Matrix* forward(Matrix *x) {
        Matrix* out = fc1.multiply(x)->relu();
        out = fc2.multiply(out)->relu();
        out = fc3.multiply(out)->relu();
        // TODO add tanh
        out = action_out.multiply(out);
        return out;
    }
};

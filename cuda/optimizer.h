#pragma once

class Optimizer {
	public:
		__host__ __device__ virtual void step() = 0;

		__host__ __device__ virtual ~Optimizer() {
		};
};

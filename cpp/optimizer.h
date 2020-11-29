#pragma once

class Optimizer {
	public:
		virtual void step() = 0;

		virtual ~Optimizer() {
		};
};
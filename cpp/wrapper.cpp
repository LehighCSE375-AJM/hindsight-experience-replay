#include "adam.h"

using namespace std;

class adam_wrapper
{
public:
	Adam* a;

	adam_wrapper(vector<Tensor *> params, double lr)
	{
		a = new Adam(params, lr);
	}

	~adam_wrapper() = default;
};

extern "C"
{
	adam_wrapper* init_adam(vector<Tensor *> params, double lr)
	{

	}
}
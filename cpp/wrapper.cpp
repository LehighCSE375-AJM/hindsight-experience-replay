// g++ -shared -Wl,-soname,wrapper -o wrapper.so -fPIC wrapper.cpp -I /usr/include/x86_64-linux-gnu -lopenblas

#include "adam.h"
#include "models.h"

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

class actor_wrapper
{
public:
	Actor* actor;
	vector<Tensor *> parameters;
	Tensor* forward_result;

	actor_wrapper(int obs, int goal, int action, double action_max)
	{
		Tensor tmp(1, 1, &action_max);
		actor = new Actor(obs, goal, action, tmp);
	}

	// copy constructor
	actor_wrapper(actor_wrapper* from)
	{
		actor = new Actor(*(from->actor));
	}

	void test()
	{
		cout << *forward_result << endl;
	}
};


extern "C"
{
	actor_wrapper* init_actor(int obs, int goal, int action, double action_max)
	{
		return new actor_wrapper(obs, goal, action, action_max);
	}

	actor_wrapper* copy_actor(actor_wrapper* from)
	{
		return new actor_wrapper(from);
	}

	vector<Tensor *> actor_parameters(actor_wrapper* a)
	{
		a->parameters = a->actor->parameters();
		return a->parameters;
	}

	void actor_forward(actor_wrapper* a, double* input, int height, int width)
	{
		// cout << "Length: " << len << endl;
		Tensor input_tensor(height, width, input);
		// cout << "Input:" << endl;
		// cout << input_tensor << endl;
		a->forward_result = &(a->actor->forward(input_tensor));
		// cout << "Result:" << endl;
		// cout << *(a->forward_result) << endl;
		// cout << tmp << endl;
		// a->test();
	}

	void get_actor_forward(actor_wrapper* a, double* out)
	{
		double* tmp = a->forward_result->get_values();
		for (int i = 0; i < a->forward_result->get_size(); i++)
		{
			out[i] = tmp[i];
		}
	}

	void test(actor_wrapper* a)
	{
		cout << *(a->forward_result) << endl;
	}
	
	adam_wrapper* init_adam_from_actor(actor_wrapper* a, double lr)
	{
		return new adam_wrapper(a->parameters, lr);
	}
}

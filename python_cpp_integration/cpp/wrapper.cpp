// g++ -shared -Wl,-soname,wrapper -o wrapper.so -fPIC wrapper.cpp -I /usr/include/x86_64-linux-gnu -lopenblas

#include "adam.h"
#include "models.h"

using namespace std;

class adam_wrapper
{
public:
	// Adam* a;
	GradientDescent* a;

	adam_wrapper(vector<Tensor *> params, double lr)
	{
		// a = new Adam(params, lr);
		a = new GradientDescent(params, lr);
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

class critic_wrapper
{
public:
	Critic* critic;
	vector<Tensor *> parameters;
	Tensor* forward_result;

	critic_wrapper(int obs, int goal, int action, double action_max)
	{
		Tensor tmp(1, 1, &action_max);
		critic = new Critic(obs, goal, action, tmp);
	}

	// copy constructor
	critic_wrapper(critic_wrapper* from)
	{
		critic = new Critic(*(from->critic));
	}
};


extern "C"
{
	////////// ACTOR:
	
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

	void actor_forward(actor_wrapper* a, int* out_dim, double* input,
						int height, int width)
	{
		Tensor input_tensor(height, width, input);
		a->forward_result = &(a->actor->forward(input_tensor));
		out_dim[0] = a->forward_result->get_height();
		out_dim[1] = a->forward_result->get_width();
	}

	void get_actor_forward(actor_wrapper* a, double* out)
	{
		double* tmp = a->forward_result->get_values();
		for (int i = 0; i < a->forward_result->get_size(); i++)
		{
			out[i] = tmp[i];
		}
	}

	void actor_soft_update(actor_wrapper* target, actor_wrapper* source,
							double polyak)
	{
		target->actor->copy_transform_params(source->actor,
			[&](Tensor* t, Tensor* s)
			{
				if (s->get_size() < 100) cout << *s;
				if (t->get_size() < 100) cout << *t;
				Tensor tmp = *s * (1 - polyak) + *t * polyak;
				if (tmp.get_size() < 100) cout << tmp;
				t->copy((tmp));
				if (t->get_size() < 100) cout << *t << endl;
			});
	}

	void test(actor_wrapper* a)
	{
		cout << *(a->forward_result) << endl;
	}

	////////// ADAM:
	
	adam_wrapper* init_adam_from_actor(actor_wrapper* a, double lr)
	{
		return new adam_wrapper(a->actor->params, lr);
	}

	void actor_adam_step(adam_wrapper* a)
	{
		a->a->step();
	}

	////////// CRITIC:

	critic_wrapper* init_critic(int obs, int goal, int action, double action_max)
	{
		return new critic_wrapper(obs, goal, action, action_max);
	}

	critic_wrapper* copy_critic(critic_wrapper* from)
	{
		return new critic_wrapper(from);
	}

	vector<Tensor *> critic_parameters(critic_wrapper* c)
	{
		c->parameters = c->critic->parameters();
		return c->parameters;
	}

	void critic_forward(critic_wrapper* c, int* out_dim,
						double* input, int i_height, int i_width,
						double* actions, int a_height, int a_width)
	{
		Tensor input_tensor(i_height, i_width, input);
		Tensor actions_tensor(a_height, a_width, actions);

		c->forward_result = &(c->critic->forward(input_tensor,
													actions_tensor));
		out_dim[0] = c->forward_result->get_height();
		out_dim[1] = c->forward_result->get_width();
	}

	void get_critic_forward(critic_wrapper* a, double* out)
	{
		double* tmp = a->forward_result->get_values();
		for (int i = 0; i < a->forward_result->get_size(); i++)
		{
			out[i] = tmp[i];
		}
	}

	void critic_backprop(critic_wrapper* c, double actual, double predicted)
	{
		c->critic->backprop(actual, Tensor(1, 1, &predicted)); //Tensor(1, 1, &actual)
	}

	void critic_soft_update(critic_wrapper* target, critic_wrapper* source,
							double polyak)
	{
		critic_parameters(target);
		target->critic->copy_transform_params(source->critic, target->parameters,
			[&](Tensor* t, Tensor* s)
			{
				if (s->get_size() < 100) cout << "s:" << *s;
				if (t->get_size() < 100) cout << "t:" << *t;
				Tensor tmp = *s * (1 - polyak) + *t * polyak;
				if (tmp.get_size() < 100) cout << "tmp:" << tmp;
				// t->copy((tmp));
				tmp.copy(*t);
				if (t->get_size() < 100) cout << "t:" << *t << endl;
			});
	}
}

import ctypes

libc = ctypes.cdll.LoadLibrary("cpp/wrapper.so")

class actor(object):
	def __init__(self, env_params=None, other_obj=None):
		if env_params:
			obs = env_params['obs']
			goal = env_params['goal']
			action = env_params['action']
			action_max = env_params['action_max']
			self.max_timesteps = env_params['max_timesteps']

			libc.init_actor.argtypes = [ctypes.c_int, ctypes.c_int,
										ctypes.c_int, ctypes.c_double]
			self.obj = libc.init_actor(obs, goal, action, action_max)
		elif other_obj:
			self.obj = other_obj
		else: raise Exception("No arguments provided to actor constructor")

	def copy(self):
		copy = libc.copy_actor(self.obj)
		return actor(other_obj=copy)
	
	def parameters(self):
		return libc.actor_parameters(self.obj)



class critic(object):
	def __init__(self, env_params):
		print("hi")

import ctypes, torch

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

	def forward(self, input_tensor):
		height = list(input_tensor.shape)[0]
		width = list(input_tensor.shape)[1]
		print(f"Height: {height}, Width: {width}")
		tmp = torch.flatten(input_tensor).tolist()
		# print(tmp)
		input_list = (ctypes.c_double * len(tmp))(*tmp)
		# if len(tmp) != 13: print(f"Input len: {len(tmp)}")
		libc.actor_forward(self.obj, input_list, height, width)

		out = (ctypes.c_double * len(input_list))()
		libc.get_actor_forward(self.obj, ctypes.byref(out))
		out = list(out)
		for i, x in enumerate(reversed(out)):
			if x:
				out = out[:-1*i]
				break
		if len(out) != 4: print(f"Forward len: {len(out)}")
		return torch.tensor(out)




class critic(object):
	def __init__(self, env_params):
		print("hi")

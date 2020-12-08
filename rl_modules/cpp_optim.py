import ctypes

libc = ctypes.cdll.LoadLibrary("cpp/wrapper.so")

class Adam(object):	
	def __init__(self, params=None, actor=None, lr=0.001):
		if params:
			self.obj = libc.init_adam(params, lr)
		elif actor:
			lr = ctypes.c_double(lr)
			self.obj = libc.init_adam_from_actor(actor.obj, lr)
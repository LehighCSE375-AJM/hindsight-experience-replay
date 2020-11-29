import ctypes

libc = ctypes.cdll.LoadLibrary("../cpp/wrapper.so")

class actor(object):
	def __init__(self, env_params):
		print("hi")


class critic(object):
	def __init__(self, env_params):
		print("hi")

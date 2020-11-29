import ctypes

libc = ctypes.cdll.LoadLibrary("../cpp/wrapper.so")

class Adam(object):
	def __init__(self, env_params):
		print("hi")
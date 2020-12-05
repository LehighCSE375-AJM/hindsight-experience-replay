import ctypes

libc = ctypes.cdll.LoadLibrary("../cpp/wrapper.so")

class Adam(object):
	def __init__(self, params, lr=0.001):
		print(params)
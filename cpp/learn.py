import ctypes

libc = ctypes.cdll.LoadLibrary("./learn_wrapper.so")

class wrapper_class(object):
	def __init__(self):
		# a = [1, 2, 3, 4]
		# b = (ctypes.c_double * len(a))(*a)
		self.obj = libc.setup()

	def run(self):
		return libc.run(self.obj)


if __name__ == '__main__':
	w = wrapper_class()
	print(w.run())

	# print(libc.run_all())
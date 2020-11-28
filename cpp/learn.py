from ctypes import *

libc = cdll.LoadLibrary("./wrapper.so")


class Matrix(object):
	def __init__(self, array):
		libc.new_matrix.argtypes = [c_double * len(array)]
		self.obj = libc.new_matrix((c_double * len(array))(*array))

	def multiply(self, val):
		# libc.new_matrix.argtypes = [POINTER, c_double]
		libc.multiply(self.obj, val)

	def print_matrix(self):
		libc.print_matrix(self.obj)


if __name__ == '__main__':
	array = [1, 2, 3, 4]
	# m = Matrix(a1) 
	# m.multiply(1.1)
	# m.print_matrix()

	# b = (ctypes.c_double * 4)()
	# b[0] = 1
	# b[1] = 2
	# b[2] = 3
	# b[3] = 4
	# b = (ctypes.c_double * len(array))(*array)

	libc.test_matrix.argtypes = [c_double * len(array)]
	libc.test_matrix((c_double * len(array))(*array))
import ctypes

libc = ctypes.cdll.LoadLibrary("./test_class.so")

class test_class(object):
	def __init__(self):
		# libc.test_set.argtypes = [ctypes.c_int]
		a = [1, 2, 3, 4]
		b = (ctypes.c_double * len(a))(*a)
		self.obj = libc.test_set(5, b, len(a))

	def get(self):
		# libc.test_get.argtypes = [ctypes.POINTER]
		# libc.test_get.restype = ctypes.c_int
		return libc.test_get(self.obj)

	def test_print(self):
		# libc.test_get.argtypes = [ctypes.POINTER]
		# libc.test_get.restype = ctypes.c_int
		return libc.test_print(self.obj)

# libc.test_set.argtypes = [ctypes.c_int]
# libc.test_set.restype = ctypes.POINTER
# ctypes.POINTER(libc.test_set(ctypes.c_int(5)))

if __name__ == '__main__':
	t = test_class()
	a = t.get()
	print(a)
	t.test_print()
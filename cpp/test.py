import ctypes

libc = ctypes.cdll.LoadLibrary("./libc.so")
libc.main()
libc.notmain()

libc.cmult.argtypes = [ctypes.c_float, ctypes.c_float]
libc.cmult.restype = ctypes.c_float
print(libc.cmult(2, 1.8))

a = [1, 2, 3]
libc.add_all.argtypes = [ctypes.c_float * len(a), ctypes.c_size_t]
libc.add_all.restype = ctypes.c_float
print(libc.add_all((ctypes.c_float * len(a))(*a), len(a)))

# libc.mult_scalar.argtypes = [ctypes.c_double * len(a), ctypes.c_size_t, ctypes.c_double]
# libc.mult_scalar.restype = ctypes.c_double * len(a) #ctypes.POINTER(ctypes.c_double)
# result = libc.mult_scalar((ctypes.c_double * len(a))(*a), len(a), 1)
# print(a)
# print(list(result))


# libc.mult_scalar.argtypes = [ctypes.POINTER(ctypes.c_double * len(a)), ctypes.c_size_t, ctypes.c_double]
# libc.mult_scalar.restype = ctypes.POINTER(ctypes.c_double)
# result = libc.mult_scalar(ctypes.byref((ctypes.c_double * len(a))(*a)), len(a), 1)
# print(a)
# for i in range(3):
# 	print(result[i])

# b = (ctypes.c_double * 3)()
# b[0] = 1
# b[1] = 2
# b[2] = 3
b = (ctypes.c_double * len(a))(*a)

print(list(b))
libc.ref.argtypes = [ctypes.POINTER(ctypes.c_double * len(a))]
libc.ref(ctypes.byref(b))
print(list(b))

libc.mult_scalar.argtypes = [ctypes.POINTER(ctypes.c_double * len(a)), ctypes.c_size_t, ctypes.c_double]
libc.mult_scalar(ctypes.byref(b), len(a), 2)
print(list(b))
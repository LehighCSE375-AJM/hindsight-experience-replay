program_NAME := learn

SRCS = learn.cpp
CU_SRCS = test.cu

ALL_HDRS = $(wildcard *.h)
CUDA_HDRS = $(wildcard cuda_*.h)
MAIN_HDRS := $(filter-out $(CUDA_HDRS), $(ALL_HDRS))

OBJS := ${SRCS:.c=.o}

CFLAGS = -O3 -g -lopenblas

program_INCLUDE_DIRS := ${OPENBLAS_DIR}

CFLAGS += $(if $(program_INCLUDE_DIRS),-I $(program_INCLUDE_DIRS),)

CC=g++
CU=nvcc

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(OBJS) $(MAIN_HDRS)
	$(CC) $(OBJS) -o $(program_NAME) $(CFLAGS)

cuda_$(program_NAME): $(CU_SRCS) $(ALL_HDRS)
	$(CU) $(CU_SRCS) -o cuda_$(program_NAME) $(CFLAGS)

clean: 
	rm $(program_NAME)
	rm cuda_$(program_NAME)

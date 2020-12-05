program_NAME := learn

SRCS = learn.cpp
CU_SRCS = test.cu

HDRS = $(wildcard *.h)
OBJS := ${SRCS:.c=.o}

CFLAGS = -O3 -g -lopenblas

program_INCLUDE_DIRS := ${OPENBLAS_DIR}

CFLAGS += $(if $(program_INCLUDE_DIRS),-I $(program_INCLUDE_DIRS),)

CC=g++
CU=nvcc

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(OBJS) $(HDRS)
	$(CC) $(OBJS) -o $(program_NAME) $(CFLAGS)

cuda: $(CU_SRCS) $(HDRS)
	$(CU) $(CU_SRCS) -o cuda_$(program_NAME) $(CFLAGS)

clean: 
	rm $(program_NAME)
	rm cuda_$(program_NAME)

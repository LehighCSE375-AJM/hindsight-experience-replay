program_NAME := a.out

SRCS = learn.cpp
HDRS = $(wildcard *.h)
OBJS := ${SRCS:.c=.o}

CFLAGS = -O3 -g -lopenblas

program_INCLUDE_DIRS := ${OPENBLAS_DIR}

CFLAGS += -I $(program_INCLUDE_DIRS)

CC=g++

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(OBJS) $(HDRS)
	$(CC) $(OBJS) -o $(program_NAME) $(CFLAGS)

clean: 
	rm a.out

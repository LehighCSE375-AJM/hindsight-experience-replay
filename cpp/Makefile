program_NAME := a.out

SRCS = learn.cpp
HDRS = $(wildcard *.h)
OBJS := ${SRCS:.c=.o}

CFLAGS += -m64 -O3 -g

program_INCLUDE_DIRS := ${OPENBLAS_DIR}

CPPFLAGS += -I $(program_INCLUDE_DIRS)

CC=g++-10

LINK.c := $(CC) $(CFLAGS) $(CPPFLAGS)

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(OBJS) $(HDRS)
	$(LINK.c) $(OBJS) -o $(program_NAME)

clean: 
	rm a.out
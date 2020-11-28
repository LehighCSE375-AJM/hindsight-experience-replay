program_NAME := a.out

SRCS = learn.cpp
HDRS = linear.h models.h matrix.h layer.h adam.h
OBJS := ${SRCS:.c=.o}

CFLAGS += -m64 -O3 -g

program_INCLUDE_DIRS := ${OPENBLAS_DIR}
#program_LIBRARY_DIRS :=
#program_LIBRARIES :=

CPPFLAGS += -I $(program_INCLUDE_DIRS)
#LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
#LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))

# TODO need to set up my computer so just g++ works. 
CC=g++-10

LINK.c := $(CC) $(CFLAGS) $(CPPFLAGS)

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(OBJS) $(HDRS)
	$(LINK.c) $(OBJS) -o $(program_NAME)

clean: 
	rm a.out
program_NAME := a.out

SRCS = learn.cc
HDRS = linear.h models.h matrix.h
OBJS := ${SRCS:.c=.o}

CFLAGS += -m64 -O3

program_INCLUDE_DIRS := /opt/openblas/include /opt/openblas/lib/libopenblas.dylib
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
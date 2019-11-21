CC=gcc
CXX=g++
CPPFLAGS=$(shell pkg-config --cflags eigen3) -O3
LDLIBS=$(shell pkg-config --libs eigen3)

SRCS=$(shell echo *.cpp)
OBJS=$(SRCS:.cpp=.out)

$(OBJS): $(SRCS)
		$(CXX) -o $(OBJS) $(SRCS) $(CPPFLAGS) $(LDLIBS)

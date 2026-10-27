CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
all: bin/engine
bin/engine: cpp/engine.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f bin/engine

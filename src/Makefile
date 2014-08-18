# Makefile
#
# Copyright (c) [2012-], Josef Robert Novak
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
#  modification, are permitted #provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above 
#    copyright notice, this list of #conditions and the following 
#    disclaimer in the documentation and/or other materials provided 
#    with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
# OF THE POSSIBILITY OF SUCH DAMAGE.
CC=g++

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
LIBS=-lfst -std=c++0x -ldl -lm
endif

ifeq ($(UNAME), Linux)
LIBS=-lfst -ldl -lm -lrt
CC=g++ -std=c++0x -fPIC -Wall -Wno-sign-compare
endif

OUT=$(OBJS)

CFLAGS ?= -O2     
THIRD_PARTIES_INCLUDE ?= -I3rdparty/utfcpp -I3rdparty/rnnlm

EXTRA=$(CFLAGS) $(THIRD_PARTIES_INCLUDE) $(CXXFLAGS) $(CPPFLAGS) #-fopenmp

OBJS=rnnlmlib.o

%.o: %.cc
	$(CC) -std=c++0x -lfst -ldl $(THIRD_PARTIES_INCLUDE) $(CXXFLAGS) $(CPPFLAGS) -c $(<) -o $(@)

all: phonetisaurus-g2prnn

phonetisaurus-g2prnn: rnnlmlib.o phonetisaurus-g2prnn.cc
	$(CC) -O2 -lfst -lm -std=c++0x -I3rdparty/rnnlm -I3rdparty/utfcpp \
	-funroll-loops -I. -ffast-math rnnlmlib.o phonetisaurus-g2prnn.cc -o phonetisaurus-g2prnn

rnnlmlib.o: 3rdparty/rnnlm/rnnlmlib.cpp
	$(CC) $(EXTRA) -std=c++0x -fPIC -I. -lfst -lm -ldl \
	-funroll-loops -ffast-math -lm -w -c 3rdparty/rnnlm/rnnlmlib.cpp

install: all
	install -c phonetisaurus-g2prnn ../phonetisaurus-g2prnn
clean:
	$(RM) $(OUT) $(TMP) *.o phonetisaurus-g2prnn ../phonetisaurus-g2prnn

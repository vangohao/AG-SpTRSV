CUDAC?=/usr/local/cuda-11.8/bin/nvcc

CUDAARCHS?=80

FLAG=-O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_$(CUDAARCHS),code=sm_$(CUDAARCHS)

LFLAGS=-lSpTRSV -lutils -lcusparse

INCLUDE_DIR=../include
UTILS_DIR=../utils
LIB_DIR=../build

all: test_ag test_ag_search

test_ag: main_ag.cu
	$(CUDAC) $^ $(FLAG) -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

test_ag_search: main_search.cu
	$(CUDAC) $^ $(FLAG) -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

.PHONY:

clean:
	rm test_ag test_ag_search

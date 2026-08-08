# Standalone Makefile for the MACKO SpMV CUDA benchmark

HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS := -m$(shell getconf LONG_BIT)
CCFLAGS := -fPIC
LDFLAGS :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES := -I$(CUDA_PATH)/include/
LIBRARIES := -lcudart -lcublas -lcusparse

SMS ?= 86
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++17
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v

all: benchmark

benchmark: benchmark.cu utils.cuh kernels.cuh cpu_compressor.cuh
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ benchmark.cu $(LIBRARIES)

clean:
	rm -f benchmark

.PHONY: all clean

# Standalone Makefile for the DASP fp16 SpMV benchmark

HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS := -m$(shell getconf LONG_BIT)
CCFLAGS := -fPIC -fopenmp-simd
LDFLAGS :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

INCLUDES := -I$(CUDA_PATH)/include/
LIBRARIES := -lcudart -lcublas -lcusparse

SMS ?= 80
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++17
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += -lineinfo

all: spmv_half

spmv_half: main_f16.cu dasp_f16.h common.h utils.h mmio.h mmio_highlevel.h
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ main_f16.cu $(LIBRARIES)

clean:
	rm -f spmv_half

.PHONY: all clean

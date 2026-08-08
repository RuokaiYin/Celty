# Makefile for hybrid register/SMEM ablation on CUDA Cores

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS := -m$(shell getconf LONG_BIT)
CCFLAGS := -fPIC
LDFLAGS :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES := -I/usr/local/cuda/include/
LIBRARIES := -lcudart -lcublas

################################################################################

# Gencode arguments
SMS ?= 86
# SMS?= 80
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++11
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v

################################################################################

# Target rules
all: test_hybrid_ablation

test_hybrid_ablation: test_hybrid_ablation.cu hybrid_api.cuh hybrid_kernels.cuh
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ test_hybrid_ablation.cu $(LIBRARIES)

clean:
	rm -f test_hybrid_ablation

.PHONY: all clean

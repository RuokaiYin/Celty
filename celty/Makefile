# Makefile for GEMV on CUDA Cores

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
all: test_spmspv

# Build shared library
libGEMV_API.so: GEMV_API.o
	$(NVCC) --shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

GEMV_API.o: GEMV_API.cuh KERNELs.cuh
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

test_spmspv: test_spmspv.cu GEMV_API.cuh KERNELs.cuh
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ test_spmspv.cu $(LIBRARIES)

test_packing: test_packing.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIBRARIES)

clean:
	rm -f libGEMV_API.so GEMV_API.o test_spmspv

.PHONY: all clean

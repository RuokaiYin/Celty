#pragma once

#include "hybrid_kernels.cuh"
#include <cuda_fp16.h>
#include <cstdio>

#define HYBRID_WARP_SIZE 32
#define HYBRID_WARPS_PER_BLOCK 8
#define HYBRID_THREADS_PER_BLOCK (HYBRID_WARP_SIZE * HYBRID_WARPS_PER_BLOCK)
#define HYBRID_BLOCK_K_SIZE 64

template <int REG_COUNT>
cudaError_t SPVSPM_CUDACores_ColMajor_Hybrid_Template_API(
    cudaStream_t stream,
    const half* A_packed,
    const unsigned char* deltas,
    const int* col_indices,
    const half* x,
    float* y,
    int M,
    int K,
    float threshold)
{
    const int block_size_m = HYBRID_THREADS_PER_BLOCK;
    const int block_size_k = HYBRID_BLOCK_K_SIZE;
    const int num_blocks_m = (M + block_size_m - 1) / (block_size_m * LOAD_SIZE);
    const int num_blocks_k = (K + block_size_k - 1) / block_size_k;
    dim3 gridDim(num_blocks_m, num_blocks_k);
    dim3 blockDim(block_size_m, 1);

    const int shared_mem_bytes = (REG_COUNT < LOAD_SIZE) ? (M * sizeof(float)) : 0;
    if (shared_mem_bytes > 0) {
        cudaFuncSetAttribute(
            SPVSPM_Kernel_ColMajor_Hybrid<REG_COUNT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_bytes
        );
    }

    SPVSPM_Kernel_ColMajor_Hybrid<REG_COUNT><<<gridDim, blockDim, shared_mem_bytes, stream>>>(
        A_packed, deltas, col_indices, x, y, M, K, block_size_k, threshold);
    return cudaGetLastError();
}

cudaError_t SPVSPM_CUDACores_ColMajor_Hybrid_API(
    cudaStream_t stream,
    const half* A_packed,
    const unsigned char* deltas,
    const int* col_indices,
    const half* x,
    float* y,
    int M,
    int K,
    float threshold,
    int reg_count)
{
    switch (reg_count) {
        case 0:
            return SPVSPM_CUDACores_ColMajor_Hybrid_Template_API<0>(
                stream, A_packed, deltas, col_indices, x, y, M, K, threshold);
        case 2:
            return SPVSPM_CUDACores_ColMajor_Hybrid_Template_API<2>(
                stream, A_packed, deltas, col_indices, x, y, M, K, threshold);
        case 4:
            return SPVSPM_CUDACores_ColMajor_Hybrid_Template_API<4>(
                stream, A_packed, deltas, col_indices, x, y, M, K, threshold);
        case 6:
            return SPVSPM_CUDACores_ColMajor_Hybrid_Template_API<6>(
                stream, A_packed, deltas, col_indices, x, y, M, K, threshold);
        case 8:
            return SPVSPM_CUDACores_ColMajor_Hybrid_Template_API<8>(
                stream, A_packed, deltas, col_indices, x, y, M, K, threshold);
        default:
            fprintf(stderr, "Unsupported hybrid reg_count=%d. Use 0, 2, 4, 6, or 8.\n", reg_count);
            return cudaErrorInvalidValue;
    }
}

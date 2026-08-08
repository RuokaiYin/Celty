#include "KERNELs.cuh"
#include "cublas_v2.h"
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define BLOCK_K_SIZE 64 
// K = 64

//! API for DRAM-acc SPVSPM baseline
cudaError_t SPVSPM_CUDACores_ColMajor_API(
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
    const int block_size_m = THREADS_PER_BLOCK;
    const int block_size_k = BLOCK_K_SIZE;
    const int num_blocks_m = (M + block_size_m - 1) / (block_size_m*8);
    const int num_blocks_k = (K + block_size_k - 1) / block_size_k;
    dim3 gridDim(num_blocks_m, num_blocks_k);
    dim3 blockDim(block_size_m, 1);

    SPVSPM_Kernel_ColMajor<<<gridDim, blockDim, 0, stream>>>(A_packed, deltas, col_indices, x, y, M, K, block_size_k, threshold);
    return cudaGetLastError();
}


//! API for Celty-SMEM
cudaError_t Celty_SMEM_Kernel_API(
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
    const int block_size_m = THREADS_PER_BLOCK;
    const int block_size_k = BLOCK_K_SIZE;
    //! Each thread handles LOAD_SIZE=8 packed elements, so one block covers block_size_m * 8 packed positions
    const int num_blocks_m = (M + block_size_m - 1) / (block_size_m * 8);
    const int num_blocks_k = (K + block_size_k - 1) / block_size_k;
    dim3 gridDim(num_blocks_m, num_blocks_k);
    dim3 blockDim(block_size_m, 1);

    const int shared_mem_bytes = (block_size_k + 1) * sizeof(int) + M * sizeof(float);
    //! Must set this BEFORE launch if shared_mem_bytes > 48KB
    cudaFuncSetAttribute(
        Celty_SMEM_Kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_bytes
    );

    Celty_SMEM_Kernel<<<gridDim, blockDim, shared_mem_bytes, stream>>>(A_packed, deltas, col_indices, x, y, M, K, block_size_k, threshold);
    return cudaGetLastError();
}

//! API for Celty-SIMT
cudaError_t Celty_SIMT_Kernel_API(
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
    const int block_size_m = THREADS_PER_BLOCK;
    const int block_size_k = BLOCK_K_SIZE;
    //! Each thread handles LOAD_SIZE=8 packed elements, so one block covers block_size_m * 8 packed positions
    const int num_blocks_m = (M + block_size_m - 1) / (block_size_m * 8);
    const int num_blocks_k = (K + block_size_k - 1) / block_size_k;
    dim3 gridDim(num_blocks_m, num_blocks_k);
    dim3 blockDim(block_size_m, 1);

    const int shared_mem_bytes = (block_size_k + 1) * sizeof(int) + M * sizeof(float); 

    // //! Must set this BEFORE launch if shared_mem_bytes > 48KB
    cudaFuncSetAttribute(
        Celty_SIMT_Kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_bytes
    );

    //! Can set SMEM into 0 if no need the SMEM hybrid accumulation. Kept for conservative performance.
    Celty_SIMT_Kernel<<<gridDim, blockDim, shared_mem_bytes, stream>>>(A_packed, deltas, col_indices, x, y, M, K, block_size_k, threshold);

    return cudaGetLastError();
}


//!!! Cublas API calls.
#define HGEMV_CHECK_CUBLAS_ERROR(call)                        \
    do {                                                      \
        cublasStatus_t status = (call);                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",   \
                    __FILE__, __LINE__, status);               \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

//! Dense Cublas
cublasHandle_t getCublasTensorOpHandle() {
    cublasHandle_t handle = nullptr;
    HGEMV_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    HGEMV_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    return handle;
}

void cublasTensorOp(half *A, half *B, half *C, size_t N, size_t K, cudaStream_t stream) {
    static cublasHandle_t handle = getCublasTensorOpHandle();
    static size_t M = 1;
    static float alpha = 1.0;
    static float beta = 0.0;

    HGEMV_CHECK_CUBLAS_ERROR(cublasSetStream(handle, stream));
    HGEMV_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                          CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
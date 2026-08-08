#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define LOAD_SIZE 8

// ============================================================================
//? Below are two helper functions
// ============================================================================

//! The real one for Celty-SMEM
template <typename T>
__inline__ __device__ T warp_exclusive_sum(T val)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    T tmp = val;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
    {   
        T y = __shfl_up_sync(FULL_MASK, tmp, offset);
        if (lane >= offset)
            tmp += y;
    }
    return tmp - val;
}

//! The dummy one for Celty-SIMT
template <typename T>
__inline__ __device__ T warp_exclusive_sum_external(T val)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    T tmp = val;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
    {   
        T y = 0;
        if (lane >= offset)
            tmp += y;
    }
    return tmp - val;
}

// ============================================================================
// Column-major SPVSPM with K blocking and atomic accumulation.
// The performance with global memory write.
//! DRAM-accumulation baseline
// ============================================================================
__global__ void
SPVSPM_Kernel_ColMajor(const half* __restrict__ A_packed,
                        const unsigned char* __restrict__ deltas,
                        const int* __restrict__ col_indices,
                        const half* __restrict__ x,
                        float* __restrict__ y,
                        int M,
                        int K,
                        int block_k,
                        float threshold)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; //! global thread id
    int row = thread_id * LOAD_SIZE;
    if (row >= M) {
        return;
    }

    int k_start = blockIdx.y * block_k;
    int k_end = k_start + block_k;
    if (k_end > K) {
        k_end = K;
    }

    for (int k = k_start; k < k_end; k++) {
        float x_val = __half2float(x[k]);
        if (fabsf(x_val) < threshold) {
            continue;
        }

        int col_begin = col_indices[k];       //! start of packed data for column k
        int col_end   = col_indices[k + 1];   //! end of packed data for column k
        if (col_begin >= col_end) continue;    //! empty column, skip

        //! Align to LOAD_SIZE boundary
        //! col_begin may not be 8-aligned because columns share a flat packed array
        int col_begin_aligned = col_begin - (col_begin % LOAD_SIZE);

        //! This thread's starting position in the flat packed array
        int thread_ordinal = col_begin_aligned + row;

        //! Vectorized loads
        //! float4 = 16 bytes = 8 half values
        //! unsigned int = 4 bytes = 8 x 4-bit deltas
        const float4 *A_vec = reinterpret_cast<const float4 *>(A_packed);
        const unsigned int *D_vec = reinterpret_cast<const unsigned int *>(deltas);

        float4 a_raw = {0, 0, 0, 0};
        unsigned int d_raw = 0;

        if (thread_ordinal < col_end) {  //! guard before loading
            a_raw = A_vec[thread_ordinal / LOAD_SIZE];
            d_raw = D_vec[thread_ordinal / LOAD_SIZE];
        }
        __half *a_vals = reinterpret_cast<__half *>(&a_raw);

        //! Zero out elements that belong to previous column
        //! When col_begin is not 8-aligned, the first thread's load straddles the boundary
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i < col_begin) {
                unsigned int delta_mask = ~(15u << (4 * i)); //! clear this delta's 4 bits
                d_raw &= delta_mask;
                a_vals[i] = __float2half(0.0f);              //! zero out the value
            }
        }
        int local_delta_sum = 0;
        int sum_deltas_in_col = 0;
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++){
            local_delta_sum += int((d_raw >> (4 * i)) & 15);
        }
        sum_deltas_in_col += warp_exclusive_sum(local_delta_sum);

        //! Write and read from/to global memory directly 
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i >= col_begin && thread_ordinal + i < col_end) {
                float a_val = __half2float(a_vals[i]);
                int real_delta = sum_deltas_in_col + thread_ordinal + i - col_begin;
                y[real_delta] = __fmaf_rn(a_val, x_val, y[real_delta]);
            }
        }
    }
}


// ============================================================================
//! SMEM-based accumulation kernel for Celty-SMEM.
// ============================================================================
__global__ void
Celty_SMEM_Kernel(const half* __restrict__ A_packed,
                           const unsigned char* __restrict__ deltas,
                           const int* __restrict__ col_indices,
                           const half* __restrict__ x,
                           float* __restrict__ y,
                           int M,
                           int K,
                           int block_k,
                           float threshold)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; //! global thread id
    int row = thread_id * LOAD_SIZE;
    if (row >= M) {
        return;
    }

    int k_start = blockIdx.y * block_k;
    int k_end = k_start + block_k;
    if (k_end > K) {
        k_end = K;
    }

    extern __shared__ float smem_out[];

    // Initialize shared memory to 0 (cooperatively across all threads)
    for (int idx = threadIdx.x; idx < M; idx += blockDim.x) {
        smem_out[idx] = 0.0f;
    }
    __syncthreads();

    for (int k = k_start; k < k_end; k++) {
        float x_val = __half2float(x[k]);
        if (fabsf(x_val) < threshold) {
            continue;
        }

        int col_begin = col_indices[k];       //! start of packed data for column k
        int col_end   = col_indices[k + 1];   //! end of packed data for column k
        if (col_begin >= col_end) continue;    //! empty column, skip

        //! Align to LOAD_SIZE boundary
        //! col_begin may not be 8-aligned because columns share a flat packed array
        int col_begin_aligned = col_begin - (col_begin % LOAD_SIZE);

        //! This thread's starting position in the flat packed array
        int thread_ordinal = col_begin_aligned + row;


        //! Vectorized loads
        //! float4 = 16 bytes = 8 half values
        //! unsigned int = 4 bytes = 8 x 4-bit deltas
        const float4 *A_vec = reinterpret_cast<const float4 *>(A_packed);
        const unsigned int *D_vec = reinterpret_cast<const unsigned int *>(deltas);

        float4 a_raw = {0, 0, 0, 0};
        unsigned int d_raw = 0;

        if (thread_ordinal < col_end) {  //! guard before loading
            a_raw = A_vec[thread_ordinal / LOAD_SIZE];
            d_raw = D_vec[thread_ordinal / LOAD_SIZE];
        }
        __half *a_vals = reinterpret_cast<__half *>(&a_raw);

        //! Zero out elements that belong to previous column
        //! When col_begin is not 8-aligned, the first thread's load straddles the boundary
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i < col_begin) {
                unsigned int delta_mask = ~(15u << (4 * i)); //! clear this delta's 4 bits
                d_raw &= delta_mask;
                a_vals[i] = __float2half(0.0f);              //! zero out the value
            }
        }
        
        int local_delta_sum = 0;
        int sum_deltas_in_col = 0;
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++){
            local_delta_sum += int((d_raw >> (4 * i)) & 15);
        }
        sum_deltas_in_col += warp_exclusive_sum(local_delta_sum);

        //! Multiply all valid packed values by x_val
        //! (delta-to-row position computation will be added later)
#pragma unroll
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i >= col_begin && thread_ordinal + i < col_end) {
                float a_val = __half2float(a_vals[i]);
                int real_delta = sum_deltas_in_col + thread_ordinal + i - col_begin;
                smem_out[real_delta] =  __fmaf_rn(a_val, x_val, smem_out[real_delta]);
            }
        }
    }
    __syncthreads();

    //! Flush out accumulated results into global memory
    float4* smem_out4 = (float4*)smem_out;
    for (int idx = threadIdx.x; idx < M / 4; idx += blockDim.x) {
        float4 val = smem_out4[idx];
        atomicAdd(&y[idx * 4 + 0], val.x);
        atomicAdd(&y[idx * 4 + 1], val.y);
        atomicAdd(&y[idx * 4 + 2], val.z);
        atomicAdd(&y[idx * 4 + 3], val.w);
    }
}

// ============================================================================
//! Register accumulation kernel for Celty, simulating the performance of Celty-SIMT.
// ============================================================================
__global__ void
Celty_SIMT_Kernel(const half* __restrict__ A_packed,
                                const unsigned char* __restrict__ deltas,
                                const int* __restrict__ col_indices,
                                const half* __restrict__ x,
                                float* __restrict__ y,
                                int M,
                                int K,
                                int block_k,
                                float threshold)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; //! global thread id
    int row = thread_id * LOAD_SIZE;
    if (row >= M) {
        return;
    }

    int k_start = blockIdx.y * block_k;
    int k_end = k_start + block_k;
    if (k_end > K) {
        k_end = K;
    }

    float sum[LOAD_SIZE] = {0.0f};
    float indices[LOAD_SIZE] = {0.0f};
    extern __shared__ float smem_out[]; //! Can turn this off if dont need the hybrid SMEM accumulation. Kept for conservative performance.

    // Initialize shared memory to 0 (cooperatively across all threads)
    for (int idx = threadIdx.x; idx < M; idx += blockDim.x) {
        smem_out[idx] = 0.0f;
    }
    __syncthreads();

    for (int k = k_start; k < k_end; k++) {
        float x_val = __half2float(x[k]);
        if (fabsf(x_val) < threshold) {
            continue;
        }

        int col_begin = col_indices[k];       //! start of packed data for column k
        int col_end   = col_indices[k + 1];   //! end of packed data for column k
        if (col_begin >= col_end) continue;    //! empty column, skip

        //! Align to LOAD_SIZE boundary
        //! col_begin may not be 8-aligned because columns share a flat packed array
        int col_begin_aligned = col_begin - (col_begin % LOAD_SIZE);

        //! This thread's starting position in the flat packed array
        int thread_ordinal = col_begin_aligned + row;


        //! Vectorized loads
        //! float4 = 16 bytes = 8 half values
        //! unsigned int = 4 bytes = 8 x 4-bit deltas
        const float4 *A_vec = reinterpret_cast<const float4 *>(A_packed);
        const unsigned int *D_vec = reinterpret_cast<const unsigned int *>(deltas);

        float4 a_raw = {0, 0, 0, 0};
        unsigned int d_raw = 0;

        if (thread_ordinal < col_end) {  //! guard before loading
            a_raw = A_vec[thread_ordinal / LOAD_SIZE];
            d_raw = D_vec[thread_ordinal / LOAD_SIZE];
        }
        __half *a_vals = reinterpret_cast<__half *>(&a_raw);

        //! Zero out elements that belong to previous column
        //! When col_begin is not 8-aligned, the first thread's load straddles the boundary
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i < col_begin) {
                unsigned int delta_mask = ~(15u << (4 * i)); //! clear this delta's 4 bits
                d_raw &= delta_mask;
                a_vals[i] = __float2half(0.0f);              //! zero out the value
            }
        }

        int local_delta_sum = 0;
        int sum_deltas_in_col = 0;
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++){
            local_delta_sum += int((d_raw >> (4 * i)) & 15);
        }
        // sum_deltas_in_col += warp_exclusive_sum(local_delta_sum);
        sum_deltas_in_col += warp_exclusive_sum_external(local_delta_sum);

        

        //! Multiply all valid packed values by x_val
        //! (delta-to-row position computation will be added later)
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i >= col_begin && thread_ordinal + i < col_end) {
                float a_val = __half2float(a_vals[i]);
                int real_delta = sum_deltas_in_col + thread_ordinal + i - col_begin;

                indices[i] += real_delta;
                sum[i] = __fmaf_rn(a_val, indices[i], sum[i]);
                // asm volatile("" : "+f"(indices[i]));
            }
        }
    }

#pragma unroll 8
    for (int i = 0; i < LOAD_SIZE; i++) {
        y[row + i] = sum[i]; //! The kernel can retire immediately after writeback to the registers, Celty unit will handle the rest (aka., no need for explicit atomicAdd).
    }
}
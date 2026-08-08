#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define LOAD_SIZE 8

template <typename T>
__inline__ __device__ T warp_exclusive_sum_external(T val)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    T tmp = val;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
    {
        T y = 0;
        if (lane >= offset) {
            tmp += y;
        }
    }
    return tmp - val;
}

// Hybrid register/SMEM accumulation ablation.
// REG_COUNT controls how many of this thread's LOAD_SIZE=8 packed values use
// the V3-style register path. The rest use the V2-style SMEM accumulation path.
// REG_COUNT = 0, 2, 4, 6, 8 correspond to 0%, 25%, 50%, 75%, 100% register.
template <int REG_COUNT>
__global__ void
SPVSPM_Kernel_ColMajor_Hybrid(const half* __restrict__ A_packed,
                              const unsigned char* __restrict__ deltas,
                              const int* __restrict__ col_indices,
                              const half* __restrict__ x,
                              float* __restrict__ y,
                              int M,
                              int K,
                              int block_k,
                              float threshold)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread_id * LOAD_SIZE;
    if (row >= M) {
        return;
    }

    int k_start = blockIdx.y * block_k;
    int k_end = k_start + block_k;
    if (k_end > K) {
        k_end = K;
    }

    float reg_sum[LOAD_SIZE] = {0.0f};
    float indices[LOAD_SIZE] = {0.0f};
    extern __shared__ float smem_out[];

    if (REG_COUNT < LOAD_SIZE) {
        for (int idx = threadIdx.x; idx < M; idx += blockDim.x) {
            smem_out[idx] = 0.0f;
        }
        __syncthreads();
    }

    for (int k = k_start; k < k_end; k++) {
        float x_val = __half2float(x[k]);
        if (fabsf(x_val) < threshold) {
            continue;
        }

        int col_begin = col_indices[k];
        int col_end = col_indices[k + 1];
        if (col_begin >= col_end) {
            continue;
        }

        int col_begin_aligned = col_begin - (col_begin % LOAD_SIZE);
        int thread_ordinal = col_begin_aligned + row;

        const float4 *A_vec = reinterpret_cast<const float4 *>(A_packed);
        const unsigned int *D_vec = reinterpret_cast<const unsigned int *>(deltas);

        float4 a_raw = {0, 0, 0, 0};
        unsigned int d_raw = 0;
        if (thread_ordinal < col_end) {
            a_raw = A_vec[thread_ordinal / LOAD_SIZE];
            d_raw = D_vec[thread_ordinal / LOAD_SIZE];
        }
        __half *a_vals = reinterpret_cast<__half *>(&a_raw);

#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i < col_begin) {
                unsigned int delta_mask = ~(15u << (4 * i));
                d_raw &= delta_mask;
                a_vals[i] = __float2half(0.0f);
            }
        }

        int local_delta_sum = 0;
        int sum_deltas_in_col = 0;
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            local_delta_sum += int((d_raw >> (4 * i)) & 15);
        }
        sum_deltas_in_col += warp_exclusive_sum_external(local_delta_sum);

#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++) {
            if (thread_ordinal + i >= col_begin && thread_ordinal + i < col_end) {
                float a_val = __half2float(a_vals[i]);
                int real_delta = sum_deltas_in_col + thread_ordinal + i - col_begin;

                if (i < REG_COUNT) {
                    indices[i] += real_delta;
                    reg_sum[i] = __fmaf_rn(a_val, indices[i], reg_sum[i]);
                } else {
                    smem_out[real_delta] = __fmaf_rn(a_val, x_val, smem_out[real_delta]);
                }
            }
        }
    }

    if (REG_COUNT < LOAD_SIZE) {
        __syncthreads();
        float4* smem_out4 = (float4*)smem_out;
        for (int idx = threadIdx.x; idx < M / 4; idx += blockDim.x) {
            float4 val = smem_out4[idx];
            atomicAdd(&y[idx * 4 + 0], val.x);
            atomicAdd(&y[idx * 4 + 1], val.y);
            atomicAdd(&y[idx * 4 + 2], val.z);
            atomicAdd(&y[idx * 4 + 3], val.w);
        }
    }

#pragma unroll 8
    for (int i = 0; i < LOAD_SIZE; i++) {
        if (i < REG_COUNT && row + i < M) {
            y[row + i] = reg_sum[i];
        }
    }
}

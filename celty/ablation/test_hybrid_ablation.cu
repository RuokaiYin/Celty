// Standalone benchmark for hybrid register/SMEM accumulation ablation.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "hybrid_api.cuh"

void generateMatrix_Sparse(half* matrix, int rows, int cols, float sparsity_ratio)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);

    int total_elements = rows * cols;
    int num_zeros = static_cast<int>(total_elements * sparsity_ratio);
    std::vector<float> abs_values(total_elements);

    for (int i = 0; i < total_elements; i++) {
        float val = normal_dis(gen);
        matrix[i] = __float2half(val);
        abs_values[i] = std::abs(val);
    }

    if (num_zeros <= 0) {
        return;
    }
    if (num_zeros >= total_elements) {
        num_zeros = total_elements - 1;
    }

    std::nth_element(abs_values.begin(),
                     abs_values.begin() + num_zeros,
                     abs_values.end());
    float threshold = abs_values[num_zeros];

    for (int i = 0; i < total_elements; i++) {
        if (std::abs(__half2float(matrix[i])) < threshold) {
            matrix[i] = __float2half(0.0f);
        }
    }
}

float generateVector_Sparse(half* vector, int length, float sparsity_ratio)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);

    int num_zeros = static_cast<int>(length * sparsity_ratio);
    std::vector<float> abs_values(length);

    for (int i = 0; i < length; i++) {
        float val = normal_dis(gen);
        vector[i] = __float2half(val);
        abs_values[i] = std::abs(val);
    }

    if (num_zeros <= 0) {
        return 0.0f;
    }
    if (num_zeros >= length) {
        num_zeros = length - 1;
    }

    std::nth_element(abs_values.begin(),
                     abs_values.begin() + num_zeros,
                     abs_values.end());
    return abs_values[num_zeros];
}

#define COMPRESSOR_UPDIV(a, b) (a + b - 1) / b
#define COMPRESSOR_FOR(i, n) for (int i = 0; i < n; ++i)

struct PackedMatrixDelta {
    half* values;
    unsigned char* deltas;
    int* col_indices;
    int compressed_size;
    int compressed_padded_size;
};

PackedMatrixDelta compress_cols_cpu(const __half *M, int A_rows, int A_cols)
{
    PackedMatrixDelta result;
    std::vector<int> col_indices_vec(A_cols + 1);
    std::vector<__half> values_vec(A_cols * A_rows + 8, __float2half(0.0f));
    std::vector<unsigned char> deltas_vec(A_cols * A_rows + 8, 0);

    col_indices_vec[0] = 0;

    int num_values = 0;
    COMPRESSOR_FOR(col, A_cols)
    {
        unsigned int delta = 0;
        int last_non_zero = num_values;

        COMPRESSOR_FOR(row, A_rows)
        {
            float val = __half2float(M[col * A_rows + row]);
            if (val != 0.0 || delta == 15)
            {
                int delta_index = num_values / 2;
                int delta_subindex = num_values % 2;
                if (delta_subindex == 0)
                {
                    deltas_vec[delta_index] = 0;
                }
                deltas_vec[delta_index] += (delta) << (4 * delta_subindex);
                values_vec[num_values] = __float2half(val);
                num_values += 1;
                delta = 0;
                if (val != 0.0) {
                    last_non_zero = num_values;
                }
            }
            else
            {
                delta += 1;
            }
        }
        num_values = last_non_zero;
        col_indices_vec[col + 1] = num_values;
    }
    result.compressed_size = num_values;
    result.compressed_padded_size = COMPRESSOR_UPDIV(num_values, 8) * 8;

    cudaMalloc(&result.values, result.compressed_padded_size * sizeof(__half));
    cudaMalloc(&result.deltas, (result.compressed_padded_size / 2) * sizeof(unsigned char));
    cudaMalloc(&result.col_indices, (A_cols + 1) * sizeof(int));

    cudaMemcpy(result.values, values_vec.data(),
               result.compressed_padded_size * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.deltas, deltas_vec.data(),
               (result.compressed_padded_size / 2) * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.col_indices, col_indices_vec.data(),
               (A_cols + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    return result;
}

void free_packed_matrix(PackedMatrixDelta& packed)
{
    cudaFree(packed.values);
    cudaFree(packed.deltas);
    cudaFree(packed.col_indices);
}

#define HGEMV_CHECK_CUBLAS_ERROR(call)                        \
    do {                                                      \
        cublasStatus_t status = (call);                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n",   \
                    __FILE__, __LINE__, status);              \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

cublasHandle_t getCublasTensorOpHandle()
{
    cublasHandle_t handle = nullptr;
    HGEMV_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    HGEMV_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    return handle;
}

void cublasTensorOp(half *A, half *B, half *C, size_t N, size_t K)
{
    static cublasHandle_t handle = getCublasTensorOpHandle();
    static size_t M = 1;
    static float alpha = 1.0;
    static float beta = 0.0;

    HGEMV_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                          N, M, K, &alpha,
                                          B, CUDA_R_16F, K,
                                          A, CUDA_R_16F, K,
                                          &beta,
                                          C, CUDA_R_16F, N,
                                          CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(int argc, char** argv)
{
    int M = 4096;
    int K = 4096;
    float sparsity_ratio_x = 0.5f;
    float sparsity_ratio_w = 0.5f;
    int num_iterations = 100;

    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) K = std::atoi(argv[2]);
    if (argc > 3) sparsity_ratio_x = std::atof(argv[3]);
    if (argc > 4) sparsity_ratio_w = std::atof(argv[4]);
    if (argc > 5) num_iterations = std::atoi(argv[5]);

    std::cout << "Hybrid Ablation Test Configuration:" << std::endl;
    std::cout << "  M = " << M << " (rows of A)" << std::endl;
    std::cout << "  K = " << K << " (columns of A)" << std::endl;
    std::cout << "  Sparsity ratio of x: " << sparsity_ratio_x << std::endl;
    std::cout << "  Sparsity ratio of w: " << sparsity_ratio_w << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;
    std::cout << std::endl;

    half* h_A_sparse = (half*)malloc(M * K * sizeof(half));
    half* h_A_col_sparse = (half*)malloc(M * K * sizeof(half));
    half* h_x = (half*)malloc(K * sizeof(half));

    std::cout << "Generating test data..." << std::endl;
    generateMatrix_Sparse(h_A_sparse, M, K, sparsity_ratio_w);
    float threshold = generateVector_Sparse(h_x, K, sparsity_ratio_x);

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < K; col++) {
            h_A_col_sparse[col * M + row] = h_A_sparse[row * K + col];
        }
    }

    PackedMatrixDelta packed_A{};
    std::cout << "Packing matrix (delta based)..." << std::endl;
    packed_A = compress_cols_cpu(h_A_col_sparse, M, K);

    float compressed_size = (packed_A.compressed_padded_size) * 1.25f;
    float original_size = M * K;
    std::cout << "Packing efficiency: " << compressed_size / original_size << std::endl;

    half* d_A_sparse;
    half* d_x;
    float* d_y;
    half* d_y_cu;

    cudaMalloc(&d_A_sparse, M * K * sizeof(half));
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMalloc(&d_y_cu, M * sizeof(half));

    cudaMemcpy(d_A_sparse, h_A_sparse, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(float));
    cudaMemset(d_y_cu, 0, M * sizeof(half));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int hybrid_reg_counts[5] = {0, 2, 4, 6, 8};
    const char* hybrid_reg_labels[5] = {"0%", "25%", "50%", "75%", "100%"};
    double avg_time_ms_hybrid[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++) {
        cublasTensorOp(d_x, d_A_sparse, d_y_cu, M, K);
        for (int h = 0; h < 5; h++) {
            SPVSPM_CUDACores_ColMajor_Hybrid_API(
                stream, packed_A.values, packed_A.deltas, packed_A.col_indices,
                d_x, d_y, M, K, threshold, hybrid_reg_counts[h]);
        }
    }
    cudaDeviceSynchronize();

    std::cout << "Running benchmark (CUBLAS)..." << std::endl;
    cudaEvent_t start_cu, stop_cu;
    cudaEventCreate(&start_cu);
    cudaEventCreate(&stop_cu);

    cudaEventRecord(start_cu);
    for (int i = 0; i < num_iterations; i++) {
        cublasTensorOp(d_x, d_A_sparse, d_y_cu, M, K);
    }
    cudaEventRecord(stop_cu);
    cudaEventSynchronize(stop_cu);

    float total_ms_cu = 0.0f;
    cudaEventElapsedTime(&total_ms_cu, start_cu, stop_cu);
    double avg_time_ms_cu = total_ms_cu / num_iterations;

    cudaEventDestroy(start_cu);
    cudaEventDestroy(stop_cu);

    for (int h = 0; h < 5; h++) {
        std::cout << "Running benchmark (SPVSPM Hybrid - "
                  << hybrid_reg_labels[h] << " register accumulation)..." << std::endl;
        cudaMemsetAsync(d_y, 0, M * sizeof(float), stream);
        cudaStreamSynchronize(stream);
        auto start_hybrid = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++) {
            SPVSPM_CUDACores_ColMajor_Hybrid_API(
                stream, packed_A.values, packed_A.deltas, packed_A.col_indices,
                d_x, d_y, M, K, threshold, hybrid_reg_counts[h]);
        }
        cudaStreamSynchronize(stream);
        auto end_hybrid = std::chrono::high_resolution_clock::now();
        auto duration_hybrid = std::chrono::duration_cast<std::chrono::microseconds>(end_hybrid - start_hybrid);
        avg_time_ms_hybrid[h] = duration_hybrid.count() / (1000.0 * num_iterations);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "HYBRID ABLATION RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "Average kernel time (CUBLAS): " << avg_time_ms_cu << " ms" << std::endl;
    double effective_flops = 2.0 * M * K * (1.0f - sparsity_ratio_x) * (1.0f - sparsity_ratio_w);

    for (int h = 0; h < 5; h++) {
        std::cout << "SPVSPM Hybrid (" << hybrid_reg_labels[h] << " reg accum): "
                  << avg_time_ms_hybrid[h] << " ms, "
                  << effective_flops / (avg_time_ms_hybrid[h] * 1e-3) / 1e9 << " GFLOPS, "
                  << "speedup vs Dense: " << avg_time_ms_cu / avg_time_ms_hybrid[h] << "x" << std::endl;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "No CORRECTNESS CHECK" << std::endl;
    std::cout << "========================================" << std::endl;

    cudaFree(d_A_sparse);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_cu);
    free_packed_matrix(packed_A);

    free(h_A_sparse);
    free(h_A_col_sparse);
    free(h_x);

    cudaStreamDestroy(stream);
    return 0;
}

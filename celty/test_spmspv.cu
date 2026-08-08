// Test/Benchmark for Celty Kernels
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "GEMV_API.cuh"

void generateMatrix(half* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = __float2half(normal_dis(gen));
    }
}


void generateMatrix_Sparse(half* matrix, int rows, int cols, float sparsity_ratio) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);
    
    int total_elements = rows * cols;
    int num_zeros = static_cast<int>(total_elements * sparsity_ratio);
    std::vector<float> abs_values(total_elements);
    
    // Generate random values and track absolute values
    for (int i = 0; i < total_elements; i++) {
        float val = normal_dis(gen);
        matrix[i] = __float2half(val);
        abs_values[i] = std::abs(val);
    }
    
    // Find threshold
    std::nth_element(abs_values.begin(), 
                     abs_values.begin() + num_zeros, 
                     abs_values.end());
    float threshold = abs_values[num_zeros];
    
    // Zero out elements below threshold
    for (int i = 0; i < total_elements; i++) {
        if (std::abs(__half2float(matrix[i])) < threshold) {
            matrix[i] = __float2half(0.0f);
        }
    }    
}

void generateVector(half* vector, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);
    
    for (int i = 0; i < length; i++) {
        vector[i] = __float2half(normal_dis(gen));
    }
}

float generateVector_Sparse(half* vector, int length, float sparsity_ratio) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);

    int num_zeros = static_cast<int>(length * sparsity_ratio);
    std::vector<float> abs_values(length); //! create abs_values buffers.

    for (int i = 0; i < length; i++) {
        float val = normal_dis(gen);
        vector[i] = __float2half(val);
        abs_values[i] = std::abs(val);
    }
    std::nth_element(abs_values.begin(),abs_values.begin()+num_zeros, abs_values.end());
    float threshold = abs_values[num_zeros];
    
    return threshold;
}


#define COMPRESSOR_UPDIV(a, b) (a + b - 1) / b
#define COMPRESSOR_FOR(i, n) for (int i = 0; i < n; ++i)
struct PackedMatrixDelta {
    half* values;       // Packed non-zero values (same as before)
    unsigned char* deltas;    // Layout: warp_bitmaps[warp * K + k]
    int* col_indices;     // Layout: col_warp_offsets[warp * K + k]
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
                if (val != 0.0)
                    last_non_zero = num_values;
            }
            else
            {
                delta += 1;
            }
        }//! Till here, the current column is scanned.
        num_values = last_non_zero;
        col_indices_vec[col + 1] = num_values;
    }   
    result.compressed_size = num_values;
    result.compressed_padded_size = COMPRESSOR_UPDIV(num_values, 8) * 8;

    cudaMalloc(&result.values, result.compressed_padded_size * sizeof(__half));
    cudaMalloc(&result.deltas, (result.compressed_padded_size / 2) * sizeof(unsigned char));
    cudaMalloc(&result.col_indices, (A_cols + 1) * sizeof(int));

    // Copy from host vectors to device
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

// ── V3 packed format: same as PackedMatrixDelta + per-warp cumulative delta offsets ──
#define V3_WARP_CHUNK_SIZE (32 * 8)  // WARP_SIZE * LOAD_SIZE = 256

struct PackedMatrixDeltaV3 {
    half* values;
    unsigned char* deltas;
    int* col_indices;
    int* warp_cum_deltas;   //! warp_cum_deltas[k * max_warp_chunks + w] = cumulative delta
                            //! sum of column k's packed elements BEFORE warp chunk w
    int compressed_size;
    int compressed_padded_size;
    int max_warp_chunks;    //! max entries per column in warp_cum_deltas
};

PackedMatrixDeltaV3 compress_cols_cpu_v3(const __half *M_mat, int A_rows, int A_cols)
{
    PackedMatrixDeltaV3 result;
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
            float val = __half2float(M_mat[col * A_rows + row]);
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
                if (val != 0.0)
                    last_non_zero = num_values;
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

    //! ── Compute per-warp cumulative delta offsets ──
    // Must be >= warp chunks per column AND >= global warps (grid: ceil(M/1024) blocks)
    const int chunks_per_col = (A_rows + V3_WARP_CHUNK_SIZE - 1) / V3_WARP_CHUNK_SIZE + 1;
    const int V3_positions_per_block = 1024;  // 128 threads * LOAD_SIZE
    const int num_blocks_m = (A_rows + V3_positions_per_block - 1) / V3_positions_per_block;
    const int num_global_warps = num_blocks_m * 4;  // WARPS_PER_BLOCK = 4
    result.max_warp_chunks = (chunks_per_col > num_global_warps) ? chunks_per_col : num_global_warps;
    std::vector<int> warp_cum_deltas_vec(A_cols * result.max_warp_chunks, 0);

    for (int k = 0; k < A_cols; k++) {
        int col_begin = col_indices_vec[k];
        int col_end = col_indices_vec[k + 1];
        int col_size = col_end - col_begin;
        int col_begin_aligned = col_begin - (col_begin % 8); // LOAD_SIZE alignment

        //! Chunk 0 always starts with cum_delta = 0
        warp_cum_deltas_vec[k * result.max_warp_chunks + 0] = 0;

        //! Single scan through column to compute cum_delta at each chunk boundary
        int cum_delta = 0;
        for (int i = 0; i < col_size; i++) {
            int abs_pos = col_begin + i;
            //! Get delta for this packed element
            int delta_index = abs_pos / 2;
            int delta_subindex = abs_pos % 2;
            int delta_val = (deltas_vec[delta_index] >> (4 * delta_subindex)) & 15;
            cum_delta += delta_val;

            //! Check if the NEXT position crosses a warp chunk boundary
            int next_pos_from_aligned = (abs_pos + 1) - col_begin_aligned;
            if (next_pos_from_aligned > 0 &&
                next_pos_from_aligned % V3_WARP_CHUNK_SIZE == 0) {
                int chunk = next_pos_from_aligned / V3_WARP_CHUNK_SIZE;
                if (chunk < result.max_warp_chunks) {
                    warp_cum_deltas_vec[k * result.max_warp_chunks + chunk] = cum_delta;
                }
            }
        }
    }

    //! ── Copy everything to device ──
    cudaMalloc(&result.values, result.compressed_padded_size * sizeof(__half));
    cudaMalloc(&result.deltas, (result.compressed_padded_size / 2) * sizeof(unsigned char));
    cudaMalloc(&result.col_indices, (A_cols + 1) * sizeof(int));
    cudaMalloc(&result.warp_cum_deltas, A_cols * result.max_warp_chunks * sizeof(int));

    cudaMemcpy(result.values, values_vec.data(),
               result.compressed_padded_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(result.deltas, deltas_vec.data(),
               (result.compressed_padded_size / 2) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(result.col_indices, col_indices_vec.data(),
               (A_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(result.warp_cum_deltas, warp_cum_deltas_vec.data(),
               A_cols * result.max_warp_chunks * sizeof(int), cudaMemcpyHostToDevice);

    return result;
}

void free_packed_matrix_v3(PackedMatrixDeltaV3& packed)
{
    cudaFree(packed.values);
    cudaFree(packed.deltas);
    cudaFree(packed.col_indices);
    cudaFree(packed.warp_cum_deltas);
}


// Reference implementation on CPU
void gemv_reference(const half* A, const half* x, float* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += __half2float(A[i * N + j]) * __half2float(x[j]);
        }
        y[i] = sum;
    }
}

void gemv_sparse_reference(const half* A, const half* x, float* y, int M, int N, float threshold) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float x_val = __half2float(x[j]);
            // Skip if magnitude below threshold (sparse element)
            if (std::abs(x_val) >= threshold) {
                sum += __half2float(A[i * N + j]) * x_val;
            }
        }
        y[i] = sum;
    }
}

float compareResults(const float* y1, const float* y2, int M, float tolerance = 1e-3f) {
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    
    for (int i = 0; i < M; i++) {
        float diff = fabsf(y1[i] - y2[i]);
        float rel_diff = (fabsf(y2[i]) > 1e-6f) ? diff / fabsf(y2[i]) : diff;
        
        max_diff = fmaxf(max_diff, diff);
        max_rel_diff = fmaxf(max_rel_diff, rel_diff);
    }
    
    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;
    
    return max_rel_diff;
}



//! Main function for spMspV kernels orchestration.
int main(int argc, char** argv) {

    std::cout << "\n========================================" << std::endl;
    std::cout << "CELTY SPMSPV GPU KERNEL BENCHMARK" << std::endl;
    std::cout << "========================================" << std::endl;


    int M = 4096;
    int K = 4096;
    float sparsity_ratio_x = 0.0f;
    float sparsity_ratio_w = 0.01f;
    
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) K = std::atoi(argv[2]);
    if (argc > 3) sparsity_ratio_x = std::atof(argv[3]);
    if (argc > 4) sparsity_ratio_w = std::atof(argv[4]);
    
    std::cout << "spMspV Test Configuration:" << std::endl;
    std::cout << "  M = " << M << " (rows of A)" << std::endl;
    std::cout << "  K = " << K << " (columns of A)" << std::endl;
    std::cout << "  Computing: y = A * x" << std::endl;
    std::cout << "  Sparsity ratio of x: " << sparsity_ratio_x << std::endl;
    std::cout << "  Sparsity ratio of w: " << sparsity_ratio_w << std::endl;
    std::cout << std::endl;
    
    // Allocate host memory
    half* h_A = (half*)malloc(M * K * sizeof(half));
    half* h_A_col = (half*)malloc(M * K * sizeof(half));
    half* h_A_sparse = (half*)malloc(M * K * sizeof(half));
    half* h_A_col_sparse = (half*)malloc(M * K * sizeof(half));

    half* h_x = (half*)malloc(K * sizeof(half));
    float* h_y = (float*)malloc(M * sizeof(float));
    float* h_y_ref = (float*)malloc(M * sizeof(float));
    
    // Generate test data
    std::cout << "Generating spMspV test data..." << std::endl;
    generateMatrix(h_A, M, K);
    generateMatrix_Sparse(h_A_sparse, M, K, sparsity_ratio_w);

    float threshold = generateVector_Sparse(h_x, K, sparsity_ratio_x);
    memset(h_y, 0, M * sizeof(float));
    memset(h_y_ref, 0, M * sizeof(float));

    // Create column-major A and sparse A for column-accumulation path
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < K; col++) {
            h_A_col[col * M + row] = h_A[row * K + col];
            h_A_col_sparse[col * M + row] = h_A_sparse[row * K + col];
        }
    }


    PackedMatrixDelta packed_A{};
    std::cout << "Packing matrix (delta based)..." << std::endl;
    packed_A = compress_cols_cpu(h_A_col_sparse, M, K);

    float compressed_size = (packed_A.compressed_padded_size)*1.25;
    float original_size = M * K;
    std::cout << "Packing efficiency: " << compressed_size/original_size << std::endl;

    //! Keep here for reference.
    // PackedMatrixDeltaV3 packed_A_v3{};
    // std::cout << "Packing matrix (delta + warp offsets for V3)..." << std::endl;
    // packed_A_v3 = compress_cols_cpu_v3(h_A_col_sparse, M, K);
    // std::cout << "  max_warp_chunks = " << packed_A_v3.max_warp_chunks << std::endl;

    // Allocate device memory
    half* d_A;
    half* d_A_col;
    half* d_A_col_sparse;
    half* d_x;
    float* d_y;
    half* d_y_cu;
    
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_A_col, M * K * sizeof(half));
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMalloc(&d_y, M * sizeof(float));
    cudaMalloc(&d_y_cu, M * sizeof(half));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col, h_A_col, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMalloc(&d_A_col_sparse, M * K * sizeof(half));
    cudaMemcpy(d_A_col_sparse, h_A_col_sparse, M * K * sizeof(half), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x, K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(float));
    cudaMemset(d_y_cu, 0, M * sizeof(half));


    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++) {
        cublasTensorOp(d_x, d_A, d_y_cu, M, K, stream); //! CuBLAS GEMV baseline
        SPVSPM_CUDACores_ColMajor_API(stream, packed_A.values, packed_A.deltas, packed_A.col_indices, d_x, d_y, M, K, threshold); //! Global DRAM-accumulation
        Celty_SMEM_Kernel_API(stream, packed_A.values, packed_A.deltas, packed_A.col_indices, d_x, d_y, M, K, threshold); //! Celty-SMEM
        Celty_SIMT_Kernel_API(stream, packed_A.values, packed_A.deltas, packed_A.col_indices, d_x, d_y, M, K, threshold); //! Celty-SIMT
    }
    cudaDeviceSynchronize();

    int num_iterations = 100;

    // One event pair, reused for every benchmark below — created once, destroyed once at the end.
    cudaEvent_t start_cu, stop_cu;
    cudaEventCreate(&start_cu);
    cudaEventCreate(&stop_cu);

    //* Benchmark Cublas GEMV
    std::cout << "Running dense GEMV baseline (CUBLAS)..." << std::endl;
    cudaEventRecord(start_cu, stream);
    for (int i = 0; i < num_iterations; i++) {
        cublasTensorOp(d_x, d_A, d_y_cu, M, K, stream);
    }
    cudaEventRecord(stop_cu, stream);
    cudaEventSynchronize(stop_cu);

    float total_ms_cu = 0;
    cudaEventElapsedTime(&total_ms_cu, start_cu, stop_cu);
    double avg_time_ms_cu = total_ms_cu / num_iterations;


    // //* Benchmark Global DRAM-accumulation spMspV kernels
    std::cout << "Running benchmark (spMspV Global-accumulation)..." << std::endl;
    cudaMemsetAsync(d_y, 0, M * sizeof(float), stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start_cu, stream);
    for (int i = 0; i < num_iterations; i++) {
        SPVSPM_CUDACores_ColMajor_API(stream, packed_A.values, packed_A.deltas, packed_A.col_indices, d_x, d_y, M, K, threshold);
    }
    cudaEventRecord(stop_cu, stream);
    cudaEventSynchronize(stop_cu);

    float total_ms_spmspv_global = 0;
    cudaEventElapsedTime(&total_ms_spmspv_global, start_cu, stop_cu);
    double avg_time_ms_spmspv_global = total_ms_spmspv_global / num_iterations;

    float* h_y_spmspv = (float*)malloc(M * sizeof(float));
    cudaMemcpy(h_y_spmspv, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);


    // //* Benchmark Celty-SIMT kernels
    std::cout << "Running benchmark (Celty-SIMT)..." << std::endl;
    cudaMemsetAsync(d_y, 0, M * sizeof(float), stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start_cu, stream);
    for (int i = 0; i < num_iterations; i++) {
        Celty_SIMT_Kernel_API(stream, packed_A.values, packed_A.deltas, packed_A.col_indices, d_x, d_y, M, K, threshold);
    }
    cudaEventRecord(stop_cu, stream);
    cudaEventSynchronize(stop_cu);

    float total_ms_spmspv_celty = 0;
    cudaEventElapsedTime(&total_ms_spmspv_celty, start_cu, stop_cu);
    double avg_time_ms_spmspv_celty = total_ms_spmspv_celty / num_iterations;


    // //* Benchmark Celty-SMEM kernels
    std::cout << "Running benchmark (Celty-SMEM)..." << std::endl;
    cudaMemsetAsync(d_y, 0, M * sizeof(float), stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start_cu, stream);
    for (int i = 0; i < num_iterations; i++) {
        Celty_SMEM_Kernel_API(stream, packed_A.values, packed_A.deltas, packed_A.col_indices, d_x, d_y, M, K, threshold);
    }
    cudaEventRecord(stop_cu, stream);
    cudaEventSynchronize(stop_cu);

    float total_ms_spmspv_celtysmem = 0;
    cudaEventElapsedTime(&total_ms_spmspv_celtysmem, start_cu, stop_cu);
    double avg_time_ms_spmspv_celtysmem = total_ms_spmspv_celtysmem / num_iterations;

    // ... (any further benchmarks follow the same pattern) ...

    cudaEventDestroy(start_cu);
    cudaEventDestroy(stop_cu);

    // Reference computation (CPU)
    std::cout << "Computing reference (CPU)..." << std::endl;
    gemv_reference(h_A, h_x, h_y_ref, M, K);
    
    // Compare results
    std::cout << "\n========================================" << std::endl;
    std::cout << "PERFORMANCE RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;

    double gflops_cu = (2.0 * M * K) / (avg_time_ms_cu * 1e-3) / 1e9;
    std::cout << "Average kernel time (CUBLAS): " << avg_time_ms_cu << " ms, "
              << gflops_cu << " GFLOPS" << std::endl;

    double effective_flops = 2.0 * M * K * (1.0f - sparsity_ratio_x) * (1.0f - sparsity_ratio_w);

    std::cout << "spMspV baseline (Global psum acc): " << avg_time_ms_spmspv_global << " ms, " 
              << effective_flops / (avg_time_ms_spmspv_global * 1e-3) / 1e9 << " GFLOPS, "
              << "speedup vs Dense: " << avg_time_ms_cu / avg_time_ms_spmspv_global << "x, "<< std::endl;

    std::cout << "Celty-SMEM: " << avg_time_ms_spmspv_celtysmem << " ms, "
              << effective_flops / (avg_time_ms_spmspv_celtysmem * 1e-3) / 1e9 << " GFLOPS, "
              << "speedup vs Dense: " << avg_time_ms_cu / avg_time_ms_spmspv_celtysmem << "x, "<< std::endl;

    std::cout << "Celty-SIMT: " << avg_time_ms_spmspv_celty << " ms, "
              << effective_flops / (avg_time_ms_spmspv_celty * 1e-3) / 1e9 << " GFLOPS, "
              << "speedup vs Dense: " << avg_time_ms_cu / avg_time_ms_spmspv_celty << "x, "<< std::endl;
    
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "No CORRECTNESS CHECK IS PROVIDED IN THIS TESTBENCH" << std::endl;
    std::cout << "========================================" << std::endl;


    std::cout << "========================================\n" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_A_col);
    cudaFree(d_A_col_sparse);

    free(h_A);
    free(h_A_col);
    free(h_x);
    free(h_y);
    free(h_y_ref);
    free(h_A_sparse);
    free(h_A_col_sparse);
    free_packed_matrix(packed_A);
    
    cudaStreamDestroy(stream);
    
    return 0;
}

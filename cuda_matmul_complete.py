import modal
import os

app = modal.App("cuda-matmul-h100-complete")

# Create CUDA image with all dependencies
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get -y install cuda-toolkit-12-3 libcublas-dev-12-3"
    )
    .env({"PATH": "/usr/local/cuda-12.3/bin:$PATH"})
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"})
)

# ==============================================================================
# KERNEL 1: NAIVE IMPLEMENTATION
# ==============================================================================

NAIVE_KERNEL = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;
    
    float tflops = (2.0f * M * N * K) / (ms * 1e9);
    printf("NAIVE KERNEL\\n");
    printf("  Time: %.2f ms\\n", ms);
    printf("  Performance: %.2f TFLOPS\\n", tflops);
    printf("  Grid: (%d, %d), Block: (%d, %d)\\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
"""

# ==============================================================================
# KERNEL 2: SHARED MEMORY 1D TILING
# ==============================================================================

SMEM_KERNEL = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;

__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                          const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN * TM;
    
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float threadResults[TM] = {0.0};
    
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint i = 0; i < TM; ++i) {
                threadResults[i] += As[(threadRow + i) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }
    
    for (uint i = 0; i < TM; ++i) {
        C[(threadRow + i) * N + threadCol] = 
            alpha * threadResults[i] + beta * C[(threadRow + i) * N + threadCol];
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / TM);
    
    sgemm_smem<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        sgemm_smem<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;
    
    float tflops = (2.0f * M * N * K) / (ms * 1e9);
    printf("SHARED MEMORY 1D TILING\\n");
    printf("  Time: %.2f ms\\n", ms);
    printf("  Performance: %.2f TFLOPS\\n", tflops);
    printf("  Tile: BM=%d, BN=%d, BK=%d, TM=%d\\n", BM, BN, BK, TM);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
"""

# ==============================================================================
# KERNEL 3: 2D BLOCK TILING
# ==============================================================================

TILING_2D_KERNEL = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_2d(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);
    
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint strideA = numThreadsBlocktile / BK;
    
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    const uint strideB = numThreadsBlocktile / BN;
    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] = 
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = 
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += 
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = 
                alpha * threadResults[resIdxM * TN + resIdxN] + 
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    
    sgemm_2d<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        sgemm_2d<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;
    
    float tflops = (2.0f * M * N * K) / (ms * 1e9);
    printf("2D BLOCK TILING\\n");
    printf("  Time: %.2f ms\\n", ms);
    printf("  Performance: %.2f TFLOPS\\n", tflops);
    printf("  Tile: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\\n", BM, BN, BK, TM, TN);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
"""

# ==============================================================================
# KERNEL 4: VECTORIZED LOADS
# ==============================================================================

VECTORIZED_KERNEL = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void sgemm_vectorized(int M, int N, int K, float alpha, const float *A,
                                 const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
    
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);
    
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;
    
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint rowStrideA = numThreadsBlocktile / (BK / 4);
    
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint rowStrideB = numThreadsBlocktile / (BN / 4);
    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerRowA + offset) * BK + innerColA * 4 + 0] = tmp.x;
            As[(innerRowA + offset) * BK + innerColA * 4 + 1] = tmp.y;
            As[(innerRowA + offset) * BK + innerColA * 4 + 2] = tmp.z;
            As[(innerRowA + offset) * BK + innerColA * 4 + 3] = tmp.w;
        }
        
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            float4 tmp = reinterpret_cast<const float4*>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
            Bs[(innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();
        
        A += BK;
        B += BK * N;
        
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += 
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp;
            tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 0];
            tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 1];
            tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 2];
            tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN + 3];
            
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
        }
    }
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    
    sgemm_vectorized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        sgemm_vectorized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;
    
    float tflops = (2.0f * M * N * K) / (ms * 1e9);
    printf("VECTORIZED LOADS (float4)\\n");
    printf("  Time: %.2f ms\\n", ms);
    printf("  Performance: %.2f TFLOPS\\n", tflops);
    printf("  Tile: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\\n", BM, BN, BK, TM, TN);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
"""

# ==============================================================================
# KERNEL 5: cuBLAS BASELINE
# ==============================================================================

CUBLAS_BENCHMARK = """
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;
    
    float tflops = (2.0f * M * N * K) / (ms * 1e9);
    printf("cuBLAS (NVIDIA Optimized)\\n");
    printf("  Time: %.2f ms\\n", ms);
    printf("  Performance: %.2f TFLOPS\\n", tflops);
    
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
"""

# ==============================================================================
# MODAL FUNCTIONS
# ==============================================================================

@app.function(gpu="H100", image=cuda_image, timeout=1800)
def run_kernel(kernel_name: str, kernel_code: str):
    """Compile and run a single CUDA kernel."""
    import subprocess
    import os
    
    os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    print(f"\n{'='*80}")
    print(f"RUNNING: {kernel_name}")
    print(f"{'='*80}\n")
    
    kernel_file = f"/tmp/{kernel_name}.cu"
    binary_file = f"/tmp/{kernel_name}"
    
    with open(kernel_file, "w") as f:
        f.write(kernel_code)
    
    if "cublas" in kernel_name.lower():
        compile_cmd = f"/usr/local/cuda-12.3/bin/nvcc -arch=sm_90 -lcublas -o {binary_file} {kernel_file}"
    else:
        compile_cmd = f"/usr/local/cuda-12.3/bin/nvcc -arch=sm_90 -o {binary_file} {kernel_file}"
    
    print(f"Compiling...")
    compile_result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    
    if compile_result.returncode != 0:
        return {"kernel": kernel_name, "status": "compilation_failed", "error": compile_result.stderr}
    
    print("✓ Compilation successful\n")
    
    print("Running kernel...")
    run_result = subprocess.run(binary_file, capture_output=True, text=True)
    
    if run_result.returncode != 0:
        return {"kernel": kernel_name, "status": "runtime_error", "error": run_result.stderr}
    
    print(run_result.stdout)
    
    return {"kernel": kernel_name, "status": "success", "output": run_result.stdout}

@app.function(gpu="H100", image=cuda_image, timeout=1800)
def check_gpu_info():
    """Display GPU information."""
    import subprocess
    import os
    
    os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ.get('PATH', '')
    
    print("\n" + "="*80)
    print("GPU INFORMATION")
    print("="*80 + "\n")
    
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
    
    result = subprocess.run(["/usr/local/cuda-12.3/bin/nvcc", "--version"], capture_output=True, text=True)
    print("\nCUDA Compiler Version:")
    print(result.stdout)
    
    return "GPU info displayed"

@app.local_entrypoint()
def main():
    """Run complete CUDA matmul assignment."""
    
    print("\n" + "="*80)
    print("WEEK 3 ASSIGNMENT: CUDA MATRIX MULTIPLICATION ON H100")
    print("="*80)
    
    check_gpu_info.remote()
    
    kernels = [
        ("01_naive", NAIVE_KERNEL),
        ("02_smem", SMEM_KERNEL),
        ("03_2d_tiling", TILING_2D_KERNEL),
        ("04_vectorized", VECTORIZED_KERNEL),
        ("05_cublas", CUBLAS_BENCHMARK),
    ]
    
    results = []
    for kernel_name, kernel_code in kernels:
        result = run_kernel.remote(kernel_name, kernel_code)
        results.append(result)
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80 + "\n")
    
    for result in results:
        print(f"\n{result['kernel']}:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            output = result['output']
            for line in output.split('\n'):
                if 'Performance:' in line or 'Time:' in line or 'Tile:' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)
    print("ASSIGNMENT COMPLETE!")
    print("="*80)
    print("\nKEY INSIGHTS:")
    print("- Naive: Memory bandwidth limited (~0.5 TFLOPS)")
    print("- SMEM: Data reuse through tiling (~10-20x speedup)")
    print("- 2D Tiling: Better register usage (~3-5x over SMEM)")
    print("- Vectorized: Coalesced access (~5-10% improvement)")
    print("- cuBLAS: Highly optimized, near-peak performance")
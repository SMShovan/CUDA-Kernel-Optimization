#include <iostream>
#include <cuda.h>

#define BLOCK_SIZE 16 // Block size in each dimension
#define BM BLOCK_SIZE // Block tile size for M dimension
#define BN BLOCK_SIZE // Block tile size for N dimension
#define BK BLOCK_SIZE // Block tile size for K dimension
#define TM 4          // Thread tile size for M dimension
#define TN 4          // Thread tile size for N dimension

// CUDA kernel for matrix multiplication using shared memory with 2D thread coarsening and vectorized memory access
__global__ void matMul2DCoarsenedVectorized(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    // Shared memory for the sub-matrices of A and B
    __shared__ float As[BK][BM]; // Note: Transposed layout for A
    __shared__ float Bs[BK][BN];
    
    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Linear thread index for vectorized loading
    int tid = ty * blockDim.x + tx;
    
    // Number of threads in the block
    int numThreads = blockDim.x * blockDim.y;
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate the indices for this thread's block
    int blockRowStart = by * BM;
    int blockColStart = bx * BN;
    
    // Calculate the indices for this thread within the block
    // Each thread computes a TM x TN sub-matrix of the output
    int threadRowStart = blockRowStart + (ty * TM);
    int threadColStart = blockColStart + (tx * TN);
    
    // Allocate thread-local cache for results in register file
    float threadResults[TM][TN] = {{0.0f}};
    
    // Register caches for As and Bs
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};
    
    // Loop over all sub-matrices of A and B required to compute the block of C
    for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; bkIdx++) {
        int kOffset = bkIdx * BK;
        
        // Collaborative loading of A and B tiles using vectorized access
        // Each thread loads multiple elements in a strided pattern
        for (int i = tid; i < (BM * BK) / 4; i += numThreads) {
            int row = i / (BK/4);
            int col = (i % (BK/4)) * 4;
            
            int globalRow = blockRowStart + row;
            int globalCol = kOffset + col;
            
            if (globalRow < M && globalCol + 3 < K) {
                // Load 4 elements at once using float4
                float4 tmp = reinterpret_cast<float4*>(&A[globalRow * K + globalCol])[0];
                
                // Transpose A during the transfer to shared memory
                As[col][row] = tmp.x;
                As[col+1][row] = tmp.y;
                As[col+2][row] = tmp.z;
                As[col+3][row] = tmp.w;
            }
            else {
                // Handle boundary conditions
                for (int j = 0; j < 4; j++) {
                    if (globalCol + j < K && globalRow < M) {
                        As[col+j][row] = A[globalRow * K + globalCol + j];
                    }
                    else {
                        As[col+j][row] = 0.0f;
                    }
                }
            }
        }
        
        // Collaborative loading of B tile using vectorized access
        for (int i = tid; i < (BK * BN) / 4; i += numThreads) {
            int row = i / (BN/4);
            int col = (i % (BN/4)) * 4;
            
            int globalRow = kOffset + row;
            int globalCol = blockColStart + col;
            
            if (globalRow < K && globalCol + 3 < N) {
                // Direct vectorized load and store
                reinterpret_cast<float4*>(&Bs[row][col])[0] = 
                    reinterpret_cast<float4*>(&B[globalRow * N + globalCol])[0];
            }
            else {
                // Handle boundary conditions
                for (int j = 0; j < 4; j++) {
                    if (globalCol + j < N && globalRow < K) {
                        Bs[row][col+j] = B[globalRow * N + globalCol + j];
                    }
                    else {
                        Bs[row][col+j] = 0.0f;
                    }
                }
            }
        }
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Compute the thread's portion of the output tile
        for (int k = 0; k < BK; k++) {
            // Load relevant As entries into registers (note transposed access)
            for (int m = 0; m < TM; m++) {
                int localRow = ty * TM + m;
                if (localRow < BM) {
                    regM[m] = As[k][localRow];
                }
            }
            
            // Load relevant Bs entries into registers
            for (int n = 0; n < TN; n++) {
                int localCol = tx * TN + n;
                if (localCol < BN) {
                    regN[n] = Bs[k][localCol];
                }
            }
            
            // Perform outer product on register cache, accumulate into threadResults
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += regM[m] * regN[n];
                }
            }
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }
    
    // Write the results to global memory
    // Use vectorized writes where possible
    for (int m = 0; m < TM; m++) {
        int globalRow = threadRowStart + m;
        if (globalRow < M) {
            for (int n = 0; n < TN; n += 4) {
                int globalCol = threadColStart + n;
                
                // Check if we can do a vectorized write (all 4 elements in bounds)
                if (globalCol + 3 < N) {
                    float4 result;
                    result.x = alpha * threadResults[m][n] + beta * C[globalRow * N + globalCol];
                    result.y = alpha * threadResults[m][n+1] + beta * C[globalRow * N + globalCol + 1];
                    result.z = alpha * threadResults[m][n+2] + beta * C[globalRow * N + globalCol + 2];
                    result.w = alpha * threadResults[m][n+3] + beta * C[globalRow * N + globalCol + 3];
                    
                    // Write 4 elements at once
                    reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0] = result;
                }
                else {
                    // Handle boundary conditions
                    for (int j = 0; j < 4 && n+j < TN; j++) {
                        if (globalCol + j < N) {
                            C[globalRow * N + globalCol + j] = 
                                alpha * threadResults[m][n+j] + beta * C[globalRow * N + globalCol + j];
                        }
                    }
                }
            }
        }
    }
}

// Function to initialize matrices
void initMatrix(float *mat, int size, float value) {
    for (int i = 0; i < size; i++) {
        mat[i] = value;
    }
}

int main() {
    int M = 256; // Matrix dimensions (M x K) for A
    int N = 256; // Matrix dimensions (K x N) for B
    int K = 256; // Common dimension
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);
    
    // Initialize matrices
    initMatrix(h_A, M * K, 1.0f);
    initMatrix(h_B, K * N, 1.0f);
    initMatrix(h_C, M * N, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions - adjusted for 2D thread coarsening
    dim3 blockDim(BN/TN, BM/TM);  // Each thread handles a TM x TN tile
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // Set scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matMul2DCoarsenedVectorized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    
    // Print a sample result
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    
    // Verify result for a simple case (all 1's matrices)
    if (h_C[0] == K) {
        std::cout << "Result verified: C[0][0] = " << h_C[0] << " (expected " << K << ")" << std::endl;
        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    } else {
        std::cout << "Result incorrect: C[0][0] = " << h_C[0] << " (expected " << K << ")" << std::endl;
    }
    
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

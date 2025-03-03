#include <iostream>
#include <cuda.h>

#define BLOCK_SIZE 16 // Define block size

// CUDA kernel for matrix multiplication using shared memory
__global__ void matMulShared(float *A, float *B, float *C, int N, int K, float alpha, float beta) {
    // Shared memory for the sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread row and column within the block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Block row and column
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    
    // Calculate global row and column positions
    int row = cRow * BLOCK_SIZE + threadRow;
    int col = cCol * BLOCK_SIZE + threadCol;
    
    // Initialize accumulator
    float tmp = 0.0f;
    
    // Loop over all sub-matrices of A and B required to compute the block of C
    for (int bkIdx = 0; bkIdx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; bkIdx++) {
        // Load data into shared memory
        if (row < N && (bkIdx * BLOCK_SIZE + threadCol) < K) {
            As[threadRow][threadCol] = A[row * K + bkIdx * BLOCK_SIZE + threadCol];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }
        
        if ((bkIdx * BLOCK_SIZE + threadRow) < K && col < N) {
            Bs[threadRow][threadCol] = B[(bkIdx * BLOCK_SIZE + threadRow) * N + col];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Multiply the two matrices together
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp += As[threadRow][k] * Bs[k][threadCol];
        }
        
        // Synchronize to make sure that the preceding computation is done
        __syncthreads();
    }
    
    // Write the result to global memory
    if (row < N && col < N) {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// Function to initialize matrices
void initMatrix(float *mat, int size, float value) {
    for (int i = 0; i < size; i++) {
        mat[i] = value;
    }
}

int main() {
    int N = 256; // Matrix dimensions (N x N)
    int K = N;   // For square matrices, K = N
    size_t bytes = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    initMatrix(h_A, N * N, 1.0f);
    initMatrix(h_B, N * N, 1.0f);
    initMatrix(h_C, N * N, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Set scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Launch kernel
    matMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, alpha, beta);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Print a sample result
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    
    // Verify result for a simple case (all 1's matrices)
    if (h_C[0] == N) {
        std::cout << "Result verified: C[0][0] = " << h_C[0] << " (expected " << N << ")" << std::endl;
    } else {
        std::cout << "Result incorrect: C[0][0] = " << h_C[0] << " (expected " << N << ")" << std::endl;
    }
    
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}

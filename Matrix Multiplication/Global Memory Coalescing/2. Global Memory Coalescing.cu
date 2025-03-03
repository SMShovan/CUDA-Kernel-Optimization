#include <iostream>
#include <cuda.h>
#define BLOCK_SIZE 16 // Define block size

// CUDA kernel for matrix multiplication with specified thread mapping
__global__ void matMulNaive(float *A, float *B, float *C, int N) {
    const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    
    if (x < N && y < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[x * N + k] * B[k * N + y];
        }
        C[x * N + y] = sum;
    }
}

// Function to initialize matrices
void initMatrix(float *mat, int N, float value) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = value;
    }
}

int main() {
    int N = 256; // Matrix size N x N
    size_t bytes = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    initMatrix(h_A, N, 1.0f);
    initMatrix(h_B, N, 1.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid sizes - Now using 1D thread blocks
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE); // Using a 1D thread block of size BLOCK_SIZE²
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Print a sample result
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    
    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
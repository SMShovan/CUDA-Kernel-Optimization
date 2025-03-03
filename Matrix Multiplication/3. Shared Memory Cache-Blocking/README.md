# CUDA Matrix Multiplication with Shared Memory

This repository contains a CUDA implementation of matrix multiplication that leverages shared memory to significantly improve performance over naive implementations.

## Overview

Matrix multiplication is a fundamental operation in many computational applications, but it can be computationally expensive for large matrices. This implementation uses CUDA and GPU acceleration with an optimized shared memory approach to achieve high performance.

## Key Features

- **Shared Memory Tiling**: Uses GPU shared memory to reduce global memory accesses
- **2D Thread Organization**: Optimized thread layout for matrix operations
- **Boundary Checking**: Handles matrices of arbitrary dimensions
- **Parameterized Scaling**: Supports alpha and beta scaling factors (C = alpha*A*B + beta*C)

## Performance Improvements

Compared to the naive implementation, this shared memory version offers several significant optimizations:

1. **Reduced Global Memory Access**: The naive implementation reads each element of A and B from global memory N times, resulting in O(N³) global memory accesses. The shared memory implementation reduces this to O(N²) by loading data into fast shared memory once and reusing it.

2. **Better Memory Access Patterns**: Organized to achieve coalesced memory access where possible.

3. **Efficient Thread Organization**: Uses a 2D thread block structure (16×16) that better matches the 2D nature of matrix multiplication, compared to the naive version's 1D thread organization.

4. **Improved Computational Density**: Each thread performs more useful work with less overhead.

## Implementation Details

- Block size: 16×16 threads per block
- Each thread computes one element of the result matrix
- Uses tiling approach to break the computation into smaller chunks that fit in shared memory
- Synchronization barriers ensure correct execution across threads

## Usage

Compile with NVCC:

```bash
nvcc -o matrix_mul matrix_mul.cu
```

Run the executable:

```bash
./matrix_mul
```

## Customization

You can modify the following parameters:

- `BLOCK_SIZE`: The dimension of thread blocks (default: 16)
- `N`: Matrix dimensions (default: 256×256)
- `alpha` and `beta`: Scaling factors in the operation C = alpha*A*B + beta*C

## Requirements

- CUDA-capable GPU
- CUDA Toolkit (compatible with your GPU)
- C++ compiler

## Performance Comparison

When comparing with the naive implementation, this shared memory version demonstrates:

1. **Memory Efficiency**: ~N times fewer global memory accesses
2. **Execution Speed**: Typically 5-10x faster for large matrices
3. **Better Scalability**: Performance advantage increases with matrix size

## Limitations

- Performance is optimal when matrix dimensions are multiples of the block size
- Very large matrices may require additional optimizations or multi-GPU approaches

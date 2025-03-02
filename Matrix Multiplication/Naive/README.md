# Naive CUDA Matrix Multiplication (`1.trivial.cu`)

## **Overview**

This project implements a **naive matrix multiplication** kernel in CUDA. Each thread computes a single element in the output matrix **C = A × B** without any memory optimizations. This implementation serves as a baseline for comparing optimized versions.

## **File Structure**

- `1.trivial.cu` - The CUDA program implementing naive matrix multiplication.
- `Makefile` (Optional) - Compilation instructions if using `make`.
- `README.md` - Documentation for this implementation.

## **Implementation Details**

### **Kernel Explanation**

The CUDA kernel **(`matMulNaive`)** assigns:

- **One thread per output matrix element**.
- **Each thread computes a dot product** of a row from **A** and a column from **B**.
- **Global memory** is used directly, making this inefficient for large matrices.

### **Grid & Block Setup**

- The kernel is launched with **2D grid and 2D blocks** for better thread organization.
- `BLOCK_SIZE = 16` (each block contains `16×16` threads).
- Grid dimensions are determined based on matrix size.

## **Compilation & Execution**

### **1. Compile the CUDA Program**

Run the following command:

```sh
nvcc 1.trivial.cu -o matmul
```

### **2. Run the Program**

```sh
./matmul
```

Expected output (for a simple test case):

```
C[0][0] = <computed value>
```

## **Profiling (Optional)**

To profile execution time:

```sh
nvprof ./matmul
```

To profile key performance metrics (if enabled):

```sh
nvprof --metrics achieved_occupancy,gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput,flop_count_sp,flop_sp_efficiency ./matmul
```

## **Performance Analysis**

This naive kernel **suffers from inefficiencies**, including:

1. **Global Memory Bottleneck** – Each thread loads matrix elements directly from slow global memory.
2. **Low Occupancy** – Inefficient use of available CUDA cores.
3. **Redundant Computation** – No shared memory is used to reuse data efficiently.

## **Next Steps: Optimizations**

For better performance, future optimizations may include:
✅ **Global memory coalescing**  
✅ **Shared memory tiling**  
✅ **Warp-level optimizations**  
✅ **Register reuse**  

These optimizations will be compared against this naive implementation.

## **License**

This project is open-source and intended for educational use.

## **Author**

- **S M Shovan**
- **Date: March 2025**
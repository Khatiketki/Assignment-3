# Assignment-3


## Performance Summary

| Kernel   | Time (ms) | Performance (TFLOPS) | Speedup vs Naive |
|----------|-----------|----------------------|------------------|
| Naive    | 275.62    | 0.50                 | 1x (baseline)    |
|SMEM 1D   | 6.93      | 19.82                | 40x              |
|2D Tiling | 5.29      | 25.99                | 52x              |
|Vectorized| 5.27      | 26.10                | 52x              |
|cuBLAS    | 2.65      | 51.92                | 104x             |

---

## Key Insights from Your Results

### 1. **Naive → SMEM (40x speedup!)**
- Went from 0.50 → 19.82 TFLOPS
- **Why?** Shared memory tiling dramatically reduces global memory traffic
- Each tile is loaded once from HBM but reused many times

### 2. **SMEM → 2D Tiling (1.3x speedup)**
- Went from 19.82 → 25.99 TFLOPS
- **Why?** Better register usage with 2D thread block tiles
- Each thread computes TM×TN results instead of just TM

### 3. **2D Tiling → Vectorized (minimal improvement)**
- Went from 25.99 → 26.10 TFLOPS
- **Why?** float4 vectorized loads improve memory coalescing slightly
- Already well-optimized, so gains are small

### 4. **Custom → cuBLAS (2x speedup)**
- Went from 26.10 → 51.92 TFLOPS
- **Why?** NVIDIA uses:
  - Tensor Cores (specialized matrix multiply hardware)
  - Advanced tiling strategies
  - Warp-level primitives
  - Years of optimization

---

## Memory Hierarchy Impact

Your results perfectly demonstrate the memory hierarchy optimization:

```
HBM (Global Memory)    →    SMEM (Shared Memory)    →    Registers
     ~3 TB/s                    ~20 TB/s                 ~30+ TB/s
     0.5 TFLOPS                 19.8 TFLOPS              26.1 TFLOPS
```

---

## H100 Performance Analysis

**Your cuBLAS achieved 51.92 TFLOPS**, which is:
- ~5% of H100's theoretical peak FP32 (1000 TFLOPS)
- This is actually **expected** because:
  - FP32 matmul doesn't use Tensor Cores efficiently
  - Tensor Cores are optimized for FP16/BF16/INT8
  - With Tensor Cores (FP16), you'd see 200-400+ TFLOPS

---

## Cost Summary

- **Time**: ~10 minutes
- **Estimated cost**: ~$0.70-$1.00 on Modal H100


###Output###:-
PS C:\Users\khati\Documents\Ai\week3> python -m modal run cuda_matmul_complete.py
✓ Initialized. View run at https://modal.com/apps/khatiketki/main/ap-ruaOpPwuB6HNEXgmhK4M5h
✓ Created objects.                                                                                                    
├── 🔨 Created mount C:\Users\khati\Documents\Ai\week3\cuda_matmul_complete.py                                        
├── 🔨 Created function run_kernel.                                                                                   
└── 🔨 Created function check_gpu_info.                                                                               

================================================================================
WEEK 3 ASSIGNMENT: CUDA MATRIX MULTIPLICATION ON H100
================================================================================

================================================================================
GPU INFORMATION
================================================================================

Thu Feb  5 03:39:50 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:D8:00.0 Off |                    0 |
| N/A   32C    P0            123W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0


================================================================================
RUNNING: 01_naive
================================================================================

Compiling...
✓ Compilation successful

Running kernel...
NAIVE KERNEL
  Time: 275.62 ms
  Performance: 0.50 TFLOPS
  Grid: (128, 128), Block: (32, 32)


================================================================================
RUNNING: 02_smem
================================================================================

Compiling...
✓ Compilation successful

Running kernel...
SHARED MEMORY 1D TILING
  Time: 6.93 ms
  Performance: 19.82 TFLOPS
  Tile: BM=64, BN=64, BK=8, TM=8


================================================================================
RUNNING: 03_2d_tiling
================================================================================

Compiling...
✓ Compilation successful

Running kernel...
2D BLOCK TILING
  Time: 5.29 ms
  Performance: 25.99 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8


================================================================================
RUNNING: 04_vectorized
================================================================================

Compiling...
✓ Compilation successful

Running kernel...
VECTORIZED LOADS (float4)
  Time: 5.27 ms
  Performance: 26.10 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8


================================================================================
RUNNING: 05_cublas
================================================================================

Compiling...
✓ Compilation successful

Running kernel...
cuBLAS (NVIDIA Optimized)
  Time: 2.65 ms
  Performance: 51.92 TFLOPS


================================================================================
FINAL RESULTS SUMMARY
================================================================================


01_naive:
  Status: success
  Time: 275.62 ms
  Performance: 0.50 TFLOPS

02_smem:
  Status: success
  Time: 6.93 ms
  Performance: 19.82 TFLOPS
  Tile: BM=64, BN=64, BK=8, TM=8

03_2d_tiling:
  Status: success
  Time: 5.29 ms
  Performance: 25.99 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

04_vectorized:
  Status: success
  Time: 5.27 ms
  Performance: 26.10 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

05_cublas:
  Status: success
  Time: 2.65 ms
  Performance: 51.92 TFLOPS

  Status: success
  Time: 5.29 ms
  Performance: 25.99 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

04_vectorized:
  Status: success
  Time: 5.27 ms
  Performance: 26.10 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

05_cublas:
  Status: success
  Time: 2.65 ms
  Performance: 51.92 TFLOPS

  Performance: 25.99 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

04_vectorized:
  Status: success
  Time: 5.27 ms
  Performance: 26.10 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

05_cublas:
  Status: success
  Time: 2.65 ms
  Performance: 51.92 TFLOPS

04_vectorized:
  Status: success
  Time: 5.27 ms
  Performance: 26.10 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

05_cublas:
  Status: success
  Time: 2.65 ms
  Performance: 51.92 TFLOPS

  Performance: 26.10 TFLOPS
  Tile: BM=128, BN=128, BK=8, TM=8, TN=8

05_cublas:
  Status: success
  Time: 2.65 ms
  Performance: 51.92 TFLOPS

================================================================================
ASSIGNMENT COMPLETE!
================================================================================

05_cublas:
  Status: success
  Time: 2.65 ms
  Performance: 51.92 TFLOPS

================================================================================
ASSIGNMENT COMPLETE!
================================================================================

================================================================================
ASSIGNMENT COMPLETE!
================================================================================

KEY INSIGHTS:
- Naive: Memory bandwidth limited (~0.5 TFLOPS)
- SMEM: Data reuse through tiling (~10-20x speedup)
- 2D Tiling: Better register usage (~3-5x over SMEM)
- Vectorized: Coalesced access (~5-10% improvement)
- cuBLAS: Highly optimized, near-peak performance
Stopping app - local entrypoint completed.
KEY INSIGHTS:
- Naive: Memory bandwidth limited (~0.5 TFLOPS)
- SMEM: Data reuse through tiling (~10-20x speedup)
- 2D Tiling: Better register usage (~3-5x over SMEM)
- Vectorized: Coalesced access (~5-10% improvement)
- cuBLAS: Highly optimized, near-peak performance
Stopping app - local entrypoint completed.
- 2D Tiling: Better register usage (~3-5x over SMEM)
- Vectorized: Coalesced access (~5-10% improvement)
- cuBLAS: Highly optimized, near-peak performance
Stopping app - local entrypoint completed.
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/khatiketki/main/ap-ruaOpPwuB6HNEXgmhK4M5h
PS C:\Users\khati\Documents\Ai\week3>

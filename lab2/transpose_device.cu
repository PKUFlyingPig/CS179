#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses
    
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // output[j + n * i] : not coalesced, it touches n cache lines
    // input[i + n * j] : coalesced
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    __shared__ float temp[64][65]; // pad the shared memory to avoid bank conflicts

    const int in_i = threadIdx.x + 64 * blockIdx.x;
    int in_j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int in_end_j = in_j + 4;

    for (; in_j < in_end_j; in_j++)
        temp[in_i%64][in_j%64] = input[in_i + n * in_j];
    
    __syncthreads();

    const int out_i = threadIdx.x + 64 * blockIdx.y;
    int out_j = 4 * threadIdx.y + 64 * blockIdx.x;
    const int out_end_j = out_j + 4;

    for (; out_j < out_end_j; out_j++)
        output[out_i + n * out_j] = temp[out_j%64][out_i%64];
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.
    __shared__ float temp[64][65]; // pad the shared memory to avoid bank conflicts

    const int in_i = threadIdx.x + 64 * blockIdx.x;
    int in_j = 4 * threadIdx.y + 64 * blockIdx.y;

    // unroll && ILP
    int in_ii = in_i%64;
    int in_jj = in_j%64;
    float var0 = input[in_i + n * in_j];
    float var1 = input[in_i + n * (in_j + 1)];
    float var2 = input[in_i + n * (in_j + 2)];
    float var3 = input[in_i + n * (in_j + 3)];

    temp[in_ii][in_jj] = var0;
    temp[in_ii][in_jj+1] = var1;
    temp[in_ii][in_jj+2] = var2;
    temp[in_ii][in_jj+3] = var3;

    __syncthreads();

    const int out_i = threadIdx.x + 64 * blockIdx.y;
    int out_j = 4 * threadIdx.y + 64 * blockIdx.x;

    // unroll
    int out_ii = out_i%64;
    int out_jj = out_j%64;
    output[out_i + n * out_j] = temp[out_jj][out_ii];
    output[out_i + n * (out_j + 1)] = temp[out_jj + 1][out_ii];
    output[out_i + n * (out_j + 2)] = temp[out_jj + 2][out_ii];
    output[out_i + n * (out_j + 3)] = temp[out_jj + 3][out_ii];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}

/* 
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#include "blur.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_header.cuh"

/*
 * NOTE: You can use this macro to easily check cuda error codes 
 * and get more information. 
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *   what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

CUDA_CALLABLE
void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    // TODO: Implement the necessary convolution function that should be
    //       completed for each thread_index. Use the CPU implementation in
    //       blur.cpp as a reference.
    if (thread_index < blur_v_size) {
        for (int j = 0; j <= thread_index; j++) {
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - j] * gpu_blur_v[j];
        }
    } else {
        for (int j = 0; j < blur_v_size; j++) {
            gpu_out_data[thread_index] += gpu_raw_data[thread_index - j] * gpu_blur_v[j];
        }
    }
}

__global__
void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    // TODO: Compute the current thread index.
    uint thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    uint total_threads = gridDim.x * blockDim.x;


    // TODO: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.
    while (thread_index < n_frames) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        // TODO: Update the thread index
        thread_index += total_threads;
    }
}


float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    // TODO: Allocate GPU memory for the raw input data (either audio file
    //       data or randomly generated data. The data is of type float and
    //       has n_frames elements. Then copy the data in raw_data into the
    //       GPU memory you allocated.
    float* gpu_raw_data;
    gpu_errchk(cudaMalloc(&gpu_raw_data, sizeof(float) * n_frames));
    gpu_errchk(cudaMemcpy(gpu_raw_data, raw_data, sizeof(float) * n_frames, cudaMemcpyHostToDevice));

    // TODO: Allocate GPU memory for the impulse signal (for now global GPU
    //       memory is fine. The data is of type float and has blur_v_size
    //       elements. Then copy the data in blur_v into the GPU memory you
    //       allocated.
    float* gpu_blur_v;
    gpu_errchk(cudaMalloc(&gpu_blur_v, sizeof(float) * blur_v_size));
    gpu_errchk(cudaMemcpy(gpu_blur_v, blur_v, sizeof(float) * blur_v_size, cudaMemcpyHostToDevice));

    // TODO: Allocate GPU memory to store the output audio signal after the
    //       convolution. The data is of type float and has n_frames elements.
    //       Initialize the data as necessary.
    float* gpu_out_data;
    gpu_errchk(cudaMalloc(&gpu_out_data, sizeof(float) * n_frames));
    
    // TODO: Appropriately call the kernel function.
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

    // TODO: Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory. (We store this channel's result
    //       in out_data on the host.)
    gpu_errchk(cudaMemcpy(out_data, gpu_out_data, sizeof(float) * n_frames, cudaMemcpyDeviceToHost));

    // TODO: Now that we have finished our computations on the GPU, free the
    //       GPU resources.
    gpu_errchk(cudaFree(gpu_raw_data));
    gpu_errchk(cudaFree(gpu_blur_v));
    gpu_errchk(cudaFree(gpu_out_data));

    // Stop the recording timer and return the computation time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
    return time_milli;
}

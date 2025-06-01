#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void kernel(int x, int y, char* A) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 && idx <= y && idx % x == 0) {
        A[idx] = 1;
    }
}

int main(int argc, char** argv) {
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);

    // Allocate an array of char of y+1 elements in the host and initialize this array to 0.
    char* h_array = (char*)malloc((y + 1) * sizeof(char));
    for (int i = 0; i <= y; ++i) {
        h_array[i] = 0;
    }

    // Allocate a similar array in the device
    char* d_array;
    cudaMalloc(&d_array, (y + 1) * sizeof(char));
    cudaMemcpy(d_array, h_array, (y + 1) * sizeof(char), cudaMemcpyHostToDevice);

    // Blocks and threads
    const int threads_per_block = 256;
    const int number_of_blocks = (y + threads_per_block - 1) / threads_per_block;

    // Each thread in every block will be assigned one or more entries of the array to update + the next step
    kernel<<<number_of_blocks, threads_per_block>>>(x, y, d_array);
    cudaDeviceSynchronize();

    // Once done, the array is copied back to the host
    cudaMemcpy(h_array, d_array, (y + 1) * sizeof(char), cudaMemcpyDeviceToHost);

    // The host counts how many 1s in the array and prints the result on the screen as shown earlier
    int count = 0;
    for (int i = 2; i <= y; ++i) {
        if (h_array[i]) ++count;
    }
    printf(
        "There are %d numbers divisible by %d in the range [2, %d].\n"
        "Number of blocks used is %d\n"
        "Number of threads per block is %d\n",
        count, x, y, number_of_blocks, threads_per_block
    );

    // Free the allocated arrays both in host and device
    cudaFree(d_array);
    free(h_array);
    return 0;
}

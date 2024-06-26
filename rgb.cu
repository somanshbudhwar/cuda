#include <cuda_runtime.h>
#include <stdio.h>


__global__ void initializeArray(int *d_array, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        d_array[y * width + x] = y * width + x;  // Example initialization
    }
}

__global__ void blurArray(int *d_array, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixels=0;
        int total = 0;
        if (col-1>=0){
            pixels++;
            total = total+d_array[row*width + col-1];
        }
        if (col+1<width){
            pixels++;
            total = total+d_array[row*width + col+1];
        }
        d_array[row*width + col] += total/(pixels+1);
    }
    __syncthreads();
}

int main() {
    int N = 64;
    const int width = 16;
    const int height = 16;
    int arraySize = width * height * sizeof(int);
    int h_array[width][height];

    int *d_array;
    cudaMalloc((void **)&d_array, arraySize);

    dim3 blockSize(8, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    initializeArray<<<gridSize, blockSize>>>(d_array, width, height);
    blurArray<<<gridSize, blockSize>>>(d_array, width, height);

    cudaMemcpy(h_array, d_array, arraySize, cudaMemcpyDeviceToHost);

    cudaFree(d_array);

    // Print the initialized array for verification
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%4d ", h_array[y][x]);
        }
        printf("\n");
    }

    return 0;
}

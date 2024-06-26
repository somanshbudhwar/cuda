#include <iostream>
#include <stdio.h>

__device__
void vecSum(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<n) {
        A[i] = float(i);
        B[i] = float(i);
        C[i] = 0.0;
    }
}


__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
//    Initialize values of A, B, and C
    vecSum(A, B, C, n);

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
        }
    }


// Vector Addition
void vecAdd(float* A_h, float* B_h, float* C_h, int n){
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

void cuda_vecAdd(float* A_h, float* B_h, float* C_h, int n){
    int size = n*sizeof(float);
    float *d_A, *d_B, *d_C;

    // Part 1: Allocate device memory for A, B, and C
//    cudaError_t err = cudaMalloc((void**) &d_A, size);
//    if (err!= cudaSuccess) {
//        printf("%s in %s at line %d\n", cudaGetErrorString(err),
//                __FILE__, __LINE__);
//        exit(EXIT_FAILURE);
//    }
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    // Copy A and B to device memory
    cudaMemcpy(d_A, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_h, size, cudaMemcpyHostToDevice);


    // Part 2: Call kernel – to launch a grid of threads
    // to perform the actual vector addition
    dim3 blocks(ceil(n/256.0));
    dim3 threadCount(256);
    vecAddKernel<<<blocks, threadCount>>>(d_A, d_B, d_C, size);

    // Part 3: Copy C from the device memory
    cudaMemcpy(C_h, d_C, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main() {
    const int size_n = 512;
    float arr[size_n];
    float arr2[size_n];
    float arr3[size_n];

//    for (int i = 0; i < size_n; i++) {
//        arr[i] = float(i);
//        arr2[i] = float(i);
//    }

    cuda_vecAdd(arr, arr2, arr3, size_n);
    printf("Size = %d\n", size_n);
    float *P = &arr3[0];

    for (int i = 0; i < 10; i++) {
        printf("Element %d : %f\n", i, P[i]);
    }

    return 0;
}

#include <cuda_runtime.h>
#include <stdio.h>

// My first ever Kernel
// __global__
//     void initializeArray(int *d_array){
//   int location = blockDim.x*blockIdx.x + threadIdx.x;
//   d_array[location] = location;
//     }
constexpr int NN = 512*32;

// My second ever Kernel
    __global__
        void turnItUp(int *d_array, int multiplier){
      int location = blockDim.x*blockIdx.x + threadIdx.x;

      if (location<NN){
        // d_array[location] = d_array[location]+multiplier;
          if (location>=0){d_array[location]=d_array[location-1];}
      }
       __syncthreads();
      }

// Let me define the main funciton first
int main(){
//  I want to multiply each element in an array by the given number
    const int N = NN;
    int *d_array;

    int multiplier = 2;

    dim3 gridSize(32,1,1);
    dim3 blockSize(ceil(N/32 ),1,1);
    cudaMalloc((void**)&d_array, N*sizeof(int));
    // initializeArray<<<gridSize,blockSize>>>(d_array);
    int h_array[N];
    for (int l=0; l<N;l++) {
        h_array[l]= l;
    }

    // for (int i=0; i<N; i++){
    //         printf("%d ",h_array[i]);
    //     }

    cudaMemcpy(d_array, h_array, N*sizeof(int), cudaMemcpyHostToDevice);


    printf("\n");
    for (int i=0; i<512*32-2; i++){
        turnItUp<<<gridSize,blockSize>>>(d_array, multiplier);
    }

    cudaMemcpy(h_array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0; i<N; i++){
        printf("%d ",h_array[i]);
    }
    printf("\n");
    cudaFree(d_array);
    return 0;


}
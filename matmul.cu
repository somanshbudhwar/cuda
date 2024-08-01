#include <stdio.h>

#define TILE_WIDTH 32

__global__ void matMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over M and N tiles to compute P element

    float Pvalue = 0;

    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k =0; k< TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row*Width + Col] = Pvalue;

}

// Write a main function to test the above code
int main() {
    const int Width = 5;
    float M[Width][Width], N[Width][Width], P[Width][Width];
    float *Md, *Nd, *Pd;
    int size = Width * Width * sizeof(float);

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            M[i][j] = 1.0;
            N[i][j] = 2.0;
        }
    }

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%f ", N[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    cudaMalloc((void **) &Md, size);
    cudaMalloc((void **) &Nd, size);
    cudaMalloc((void **) &Pd, size);

    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);



    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%f ", P[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Done\n");

    return 0;
}
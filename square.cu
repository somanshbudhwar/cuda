#include<stdio.h>
#include<cuda_runtime.h>
#include<math.h>

const int N = 64;

__global__
void rotate(char *d_array, char *d_out_array, float angle){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < N && col < N){
        int shifted_row = row - int(N/2);
        int shifted_col = col - int(N/2);

        int newRow = (int)roundf(shifted_row*cos(angle*3.141/180) - shifted_col*sin(angle*3.141/180)) + int(N/2);
        int newCol = (int)roundf(shifted_row*sin(angle*3.141/180) + shifted_col*cos(angle*3.141/180)) + int(N/2);

        if (newRow >= 0 && newRow < N && newCol >= 0 && newCol < N){
            d_out_array[newRow*N + newCol] = d_array[row*N + col];
        }
    }
}

int main(){
    int height = N;
    int width = N;
    int arraySize = width*height*sizeof(char);

//    char h_array[16][16] = {
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ','#','#','#','#',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ','e','#','#','#','#','e',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ','e','#','#','#','#','e',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ','#','#','#','#',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//        {' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '},
//    };

    char h_array[N][N];

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            h_array[i][j] = ' ';
        }
    }

    for(int i=0; i<int(N/4); i++){
        for(int j=0; j<int(N/4); j++){
                    h_array[i+int(N/4)][j+int(N/4)] = '#';
        }
    }

    printf("Original array:\n");
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            printf("%c",h_array[i][j]);
        }
        printf("\n");
    }
    printf("\n**********************************\n");

    dim3 gridSize = dim3(ceil(N/16),ceil(N/16));
    dim3 blockSize = dim3(16,16);
    char *d_array, *d_out_array;
    float angle = 25.0;

    cudaMalloc((void**)&d_array, arraySize);
    cudaMalloc((void**)&d_out_array, arraySize);

    cudaMemset(d_out_array, ' ', arraySize);

    cudaMemcpy(d_array, h_array, arraySize, cudaMemcpyHostToDevice);

    rotate<<<gridSize,blockSize>>>(d_array, d_out_array, angle);

    cudaMemcpy(h_array, d_out_array, arraySize, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_out_array);


    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            printf("%c",h_array[i][j]);
        }
        printf("\n");
    }

    return 0;
}
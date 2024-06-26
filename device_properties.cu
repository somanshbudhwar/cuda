#include <stdio.h>
int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("%d\n",devCount);

    cudaDeviceProp devProp;
    for(unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf(" Threads per block %d\n",devProp.maxThreadsPerBlock);
        printf(" MultiProcessor count %d\n",devProp.multiProcessorCount);
        printf(" Shared Memory per block %zu\n",devProp.sharedMemPerBlock);
        printf(" Total Global Memory %zu\n",devProp.totalGlobalMem);
        printf(" Total Constant Memory %zu\n",devProp.totalConstMem);

        printf(" Max block size %d\n",devProp.maxThreadsPerBlock);

        printf(" Clock Rate: %d\n",devProp.clockRate);
        printf(" Warp size %d\n",devProp.warpSize);
        printf(" Registers per block %d\n",devProp.regsPerBlock);

        printf(" Max threads along x %d\n",devProp.maxThreadsDim[0]);
        printf(" Max threads along y %d\n",devProp.maxThreadsDim[1]);
        printf(" Max threads along z %d\n",devProp.maxThreadsDim[2]);

        printf(" Max grid size along x %d\n",devProp.maxGridSize[0]);
        printf(" Max grid size along y %d\n",devProp.maxGridSize[1]);
        printf(" Max grid size along z %d\n",devProp.maxGridSize[2]);

        printf("%d\n",devProp.maxBlocksPerMultiProcessor);
        printf("%d\n",devProp.maxThreadsPerMultiProcessor);
    }
}
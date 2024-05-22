#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "bitonicSort.h"


int main(int argc,char* argv[]) {
    if(argc != 2) {
        fprintf(stderr,"Invalid number of arguments");
        return 1;
    }
    // set up device
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice (dev)); // a gpu que vou utilizar

    int K = 1024; // number of gpu threads

    char* filePath = argv[1];
    FILE* file = fopen(filePath,"rb");
    int N;
    fread(&N,sizeof(int),1,file);

    int nBytes = N * sizeof(int);
    int* h_c = (int*)malloc(nBytes);
    fread(h_c,sizeof(int),N,file);
    fclose(file);

    int numIters = log2(N);

    int* d_c;
    CHECK(cudaMalloc((void **)&d_c, nBytes));
    CHECK(cudaMemcpy(d_c, h_c,nBytes, cudaMemcpyHostToDevice));

    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    blockDimX = 1 << 0;    // optimize! // 1 thread
    blockDimY = 1;         // optimize!
    blockDimZ = 1;         // do not change!
    gridDimX = nElem >> 0; // optimize!
    gridDimY = 1;          // optimize!
    gridDimZ = 1;          // do not change!

    dim3 grid(gridDimX, gridDimY, gridDimZ);
    dim3 block(blockDimX, blockDimY, blockDimZ);

    if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != nElem)
    {
        printf("Wrong configuration!\n");
        return 1;
    }

    (void)get_delta_time();
    for(int iter = 0; iter < numIters; iter++) {
        sort_gpu<<<grid,block>>>(d_c,N,k,iter);
        CHECK(cudaDeviceSynchronize()); // wait for kernel to finish - aguarda que o gpu acabe de executar
        CHECK(cudaGetLastError());      // check for kernel errors // por sempre
    }
    printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time());

    CHECK (cudaMemcpy (h_c, d_c, nBytes, cudaMemcpyDeviceToHost));
    CHECK (cudaFree (d_c)); //gpu
    CHECK (cudaDeviceReset ());

    int i = 0;
    for (i = 0; i < N - 1; i++)
        if (val[i] < val[i + 1])
        {
            printf("Error in position %d between element %d and %d\n", i, val[i], val[i + 1]);
            break;
        }
    if (i == (N - 1))
        printf("Everything is OK!\n");

    free (h_c);
    return 0;
}

__global__ static void sort_gpu(int* seq, int N,int K,int iter) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = gridDim.x * blockDim.x * y + x;

    if(idx > (K >> iter))
        return;
    
    int* subseq = seq + N/K * (1 << iter) * idx;
    bitonic_sort(subseq,0,N/K * (1 << iter),0);
}
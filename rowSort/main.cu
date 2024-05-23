#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// Function to compare and swap elements in the bitonic merge
__device__ void compareAndSwap(int *arr, int i, int j, int dir) {
    if (dir == (arr[i] > arr[j])) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// Function to perform bitonic merge
__device__ void bitonicMerge(int *arr, int low, int count, int dir) {
    if (count > 1) {
        int k = count / 2;
        for (int i = low; i < low + k; i++) {
            compareAndSwap(arr, i, i + k, dir); // Always compare in the same direction
        }
        bitonicMerge(arr, low, k, dir); // Merge first half
        bitonicMerge(arr, low + k, k, dir); // Merge second half
    }
}

// Function to perform bitonic sort
__global__ void bitonicSort(int *seq, int N, int K, int iters,int dir) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = gridDim.x * blockDim.x * y + x;

    int _dir = dir;
    for(int iter = 0; iter < iters; iter++) {
        int limit = (K >> iter);
        if (idx >= limit)
            return;

        int size = (N / K) * (1 << iter);
        printf("size of sequence: %d\n",size);
        int* subseq = seq + size * idx;

        _dir = (idx % 2 == 0) ? dir : (dir ^ 1);
        int mid = size / 2;
        for (int i = 0; i < mid; i++) {
            compareAndSwap(subseq, i, i + mid, _dir);
        }
        bitonicMerge(subseq, 0, size, _dir);
        __syncthreads();
    }
}

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

    int threadsPerBlock = 32; // Number of threads per block
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed

    // (void)get_delta_time();
    bitonicSort<<<numBlocks, threadsPerBlock>>>(d_c, N, K, numIters, 0);
    CHECK(cudaDeviceSynchronize()); // wait for kernel to finish - aguarda que o gpu acabe de executar
    CHECK(cudaGetLastError());      // check for kernel errors // por sempre
    // printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
    //        gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time());

    CHECK (cudaMemcpy (h_c, d_c, nBytes, cudaMemcpyDeviceToHost));
    CHECK (cudaFree (d_c)); //gpu
    CHECK (cudaDeviceReset ());

    int i = 0;
    for (i = 0; i < N - 1; i++)
        if (h_c[i] < h_c[i + 1])
        {
            printf("Error in position %d between element %d and %d\n", i, h_c[i], h_c[i + 1]);
            break;
        }
    if (i == (N - 1))
        printf("Everything is OK!\n");

    free (h_c);
    return 0;
}


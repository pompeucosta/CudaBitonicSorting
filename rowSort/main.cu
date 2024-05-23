#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

/**
* \brief Get the process time that has elapsed since last call of this time.
*
* \return process elapsed time
*/
static double get_delta_time(void)
{
    static struct timespec t0, t1;
    t0 = t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
    {
        perror("clock_gettime");
        exit(1);
    }
    return (double)(t1.tv_sec - t0.tv_sec) +
           1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}

__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void sort_desc(int* arr,int n) {
    for (int k = 2; k <= n; k <<= 1) 
        for (int j = (k >> 1); j > 0; j >>= 1) 
            for (int i = 0; i < n; i++)
            {
                int l = i ^ j;
                if(l > i)
                    if ((((i & k) == 0) && (arr[i] < arr[l])) || (((i & k) != 0) && (arr[i] > arr[l]))) {
                        swap(&arr[i], &arr[l]);
                    }
            }
}

__device__ void sort_asc(int* arr, int n) {
    for (int k = 2; k <= n; k <<= 1)
        for (int j = (k >> 1); j > 0; j >>= 1)
            for (int i = 0; i < n; i++)
            {
                int l = i ^ j;
                if (l > i)
                    if ((((i & k) == 0) && (arr[i] > arr[l])) || (((i & k) != 0) && (arr[i] < arr[l]))) {
                        swap(&arr[i], &arr[l]);
                    }
            }
}

__global__ void bitonicSort(int *seq, int N, int K, int iters,int dir) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = gridDim.x * blockDim.x * y + x;

    int _dir = dir;
    int iter = 0;
    int size = (N / K) * (1 << iter);

    for(int iter = 0; size <= N; iter++) {
        int limit = (K >> iter);
        if (idx >= limit)
            return;

        size = (N / K) * (1 << iter);
        int* subseq = seq + size * idx;

        _dir = (idx % 2 == 0) ? dir : (dir ^ 1);
        if(_dir == 0) {
            sort_asc(subseq,size);
        }
        else {
            sort_desc(subseq,size);
        }

        __syncthreads();
    }

}

int main(int argc,char* argv[]) {
    if(argc != 3) {
        fprintf(stderr,"Invalid number of arguments");
        return 1;
    }
    // set up device
    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice (dev)); // a gpu que vou utilizar

    int K = atoi(argv[2]); // number of gpu threads

    char* filePath = argv[1];
    FILE* file = fopen(filePath,"rb");
    int N;
    fread(&N,sizeof(int),1,file);

    int nBytes = N * sizeof(int);
    int* h_c = (int*)malloc(nBytes);
    fread(h_c,sizeof(int),N,file);
    fclose(file);

    int numIters = log2(N);
    printf("numIters: %d\n",numIters);

    int* d_c;
    CHECK(cudaMalloc((void **)&d_c, nBytes));
    CHECK(cudaMemcpy(d_c, h_c,nBytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 32; // Number of threads per block
    int numBlocks = (K + threadsPerBlock) / threadsPerBlock;
    // int numBlocks = (N + K - 1) / K; // Calculate number of blocks needed

    (void)get_delta_time();
    bitonicSort<<<numBlocks,threadsPerBlock>>>(d_c, N, K, numIters, 1);
    CHECK(cudaDeviceSynchronize()); // wait for kernel to finish - aguarda que o gpu acabe de executar
    CHECK(cudaGetLastError());      // check for kernel errors // por sempre
    printf("time elapsed %.5f\n",get_delta_time());

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


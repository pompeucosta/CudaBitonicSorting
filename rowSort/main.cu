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

__device__ void iterative_bitonic_sort(int* arr, int n,int dir,int initialK) {
    for (int k = initialK; k <= n; k <<= 1) {
        for (int j = k; j > 1; j >>= 1) {
            int z = 0;
            for (int i = 0; i < n/j; i ++) {
                if (z >= k) {
                    z = 0;
                    dir ^= 1;
                }

                for (int x = i * j; x < i * j + (j / 2); x++) {
                    z += 2;
                    if (dir == (arr[x] > arr[x + (j / 2)])) {
                        swap(&arr[x], &arr[x + j / 2]);
                    }
                }
            }
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
    int initialK = 2;

    for(int iter = 0; size <= N; iter++) {
        int limit = (K >> iter);
        if (idx >= limit)
            return;

        size = (N / K) * (1 << iter);
        int* subseq = seq + size * idx;

        _dir = (idx % 2 == 0) ? dir : (dir ^ 1);
        iterative_bitonic_sort(subseq,size,_dir,initialK);
        initialK = size << 1;
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

    (void)get_delta_time();
    bitonicSort<<<1,K>>>(d_c, N, K, numIters, 0);
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


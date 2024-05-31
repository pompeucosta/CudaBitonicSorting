/**
 * Pompeu Costa and Guilherme Craveiro,May 2024
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

/* allusion to internal functions */

static double get_delta_time(void);

static __device__ void iterative_bitonic_sort(int* arr, int n,int dir,int initialK);

static __global__ void bitonicSort(int *seq, int N, int K, int dir);

/**
 *   main program
 */

int main(int argc,char* argv[]) {
    if(argc != 3) {
        fprintf(stderr,"Invalid number of arguments");
        return 1;
    }

    /* set up the device */

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

    int* d_c;
    CHECK(cudaMalloc((void **)&d_c, nBytes));
    CHECK(cudaMemcpy(d_c, h_c,nBytes, cudaMemcpyHostToDevice));

    (void)get_delta_time();
    bitonicSort<<<1,K>>>(d_c, N, K, 0);
    CHECK(cudaDeviceSynchronize()); // wait for kernel to finish - aguarda que o gpu acabe de executar
    CHECK(cudaGetLastError());      // check for kernel errors // por sempre
    printf("time elapsed %.5f\n",get_delta_time());

    /* free device global memory */

    CHECK (cudaMemcpy (h_c, d_c, nBytes, cudaMemcpyDeviceToHost));
    CHECK (cudaFree (d_c)); //gpu

    /* reset the device */

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

static __device__ void iterative_bitonic_sort(int* arr, int n,int dir,int initialK) {
    for (int k = initialK; k <= n; k <<= 1) {
    for (int j = k; j > 1; j >>= 1) {
        int jDiv2 = j >> 1;
        for (int i = 0,z = 0; i < n; i+= j) {
            if (z >= k) {
                z = 0;
                dir ^= 1;
            }
            
            for (int x = i; x < i + jDiv2; x++, z+= 2) {
                if (dir == (arr[x] > arr[x + jDiv2])) {
                    int temp = arr[x];
                    arr[x] = arr[x + jDiv2];
                    arr[x + jDiv2] = temp;
                }
            }
        }
    }
}
}

/**
 * \brief Perform bitonic sort on the array.
 *
 * \param seq The array to be sorted
 * \param N The total number of elements in the array
 * \param K The number of rows
 * \param dir The sorting direction (1 for ascending, 0 for descending)
 */
static __global__ void bitonicSort(int *seq, int N, int K, int dir) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = gridDim.x * blockDim.x * y + x;

    int* subseq;
    const int _dir = ((idx & 1) == 0) ? dir : (dir ^ 1);
    const int nDivK = N / K;
    for(int iter = 0,size = nDivK,initialK = 2; size <= N; iter++) {
        if (idx >= (K >> iter))
            return;

        size = nDivK * (1 << iter);
        subseq = seq + size * idx;

        iterative_bitonic_sort(subseq,size,_dir,initialK);
        initialK = size << 1;
        __syncthreads();
    }

}
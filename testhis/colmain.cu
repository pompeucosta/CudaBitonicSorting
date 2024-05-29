#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

/* allusion to internal functions */

static double get_delta_time(void);

static __device__ void iterative_bitonic_sort_column(int* arr, int n, int ns, int logNs, int dir);

static __global__ void bitonicSort(int *seq, int N, int Ns, int logNs, int K, int dir);

/**
 *   main program
 */

int main(int argc, char* argv[]) {
    if(argc != 3) {
        fprintf(stderr,"Invalid number of arguments\n");
        return 1;
    }

    /* set up the device */

    int dev = 0;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int K = atoi(argv[2]);

    char* filePath = argv[1];
    FILE* file = fopen(filePath, "rb");
    int N;
    fread(&N, sizeof(int), 1, file);

    int nBytes = N * sizeof(int);
    int Ns = sqrt(N);
    int* h_c = (int*)malloc(nBytes);

    fread(h_c, sizeof(int), N, file);
    fclose(file);

    int* d_c;
    CHECK(cudaMalloc((void **)&d_c, nBytes));
    CHECK(cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice));

    int logNs = log2(Ns);

    (void)get_delta_time();
    bitonicSort<<<1, K>>>(d_c, N, Ns, logNs, K, 1);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());  
    printf("time elapsed %.5f\n", get_delta_time());

    /* free device global memory */

    CHECK(cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaFree(d_c));

    /* reset the device */

    CHECK(cudaDeviceReset());

    int i = 0;
    for (i = 0; i < N - 1; i++)
        if (h_c[Ns * (i % Ns) + (i / Ns)] < h_c[Ns * ((i + 1) % Ns) + ((i + 1) / Ns)]) {
            printf("Error in position %d between element %d and %d\n", i, h_c[Ns * (i % Ns) + (i / Ns)], h_c[Ns * ((i + 1) % Ns) + ((i + 1) / Ns)]);
            break;
        }
    if (i == (N - 1))
        printf("Everything is OK!\n");

    free(h_c);
    return 0;
}

/**
* \brief Get the process time that has elapsed since last call of this time.
*
* \return process elapsed time
*/
static double get_delta_time(void) {
    static struct timespec t0, t1;
    t0 = t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double)(t1.tv_sec - t0.tv_sec) +
           1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}

/**
 * \brief Perform an iterative bitonic sort on a column.
 *
 * \param arr The array to be sorted
 * \param n The total number of elements in the array
 * \param ns The size of a column
 * \param logNs The log2 value of ns
 * \param dir The sorting direction (1 for ascending, 0 for descending)
 */
static __device__ void iterative_bitonic_sort_column(int* arr, int n, int ns, int logNs, int dir) {
    extern __shared__ int shared[];
    int tx = threadIdx.x;

    // Load data into shared memory
    for (int i = tx; i < n; i += blockDim.x) {
        shared[i] = arr[i];
    }
    __syncthreads();

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tx; i < n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if ((dir == 1) == (shared[i] > shared[ixj])) {
                            int temp = shared[i];
                            shared[i] = shared[ixj];
                            shared[ixj] = temp;
                        }
                    } else {
                        if ((dir == 1) == (shared[i] < shared[ixj])) {
                            int temp = shared[i];
                            shared[i] = shared[ixj];
                            shared[ixj] = temp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write back the results to global memory
    for (int i = tx; i < n; i += blockDim.x) {
        arr[i] = shared[i];
    }
}

/**
 * \brief Perform bitonic sort on the array.
 *
 * \param seq The array to be sorted
 * \param N The total number of elements in the array
 * \param Ns The size of a column
 * \param logNs The log2 value of ns
 * \param K The number of columns
 * \param dir The sorting direction (1 for ascending, 0 for descending)
 */
static __global__ void bitonicSort(int *seq, int N, int Ns, int logNs, int K, int dir) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = gridDim.x * blockDim.x * y + x;

    if (idx >= Ns)
        return;

    int nDivK = N / K;
    int nsDivK = Ns / K;
    int* subseq = seq + nsDivK * idx;

    for (int iter = 0, size = nDivK, initialK = 2; size <= N; iter++) {
        int limit = (K >> iter);
        if (idx >= limit)
            return;

        size = nDivK * (1 << iter);
        iterative_bitonic_sort_column(subseq, size, Ns, logNs, dir);
        initialK = size << 1;
        __syncthreads();
    }
}
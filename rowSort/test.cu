#include <stdio.h>
#include <cuda_runtime.h>

// Function to compare and swap elements in the bitonic merge
__device__ void compareAndSwap(int *arr, int i, int j, int dir) {
    if (dir == (arr[i] > arr[j])) {
        printf("swapping %d with %d\n",arr[i],arr[j]);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    else {
        printf("didnt swap %d with %d\n",arr[i],arr[j]);
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

__device__ void printArrayGPU(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
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
        printf("merging\n");
        bitonicMerge(subseq, 0, size, _dir);
        __syncthreads();
    }
}

// Function to print an array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Main function to test the bitonic sort
int main() {
    int arr[] = {3, 7, 4, 8, 6, 2, 1, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    int *d_arr;

    printf("Original array: \n");
    printArray(arr, n);

    // Allocate memory on device
    cudaMalloc(&d_arr, n * sizeof(int));
    // Copy array from host to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int k = 4; // Number of threads per block
    int numBlocks = (n + k - 1) / k; // Calculate number of blocks needed

    // Determine number of iterations for bitonic sort
    int iterations = 0;
    for (int temp = n; temp > 1; temp >>= 1)
        iterations++;

    bitonicSort<<<numBlocks, k>>>(d_arr, n, k, iterations,0);

    // Copy sorted array from device to host
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array: \n");
    printArray(arr, n);

    // Free device memory
    cudaFree(d_arr);

    return 0;
}

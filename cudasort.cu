#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#define THREADS 512
#ifdef __cplusplus
extern "C"
{
#endif


__global__ void bitonic_sort(float *arr, int i, int j)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("\nidx : %d dim: %d th: %d",blockIdx.x, blockDim.x, threadIdx.x);
  int k = index ^ j;
  int l = index & i;
  float temp;
  if (k > index) {    
    if (((l==0) && (arr[index]>arr[k])) || ((l!=0) && (arr[index]<arr[k]))) {
      temp = arr[index];
      arr[index] = arr[k];
      arr[k] = temp;
    }
  }
}


int cuda_sort(int number_of_elements, float *a)
{
  
  float *arr;
  
  cudaMalloc((void**) &arr, number_of_elements * sizeof(float));
  cudaMemcpy(arr, a, number_of_elements * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 dimGrid(number_of_elements/THREADS);
  dim3 dimBlock(THREADS);

  for (int i = 2; i <= number_of_elements; i*=2) {
    int j = i/2;
    while (j > 0){
      bitonic_sort<<<dimGrid, dimBlock>>>(arr, i, j);
      j/=2;
    }
  }
  cudaMemcpy(a, arr, number_of_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);

  return 0;
}

#ifdef __cplusplus
}
#endif


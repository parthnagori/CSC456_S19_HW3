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


__global__ void bitonic_sort(float *arr, int k, int j)
{
  unsigned int i, ij; 
    i = threadIdx.x + blockDim.x * blockIdx.x;
  ij = i^j;

  if ((ij)>i) {
    if ((i&k)==0) {
      if (arr[i]>arr[ij]) {
        float temp = arr[i];
        arr[i] = arr[ij];
        arr[ij] = temp;
      }
    }
    if ((i&k)!=0) {
      if (arr[i]<arr[ij]) {
        float temp = arr[i];
        arr[i] = arr[ij];
        arr[ij] = temp;
      }
    }
  }
}


int cuda_sort(int number_of_elements, float *a)
{
  
  float *arr;
  
  cudaMalloc((void**) &arr, number_of_elements * sizeof(float));
  cudaMemcpy(arr, a, number_of_elements * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 dimGrid(number_of_elements/512,1);
  dim3 dimBlock(512,1);

  for (int i = 2; i <= number_of_elements; i <<= 1) {
    for (int j= i>>1 ; j>0; j = j>>1) {
      bitonic_sort<<<dimGrid, dimBlock>>>(arr, i, j);
    }
  }
  cudaMemcpy(a, arr, number_of_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);

  return 0;
}

#ifdef __cplusplus
}
#endif


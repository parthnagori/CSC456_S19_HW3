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


__global__ void bitonic_sort_step(float *arr, int j, int k)
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
  
  int thread_cnt = 0;
  int block_cnt = 0;
   if(number_of_elements % 512 == 0)
  {
    thread_cnt = 512;
    block_cnt = number_of_elements/512;
  }
  //   else if(number_of_elements < 512){
  //   thread_cnt =number_of_elements;
  //   block_cnt = 1;
  // }
  //   else{
  // thread_cnt = number_of_elements%512;
  // block_cnt = number_of_elements/512;
  // }
  dim3 blocks_per_grid(block_cnt,1);    /* Number of blocks   */
  dim3 threads_per_block(thread_cnt,1);  /* Number of threads  */

  int l, m;
  for (l = 2; l <= number_of_elements; l <<= 1) {
    for (m=l>>1; m>0; m=m>>1) {
      bitonic_sort_step<<<blocks_per_grid, threads_per_block>>>(arr, m, l);
    }
  }
  cudaMemcpy(a, arr, number_of_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);

  return 0;
}

#ifdef __cplusplus
}
#endif


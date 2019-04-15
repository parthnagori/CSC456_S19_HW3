 #ifdef __cplusplus
  extern "C"
  {
  #endif


__global__ void bitonic_sort_step(float *gpu_val, int j, int k)
{
  unsigned int i, ij; 
    i = threadIdx.x + blockDim.x * blockIdx.x;
  ij = i^j;

  if ((ij)>i) {
    if ((i&k)==0) {
      if (gpu_val[i]>gpu_val[ij]) {
        float temp = gpu_val[i];
        gpu_val[i] = gpu_val[ij];
        gpu_val[ij] = temp;
      }
    }
    if ((i&k)!=0) {
      if (gpu_val[i]<gpu_val[ij]) {
        float temp = gpu_val[i];
        gpu_val[i] = gpu_val[ij];
        gpu_val[ij] = temp;
      }
    }
  }
}


int cuda_sort(int number_of_elements, float *values)
{
  
  float *gpu_arr;
  size_t size = number_of_elements * sizeof(float);

  cudaMalloc((void**) &gpu_arr, size);
  cudaMemcpy(gpu_arr, values, size, cudaMemcpyHostToDevice);
  
  int threads_create = 0;
  int blocks_create = 0;
   if(number_of_elements % 512 == 0)
  {
    threads_create = 512;
    blocks_create = number_of_elements/512;
  }
    else if(number_of_elements < 512){
    threads_create =number_of_elements;
    blocks_create = 1;
  }
    else{
  threads_create = number_of_elements%512;
  blocks_create = number_of_elements/512;
  }
  dim3 blocks(blocks_create,1);    /* Number of blocks   */
  dim3 threads(threads_create,1);  /* Number of threads  */

  int l, m;
  for (l = 2; l <= number_of_elements; l <<= 1) {
    for (m=l>>1; m>0; m=m>>1) {
      bitonic_sort_step<<<blocks, threads>>>(gpu_arr, m, l);
    }
  }
  cudaMemcpy(values, gpu_arr, size, cudaMemcpyDeviceToHost);
  cudaFree(gpu_arr);

  return 0;
}

#ifdef __cplusplus
}
#endif


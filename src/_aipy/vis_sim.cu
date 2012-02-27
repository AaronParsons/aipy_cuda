#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "cuda_add.h"


__global__ void find_vis( float *baseline, float *src_dir, float *src_int, float *freqs, int N_fq, int N_src, float *re_part, float *im_part) {
	int tid = blockIdx.x;
    int c = 3*10**8;
    if (tid < N_fq){
        fq = freqs[tid];
        for(int i =0;i<N_src;i++){//iterate over all sources
            float dot = 0;
            for (int j = 0;j<3;j++){//compute the dot product of baseline and source direction
                   dot += src_dir[i][j]*baseline[j];
                }
            coeff = src_int[i]*(fq/mfreq)**src_index[i];
            re_part = coeff*cos(-2*pi*fq*dot/c);
            im_part = coeff*sin(-2*pi*fq*dot/c);
            }
    }
}

int *cuda_add(float *baseline, float *src_dir, float *src_int, 
            float *freqs, float *vis_arr,
            int N_fq, int N_src){
    float *re_part, *im_part;
	float *dev_baseline, *dev_src_dir, *dev_src_int,*dev_freqs,
          *dev_re_part, *dev_im_part;
    int dev_N_fq, dev_N_src;

	// Allocate memory on the GPU
	cudaMalloc((void**) &dev_baseline,  3*sizeof(float));
	cudaMalloc((void**) &dev_src_dir,   3*N_src*sizeof(float));
	cudaMalloc((void**) &dev_src_int,   N_src*sizeof(float));
	cudaMalloc((void**) &dev_src_freqs, N_fq*sizeof(float));
	cudaMalloc((void**) &dev_re_part,   N_fq*sizeof(float));
	cudaMalloc((void**) &dev_im_part,   N_fq*sizeof(float));	    
    cudaMalloc((void**) &dev_N_fq,      sizeof(int));
	cudaMalloc((void**) &dev_N_src,     sizeof(int));
	
	// Move the arrays onto the GPU
    cudaMemcpy(dev_baseline,  baseline,  3*sizeof(float),         
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_src_dir,   src_dir,   3*N_src*sizeof(float),         
                cudaMemcpyHostToDevice);    
    cudaMemcpy(dev_src_int,   src_int,   N_src*sizeof(float),         
                cudaMemcpyHostToDevice);    
    cudaMemcpy(dev_src_freqs, src_freqs, N_fq*sizeof(float),         
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_re_part,   re_part,   N_fq*sizeof(float),         
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_im_part,   im_part,   N_fq*sizeof(float),         
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N_fq,      N_fq,      sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N_src,     N_src,     sizeof(int)), cudaMemcpyHostToDevice);


	find_vis<<<N_fq,1>>>(baseline, src_dir, src_int, freqs, 
                        N_fq, N_src, 
                        re_part, im_part);
	
	// copy the array back
	cudaMemcpy(re_part, dev_re_part, N_fq * sizeof(float), 
                cudaMemcpyDeviceToHost);
    cudaMemcpy(im_part, dev_im_part, N_fq * sizeof(float),
                cudaMemcpyDeviceToHost);
	
    //frees memory allocated on GPU
    
    cudaFree(dev_baseline);
    cudaFree(dev_src_dir);
    cudaFree(dev_src_int);
    cudaFree(dev_src_freqs);
    cudaFree(dev_re_part);
    cudaFree(dev_im_part);
    cudaFree(dev_N_fq);
    cudaFree(dev_N_src);
    
    //interleave re_part and im_part in the output array
    for(int i = 0;i<N_fq;i++){
        *vis_arr[2*i] = re_part[i];
        *vis_arr[2*i+1] = im_part[i];
        }

	return 1;
}
/

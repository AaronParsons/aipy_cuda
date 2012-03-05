#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "cuda_add.h"


__global__ void find_vis( float *baseline, float *src_dir, float *src_int, float *src_index, float *freqs, float* mfreqs, int N_fq, int N_src, float *vis_arr) {
	//Inputs: Baseline is length 3 vector in nanoseconds, src_dir is N_src*3 array, src_int is an N_src array, src_index is a N_src array, freqs is an N_fq array of frequencies in GHz, mfreqs is an N_src array
    //Outputs: re_part and im_part are N_fq arrays holding the computed visibility.
    int tid = blockIdx.x;
    if (tid < N_fq){ //Each thread handles the calculation of visibility for one frequency
        fq = freqs[tid];
        vis_arr[2*tid] = 0;
        vis_arr[2*tid+1] = 0;
        for(int i =0;i<N_src;i++){//iterate over all sources
            float dot = 0;
            for (int j = 0;j<3;j++){//compute the dot product of baseline and source direction
                   dot += src_dir[3*i+j] * baseline[j];
            }
            coeff = src_int[i]*(fq/mfreqs[i])**src_index[i];
            vis_arr[2*tid] += coeff*cos(-2*pi*fq*dot);
            vis_arr[2*tid+1] += coeff*sin(-2*pi*fq*dot);
        }
    }
}

int vis_sim(float *baseline, float *src_dir, float *src_int, float *src_index,
            float *freqs, float *mfreqs, float *vis_arr,
            int N_fq, int N_src){
	float *dev_baseline, *dev_src_dir, *dev_src_int, *dev_src_index, *dev_freqs, *dev_mfreqs,
          *dev_vis_arr;
    int dev_N_fq, dev_N_src;

	// Allocate memory on the GPU, do we need to check for success on cudaMalloc?
	cudaMalloc((void**) &dev_baseline,  3*sizeof(float));
	cudaMalloc((void**) &dev_src_dir,   3*N_src*sizeof(float));
	cudaMalloc((void**) &dev_src_int,   N_src*sizeof(float));
    cudaMalloc((void**) &dev_src_index, N_src*sizeof(float));
	cudaMalloc((void**) &dev_freqs,     N_fq*sizeof(float));
    cudaMalloc((void**) &dev_mfreqs,    N_src*sizeof(float));
	cudaMalloc((void**) &dev_vis_arr,   2 * N_fq*sizeof(float));	    
    cudaMalloc((void**) &dev_N_fq,      sizeof(int));
	cudaMalloc((void**) &dev_N_src,     sizeof(int));
	
	// Move the arrays onto the GPU
    cudaMemcpy(dev_baseline,  baseline,  3*sizeof(float),         
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_src_dir,   src_dir,   3*N_src*sizeof(float),         
                cudaMemcpyHostToDevice);    
    cudaMemcpy(dev_src_int,   src_int,   N_src*sizeof(float),         
                cudaMemcpyHostToDevice);    
    cudaMemcpy(dev_src_index, src_index, N_src*sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_freqs, freqs, N_fq*sizeof(float),         
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mfreqs, mfreqs, N_src*sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N_fq,      N_fq,      sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N_src,     N_src,     sizeof(int)), cudaMemcpyHostToDevice);


	find_vis<<<N_fq,1>>>(dev_baseline, dev_src_dir, dev_src_int, dev_src_index, dev_freqs, dev_mfreqs, 
                        dev_N_fq, dev_N_src, 
                        dev_vis_arr);
	
	// copy the array back
	cudaMemcpy(vis_arr, dev_vis_arr, 2*N_fq * sizeof(float), 
                cudaMemcpyDeviceToHost);
	
    //frees memory allocated on GPU
    
    cudaFree(dev_baseline);
    cudaFree(dev_src_dir);
    cudaFree(dev_src_int);
    cudaFree(dev_src_index);
    cudaFree(dev_src_freqs);
    cudaFree(dev_src_mfreqs);
    cudaFree(dev_vis_arr);
    cudaFree(dev_N_fq);
    cudaFree(dev_N_src);
    

	return 0;
}

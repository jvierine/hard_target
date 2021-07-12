// header file for the plasmaline project
#include "gmfgpu.h"

void print_devices()
{
  int nDevices;

  int ret=cudaGetDeviceCount(&nDevices);
 
  for (int i = 0; i < nDevices; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}


/*
  For each range gate (i), multiply transmit pulse with range delayed echo
  each range gate has a nfft2 length block in d_z_echo.
  all of these will be fft'ed in parallel with CuFFT at the next stage.

 */
__global__ void form_input(cufftComplex *z_tx, int z_tx_len, int nfft2,
                           cufftComplex *z_rx, int *rgs, int dec,
			   cufftComplex *d_z_echo, int *tx_idx, int nzi)
{
  int i = blockIdx.x;
  int rg=rgs[i];
  for(int ni=0 ; ni<nzi; ni++)
  {
    int ti=tx_idx[ni];
    d_z_echo[i*nfft2+ti/dec] = cuCaddf(d_z_echo[i*nfft2+ti/dec],cuCmulf(z_tx[ti], z_rx[rg+ti]));
  }
}

/*
  For each range gate (i), find the value of the spectrum that has highest power.
  Also store DC value for noise floor determination.
 */
__global__ void peak_find(cufftComplex *z_out, float *gmf_vec, float *gmf_dc_vec, float *v_vec, float *a_vec, int nfft2, int acc_idx)
{
  int i = blockIdx.x;
  if(acc_idx==0)
    gmf_dc_vec[i]=z_out[i*nfft2].x*z_out[i*nfft2].x+z_out[i*nfft2].y*z_out[i*nfft2].y;
  for(int j=0;j<nfft2;j++)
  {
    float this_gmf=z_out[i*nfft2+j].x*z_out[i*nfft2+j].x+z_out[i*nfft2+j].y*z_out[i*nfft2+j].y;
    if(this_gmf>gmf_vec[i])
    {
      gmf_vec[i]=this_gmf;
      v_vec[i]=(float)j;
      a_vec[i]=(float)acc_idx ;
    }
  }
}

/*
  Multiply echo with acceleration phasor, store input in d_z_in
 */
__global__ void phasor_multiply(cufftComplex *d_z_echo, cufftComplex *d_z_in, int nfft2, int i, cufftComplex *d_acc_phasors)
{
  int rgi = blockIdx.x;
  /* 
     tbd: only multiply non-zero values.
  */
  for(int j=0;j<nfft2;j++)
  {
    d_z_in[rgi*nfft2+j]=cuCmulf(d_z_echo[rgi*nfft2+j],d_acc_phasors[i*nfft2+j]);
  }
}

/* 
   This is the main code. If you have N GPUs, you can run N gmf functions in parallel.
*/
extern "C" int gmf(float *z_tx, int z_tx_len, float *z_rx, int z_rx_len, float *acc_phasors, int n_accs,
       	       	   float *rgs, int n_rg, int dec, float *gmf_vec, float *gmf_dc_vec, float *v_vec, float *a_vec, int rank)
{
  cudaSetDevice(rank);
  // initializing pointers to device (GPU) memory, denoted with "d_"
  cufftComplex *d_z_tx;
  cufftComplex *d_z_rx;
  //  cufftComplex *d_z_out;
  cufftComplex *d_z_echo;  
  cufftComplex *d_z_in;      
  cufftComplex *d_acc_phasors;
  float *d_gmf_vec;
  float *d_gmf_dc_vec;
  float *d_v_vec;
  float *d_a_vec;
  int *d_tx_idx;
  
  int *d_rgs;
  int *h_rgs;
  h_rgs=(int *)malloc(sizeof(int)*n_rg);
  for(int i=0;i<n_rg;i++)
  {
    h_rgs[i]=(int)rgs[i];
  }
      
  float *tx_power;
  tx_power=(float*)malloc(sizeof(float)*z_tx_len);
  
  int n_nonzero_tx=0;
  for(int ti=0 ; ti<z_tx_len ; ti++)
  {
    tx_power[ti]=z_tx[2*ti]*z_tx[2*ti] + z_tx[2*ti+1]*z_tx[2*ti+1];
    if(tx_power[ti] > 1e-10)
      n_nonzero_tx++;
  }

  int *tx_idx;
  tx_idx=(int*)malloc(sizeof(int)*n_nonzero_tx);
  int nzi=0;
  for(int ti=0 ; ti<z_tx_len ; ti++)
  {
    if(tx_power[ti] > 1e-10)
    {
      tx_idx[nzi]=ti;
      nzi++;
    }
  }
  
  int nfft2;
  
  nfft2=(int)(z_tx_len/dec);    

  // allocating device memory to the above pointers
  // the signal and echo here are only one row of the CPU data (one time step)
  int res= cudaMalloc((void **) &d_z_tx, sizeof(cufftComplex) * z_tx_len);
  if (res != cudaSuccess)
  {
    printf("error %d\n",res);
    fprintf(stderr, "Cuda error: Failed to allocate tx\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_rgs, sizeof(int) * n_rg) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate tx\n");
    exit(EXIT_FAILURE);
  }
  
  if (cudaMalloc((void **) &d_tx_idx, sizeof(int) * nzi) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate tx\n");
    exit(EXIT_FAILURE);
  }


  if (cudaMalloc((void **) &d_z_rx, sizeof(cufftComplex) * z_rx_len) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate echo\n");
    exit(EXIT_FAILURE);
  }
    
  if (cudaMalloc((void **) &d_z_in, sizeof(cufftComplex) * nfft2 * n_rg)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate batch\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_z_echo, sizeof(cufftComplex) * nfft2 * n_rg)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate batch\n");
    exit(EXIT_FAILURE);
  }
  //  if (cudaMalloc((void **) &d_z_out, sizeof(cufftComplex) * nfft2 * n_rg)
  //  != cudaSuccess)
  //  {
  // fprintf(stderr, "Cuda error: Failed to allocate batch\n");
  // exit(EXIT_FAILURE);
  // }
  if (cudaMalloc((void **) &d_acc_phasors, sizeof(cufftComplex) * n_accs*nfft2)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_gmf_vec, sizeof(float) * n_rg)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_gmf_dc_vec, sizeof(float) * n_rg)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_v_vec, sizeof(float) * n_rg)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d_a_vec, sizeof(float) * n_rg)
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }
  //  printf("malloced stuff\n");
  // initializing 1D FFT plan, this will tell cufft execution how to operate
  // cufft is well optimized and will run with different parameters than above
  cufftHandle plan;
  if (cufftPlan1d(&plan, nfft2, CUFFT_C2C, n_rg) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed\n");
    exit(EXIT_FAILURE);
  }
  //  printf("planned fft\n");
  // execution of the prepared kernels n_ipp times
  // ensure empty device spectrum
  if (cudaMemset(d_z_in, 0, sizeof(cufftComplex) * nfft2*n_rg) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to zero device spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMemset(d_z_echo, 0, sizeof(cufftComplex) * nfft2*n_rg) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to zero device spectrum\n");
    exit(EXIT_FAILURE);
  }

  // copying n_ipp'th row of host data into device
  if (cudaMemcpy(d_z_tx, z_tx, sizeof(cufftComplex) * z_tx_len,
		 cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
    exit(EXIT_FAILURE);
  }
  // copying n_ipp'th row of host data into device
  if (cudaMemcpy(d_rgs, h_rgs, sizeof(int) * n_rg,
		 cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
    exit(EXIT_FAILURE);
  }

  // copying n_ipp'th row of host data into device
  if (cudaMemcpy(d_tx_idx, tx_idx, sizeof(int) * nzi,
		 cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
    exit(EXIT_FAILURE);
  }
  

  // copying n_ipp'th row of host data into device
  if (cudaMemcpy(d_acc_phasors, acc_phasors, sizeof(cufftComplex) * nfft2*n_accs,
		 cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
    exit(EXIT_FAILURE);
  }

  if (cudaMemcpy(d_z_rx, z_rx, sizeof(cufftComplex) * z_rx_len,
		 cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, echo HtD\n");
    exit(EXIT_FAILURE);
  }

  //  if (cudaMalloc((void **) &d_acc_phasors, sizeof(cufftComplex) * n_accs*nfft2)
  
        
  // form input
  form_input<<<n_rg,1>>>(d_z_tx, z_tx_len, nfft2, d_z_rx, d_rgs, dec, d_z_echo, d_tx_idx, nzi);
  
  for(int i=0 ; i < n_accs; i++)
  {
    if (cudaMemset(d_z_in, 0, sizeof(cufftComplex) * nfft2*n_rg) != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to zero device spectrum\n");
      exit(EXIT_FAILURE);
    }
    phasor_multiply<<<n_rg,1>>>(d_z_echo, d_z_in, nfft2, i, d_acc_phasors);
    
    // cufft kernel execution
    if (cufftExecC2C(plan, (cufftComplex *)d_z_in, (cufftComplex *)d_z_in, CUFFT_FORWARD)
	       != CUFFT_SUCCESS)
    {
      fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
      exit(EXIT_FAILURE);
    }
    peak_find<<<n_rg,1>>>(d_z_in, d_gmf_vec, d_gmf_dc_vec, d_v_vec, d_a_vec, nfft2, i);
  }


  // copying device resultant spectrum to host, now able to be manipulated
  if (cudaMemcpy(gmf_vec, d_gmf_vec, sizeof(float) * n_rg,
		 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
    exit(EXIT_FAILURE);
  }
  // copying device resultant spectrum to host, now able to be manipulated
  if (cudaMemcpy(gmf_dc_vec, d_gmf_dc_vec, sizeof(float) * n_rg,
		 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
    exit(EXIT_FAILURE);
  }
  // copying device resultant spectrum to host, now able to be manipulated
  if (cudaMemcpy(v_vec, d_v_vec, sizeof(float) * n_rg,
		 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
    exit(EXIT_FAILURE);
  }
  // copying device resultant spectrum to host, now able to be manipulated
  if (cudaMemcpy(a_vec, d_a_vec, sizeof(float) * n_rg,
		 cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
    exit(EXIT_FAILURE);
  }

  free(h_rgs);
  free(tx_idx);
  free(tx_power);
  
  // memory clean up
  if (cudaFree(d_z_tx) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free tx\n");
    exit(EXIT_FAILURE);
  }

  if (cudaFree(d_rgs) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free tx\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_z_rx) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free echo\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_z_in) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free batch\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_z_echo) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free batch\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_v_vec) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_a_vec) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_gmf_vec) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_gmf_dc_vec) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_acc_phasors) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d_tx_idx) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spectrum\n");
    exit(EXIT_FAILURE);
  }
  if (cufftDestroy(plan) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Failed to destroy plan\n");
    exit(EXIT_FAILURE);
  }
    return(1);
}

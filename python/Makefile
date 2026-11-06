GPU_CC = nvcc
CC=gcc
CFLAGS =
GPU_SO_CFLAGS = --compiler-options '-fPIC' -shared -L /usr/local/cuda-10.1/targets/x86_64-linux/lib/
GPU_LDFLAGS =  -lcufft
SO_CFLAGS = -fPIC -shared
LDFLAGS = -lfftw3f

ALL: libgmf.so

#libgmfgpu.so: cudafiles/gmfgpu.cu
#	$(GPU_CC) $(GPU_SO_CFLAGS) $(GPU_LDFLAGS) cudafiles/gmfgpu.cu -o $@
libgmf.so: cfiles/gmf.c
	$(CC) $(SO_CFLAGS)  cfiles/gmf.c -o $@ $(LDFLAGS)

libgmfgpu.so: cudafiles/gmfgpu.cu
	$(GPU_CC) $(GPU_SO_CFLAGS)  cudafiles/gmfgpu.cu -o $@ $(GPU_LDFLAGS)

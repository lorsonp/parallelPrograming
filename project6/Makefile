CUDA_PATH	=	/usr/local/apps/cuda/cuda-9.2
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc ; \

project6: project6.cu ; \
	$(CUDA_NVCC)  -DBLOCKSIZE=$$16 -DNUMTRIALS=$$16000 -o proj project6.cu ; \
	./proj ; \
project6: project6.cu ; \
	$(CUDA_NVCC)  -DBLOCKSIZE=$$32 -DNUMTRIALS=$$16000 -o proj  project6.cu ; \
	./proj ; \
project6: project6.cu ; \
	$(CUDA_NVCC)  -DBLOCKSIZE=$$64 -DNUMTRIALS=$$16000 -o proj project6.cu ; \
	./proj ; \


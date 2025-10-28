CC		= nvcc 
EXEC	= cuda_pca

SOURCES = \
cuda_pca.cu

OBJS = $(SOURCES:.cu=.o)

%.o: %.cu
	$(CC) -c -O3 -o $@ $<

all: $(OBJS)
	$(CC) -link -L/usr/local/cuda/lib64/ -O3 $(OBJS) -o $(EXEC)

clean:
	@rm -f $(OBJS) $(EXEC)

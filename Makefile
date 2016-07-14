# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_30 -code=sm_30
NVCCFLAGS = -O3 -arch=compute_30 -code=sm_30
LIBS = 

TARGETS = serial gpu autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu
autograder.o: autograder.cu common.h
	$(CC) -c $(CFLAGS) autograder.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
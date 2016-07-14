#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include "common.h"


#define NUM_THREADS 256

#define MAXITEM 10 
#define CUTOFF_SCALE 10


extern double size;

double SizeOfBasket;
int numberOfBaskets;


__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__constant__ const int dir[9][2]={{0,0},{-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0}};

__global__ void forces_compute(particle_t * particles, int*opt,int n,double SizeOfBasket,int numberOfBaskets)
{
  
    //creating thread index for each thread 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int bb=tid;bb<n;bb+=offset)
    {
        particle_t p = particles[bb];
        p.ax=p.ay=0;
        int i = int(p.x / SizeOfBasket);
        int j = int(p.y / SizeOfBasket);
        for(int t=0;t<9;t++)
        {
            int x = i + dir[t][0];
            int y = j + dir[t][1];
            if (x >= 0 && x < numberOfBaskets && y >= 0 && y < numberOfBaskets)
            {
                int id = x*numberOfBaskets+y;
                int start = opt[id-1],end = opt[id];
                for (int k = start; k < end; k++)
                    apply_force_gpu(p, particles[k]);
            }
        }
        particles[bb].ax = p.ax;
        particles[bb].ay = p.ay;
    }
}


__global__ void move_gpu (particle_t * __restrict__ particles, int n, double size)
{
    //thread index fetched
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {

        particle_t * p = &particles[i];
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x  += p->vx * dt;
        p->y  += p->vy * dt;
    
      while( p->x < 0 || p->x > size )
        {
            p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
            p->vx = -(p->vx);
        }
        while( p->y < 0 || p->y > size )
        {
            p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
            p->vy = -(p->vy);
        }
    }

}

__global__ void Form_Basket(particle_t* particles,particle_t* temp,int* total,int n,double SizeOfBasket,int numberOfBaskets)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {
        int x = int(particles[i].x / SizeOfBasket);
        int y = int(particles[i].y / SizeOfBasket);
        int id = atomicSub(total+x*numberOfBaskets+y,1);
        temp[id-1] = particles[i];
    }
}
__global__ void Count_chunks(particle_t* particles, int* total,int n,double SizeOfBasket,int numberOfBaskets)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x*blockDim.x;
    for(int i=tid;i<n;i+=offset)
    {
        int x = int(particles[i].x / SizeOfBasket);
        int y = int(particles[i].y / SizeOfBasket);
        atomicAdd(total+x*numberOfBaskets+y,1);
    }
}

int main( int argc, char **argv )
{    
    cudaThreadSynchronize(); //Waiting until all the threads have finished execution.

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
  
  
  int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
  
  
  //construction of Particles
    particle_t * d_particles,*temp;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));//Linear size memory allocated and pointer is returned.
    cudaMalloc((void **) &temp, n * sizeof(particle_t));//Linear size memory allocated and pointer is returned.
  
  
  set_size( n );
    SizeOfBasket = cutoff*CUTOFF_SCALE;  
    numberOfBaskets = int(size / SizeOfBasket)+1; 
    printf("Size of the Mesh Basket: %.4lf\n",size);
    printf("Number of Baskets: %d*%d\n",numberOfBaskets,numberOfBaskets);
    printf("Size of Basket: %.2lf\n",SizeOfBasket);
    
  int* opt;
    cudaMalloc((void **) &opt, (numberOfBaskets*numberOfBaskets+1) * sizeof(int));//Linear size memory allocated and pointer is returned.
    cudaMemset(opt,0,(numberOfBaskets*numberOfBaskets+1)*sizeof(int));
    opt+=1; 
    int* total = (int*) malloc(numberOfBaskets*numberOfBaskets * sizeof(int));
      
    init_particles( n, particles );
  
  
  cudaThreadSynchronize();//Waiting until all the threads have finished execution.
    double cptime = read_timer( );

    
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice); //Memcpy used to copy particles in the system
  
  cudaThreadSynchronize();//Waiting until all the threads have finished execution.
    cptime = read_timer( ) - cptime;
  
  
  cudaThreadSynchronize();
    double simtime = read_timer( );
  
  for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //
        int threadcount = NUM_THREADS;
        int chunks = min(1024,(n + NUM_THREADS - 1) / NUM_THREADS);
      int noOfChunks = chunks;//min(512,(n+threadcount-1)/threadcount);
        
    
        cudaMemset(opt,0,numberOfBaskets*numberOfBaskets*sizeof(int));
        Count_chunks<<<noOfChunks,threadcount>>>(d_particles,opt,n,SizeOfBasket,numberOfBaskets);
    
    cudaMemcpy(total, opt, numberOfBaskets*numberOfBaskets * sizeof(int), cudaMemcpyDeviceToHost);//particles are copied back to the CPU
        for(int i=1;i<numberOfBaskets*numberOfBaskets;i++)  
            total[i]+=total[i-1];
        cudaMemcpy(opt, total, numberOfBaskets*numberOfBaskets * sizeof(int), cudaMemcpyHostToDevice);//particles are copied back to the CPU
        Form_Basket<<<noOfChunks,threadcount>>>(d_particles,temp,opt,n,SizeOfBasket,numberOfBaskets);
        std::swap(d_particles,temp);
        cudaMemcpy(opt, total, numberOfBaskets*numberOfBaskets * sizeof(int), cudaMemcpyHostToDevice);//particles are copied back to the CPU
    
    
    forces_compute<<<chunks, NUM_THREADS>>> (d_particles,opt,n,SizeOfBasket,numberOfBaskets);
    
    move_gpu <<< chunks, NUM_THREADS >>> (d_particles, n, size);
    
    if( fsave && (step%SAVEFREQ) == 0 ) {
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);//particles are copied back to the CPU
            save( fsave, n, particles);
      }
    }
  
  
   cudaThreadSynchronize();
    simtime = read_timer( ) - simtime;
  
  printf( "Copying time from CPU-GPU = %g seconds\n", cptime);
    printf( "n = %d, simulation time = %g seconds\n", n, simtime );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}

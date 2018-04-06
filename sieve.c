#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#define UPPER_BOUND 10000000000
#define MODE 1  // 0 to forsake CL-ARG
#define PRINT 0 // 1 to print primes.
#define PAGE 2500

// MACROS for bit operations
#define SetBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )  
#define TestBit(A,k)    ( A[(k/32)] & (1 << (k%32)) ) 

// MACROS for comparisons
#define MAX(a,b)  ( (a > b) ? a : b )
#define MIN(a,b)  ( (a < b) ? a : b )


int getSmallPrimes(int limit, int** rootPrimes){
    /*******************************************************************
    * DESCRIPTION : Get prime number for smaller limits. Used as Subroutine for each processes to get primes less than root(limit).
    * INPUT :
    *           [1] limit : limit of the range for which prime number needs to be generated.
    *           [2] rootPrimes : Pointer to the integer array that can store the prime number generated.
    * OUTPUT :
    *           [1] Returns size of rootPrimes array.
    *
    */

    int i, count = 0, prime = 2;
    int llsqrt = ceil(sqrt(limit));
    
    // Bit Vector for storing flags used in sieve for each number
    int *arr = (int*) calloc(ceil(limit/32), sizeof(int));
    
    // Marking each number that is not prime as per the sieve.
    while(prime <= llsqrt){
        while(prime <= llsqrt && TestBit(arr,prime))
            prime++;
        for(i = prime * prime; i <= limit; i += prime)
                SetBit(arr, i);
        prime++;
    }

    // Counting number of prime for array allocation
    for(i = 2; i < limit; i++){
        if(TestBit(arr, i))
            count += 1;
    }
    (*rootPrimes) = (int*) calloc(count, sizeof(int));
    
    // Adding each Prime in the Array
    count = 0;
    for(i = 2; i < limit; i++){
        if(!TestBit(arr, i)){
            (*rootPrimes)[count] = i;
                count+=1;
        }
    }

    return count;
}

int main(int argc, char *argv[]){
    /*******************************************************************
    * DESCRIPTION : Driver Function to implement algorithm that runs on multiple proccesses.
    * INPUT :
    *           [1] argv[1] : power of 10, that is treated as limit for finding sieve.
    *
    */

    long limit, n_hi, n_lo;
    int world_size, id, root = 0;
    double wtime;
    
    if (argc > 1 && MODE == 1)
        limit = (long) pow(10, atoi(argv[1]));
    else
        limit = UPPER_BOUND;

    // Initialization of MPI environment
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    // Inittilizing the timer
    if (id == 0)
        wtime = MPI_Wtime();
    
    // Setting the range (n_lo, n_hi] for each process.
    // Each process will apply algorithm on this range and finally root process will combine the results.
    n_lo = (long) (limit / world_size) * id;
    if (id == world_size - 1)
       n_hi = limit;
    else
       n_hi = (long) (limit / world_size) * (id + 1);
    
    long lsqrt = (long) ceil(sqrt(limit));
    
    // If limit is too small then it will only run on one process.
    if (lsqrt < 10000){
        if(id == root){
            int* rootPrimes ;
            int rootCount =  getSmallPrimes(limit , &rootPrimes);
            if(PRINT == 1){
                int i;
                for(i = 0; i < rootCount; i++)
                    printf(rootPrimes[i]);
            }
            wtime = MPI_Wtime() - wtime;
            printf("         N        Pi          Time\n");
            printf("  %10ld    %10d  %16f\n", limit, rootCount, wtime);
        }
        MPI_Finalize();
    }
    else{
        long part_size = n_hi - n_lo;
        long base;
        long i;
        
        // Computing primes in the range of root(limit)
        int* rootPrimes ;
        int rootCount = getSmallPrimes(lsqrt , &rootPrimes);

        // Allocating Bit vector for marking compositions
        int *arr = (int *)calloc((int)ceil((n_hi-n_lo)/32), sizeof(int));
        
        // Marking some initial values for algorithm to run smoothly
        if(n_lo == 0){
            SetBit(arr,1);
            SetBit(arr,4);       
        }
        SetBit(arr,0);

        long j, count = 0;
        long start, dprime;
        
        // Marking Composites
        for(i = 0; i < rootCount; i++){   
            dprime = 2 * rootPrimes[i];
            if(n_lo == 0) 
                start = ((long)rootPrimes[i]) * ((long)rootPrimes[i]);
            else 
                start = MAX((long) ceil(((double)n_lo) / rootPrimes[i]) * rootPrimes[i], ((long)rootPrimes[i]) * ((long)rootPrimes[i])) -  n_lo;  

            if( start % 2 == 0 && rootPrimes[i] > 2)
                start += rootPrimes[i];
            if(rootPrimes[i] == 2)
                dprime = 2;

            for(j = start ; j < part_size; j += dprime )
                SetBit(arr,j);
        }

        // Initializing a Bit vector in the root process that gather bit vectors from all the proccesses.
        int* all_range;
        if(id == root)
            all_range = (int*) calloc((limit/32), sizeof(int));

        MPI_Gather(arr, (part_size/32), MPI_INT, all_range, (part_size/32), MPI_INT, root, MPI_COMM_WORLD);

        // total_count initialized one because the BitVector does not count 1 as prime so we added it afterwards.
        long total_count = 1;

        // Final Counting and printing of all the prime Numbers.
        if(id == root){
            for(i = 1; i < limit; i += 2){
                if (!TestBit(all_range,i)){
                    total_count += 1;
                    if(PRINT == 1)
                        printf("%ld \n", i);
                }
            }
            
            wtime = MPI_Wtime() - wtime;
            printf("         N        Pi          Time\n");
            printf("  %10ld %10ld  %16f\n", limit, total_count, wtime);
        }
        MPI_Finalize();
    }
    return 0;
}        
       
       
       



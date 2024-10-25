#include <string.h>

#include <iostream>
#include <thread>   // For std::this_thread
#include <chrono> 
#include "zmorton.hpp"
#include "binhash.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 * 
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 * 
 *@c*/

#define HASH_MASK (HASH_DIM-1)

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    // Compute the grid coordinates of the particle
    int ix = p->x[0] / h;
    int iy = p->x[1] / h;
    int iz = p->x[2] / h;

    unsigned count = 0;

    //#pragma omp single nowait 
    //#pragma omp parallel for collapse(3) private(count)
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                
                //#pragma omp task firstprivate(dx, dy, count)
                //{
        
                    int nx = ix + dx;
                    int ny = iy + dy;
                    int nz = iz + dz;

                    // Compute the hash for the neighboring bin
                    unsigned neighbor_hash = zm_encode(nx & HASH_MASK, ny & HASH_MASK, nz & HASH_MASK);

                    // Add to the buckets array
                    //#pragma omp critical
                    //{
                        buckets[count++] = neighbor_hash;
                    //}
    				
                //}
            }
            
        }
    }

    return count; // Number of neighboring buckets
}

// unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
// {
//     // Compute the grid coordinates of the particle
//     int ix = p->x[0] / h;
//     int iy = p->x[1] / h;
//     int iz = p->x[2] / h;

//     unsigned count_global = 0; // Shared global count across threads

//     #pragma omp parallel
//     {
//         unsigned count_local = 0; // Thread-local count within the parallel region
//         unsigned local_buckets[27]; // Thread-local array for buckets

//         // Iterate over neighboring bins (including the current bin)
//         #pragma omp for collapse(3) nowait
//         for (int dx = -1; dx <= 1; ++dx) {
//             for (int dy = -1; dy <= 1; ++dy) {
//                 for (int dz = -1; dz <= 1; ++dz) {
//                     int nx = ix + dx;
//                     int ny = iy + dy;
//                     int nz = iz + dz;

//                     // Compute the hash for the neighboring bin
//                     unsigned neighbor_hash = zm_encode(nx & HASH_MASK, ny & HASH_MASK, nz & HASH_MASK);

//                     local_buckets[count_local++] = neighbor_hash; // Store in local array
//                 }
//             }
//         }

//         // Critical section only for updating the shared global bucket array
//         #pragma omp critical
//         {
//             for (unsigned i = 0; i < count_local; ++i) {
//                 buckets[count_global++] = local_buckets[i]; // Safely update the global bucket array
//             }
//         }
//     }

//     return count_global; // Return the total number of neighboring buckets
// }

void hash_particles(sim_state_t* s, float h)
{
	
	//memset(s->hash, 0, sizeof(particle_t*) * HASH_DIM);
	for (int i = 0; i < HASH_SIZE; i++) {
	    s->hash[i] = NULL;
	}
	
	// Iterate over all particles to insert them into the hash table
    for (int i = 0; i < s->n; ++i) {
        particle_t* p = &s->part[i];
        unsigned b = particle_bucket(p, h);

        // Insert particle at the beginning of the linked list for bin b
        p->next = s->hash[b];
        s->hash[b] = p;
    }

}

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <thread>   // For std::this_thread
#include <chrono> 
#include <omp.h>

#include "vec3.hpp"
#include "zmorton.hpp"

#include "params.hpp"
#include "state.hpp"
#include "interact.hpp"
#include "binhash.hpp"

/* Define this to use the bucketing version of the code */
#define USE_BUCKETING 

/*@T
 * \subsection{Density computations}
 * 
 * The formula for density is
 * \[
 *   \rho_i = \sum_j m_j W_{p6}(r_i-r_j,h)
 *          = \frac{315 m}{64 \pi h^9} \sum_{j \in N_i} (h^2 - r^2)^3.
 * \]
 * We search for neighbors of node $i$ by checking every particle,
 * which is not very efficient.  We do at least take advange of
 * the symmetry of the update ($i$ contributes to $j$ in the same
 * way that $j$ contributes to $i$).
 *@c*/

inline
void update_density(particle_t* pi, particle_t* pj, float h2, float C)
{
    float r2 = vec3_dist2(pi->x, pj->x);
    float z  = h2-r2;
    if (z > 0) {
        float rho_ij = C*z*z*z;
        pi->rho += rho_ij;
        pj->rho += rho_ij;
    }
}

void compute_density(sim_state_t* s, sim_param_t* params, double* dt_compute, double* dt_compute_par)
{
    // Use this command to make the code  sleep. Useful for debugging
    //std::this_thread::sleep_for(std::chrono::seconds(100));
    double time_start, time_end;
    time_start = omp_get_wtime();
    
    int n = s->n;
    particle_t* p = s->part;
    particle_t** hash = s->hash;


    float h  = params->h;
    float h2 = h*h;
    float h3 = h2*h;
    float h9 = h3*h3*h3;
    float C  = ( 315.0/64.0/M_PI ) * s->mass / h9;

    // Clear densities
    for (int i = 0; i < n; ++i)
        p[i].rho = 0;

    // Accumulate density info
#ifdef USE_BUCKETING
    /* BEGIN TASK */
    

    const int MAX_NEIGHBORS = 27; // 3x3x3 neighboring buckets
    unsigned buckets[MAX_NEIGHBORS];

    for (int i = 0; i < n; ++i) {
        particle_t* pi = s->part + i;
        pi->rho += ( 315.0/64.0/M_PI ) * s->mass / h3;
    }

    time_end = omp_get_wtime();
    *dt_compute += time_end - time_start;

    #pragma omp parallel for private(buckets)
    for (int i = 0; i < n; ++i) {
        time_start = omp_get_wtime();
        particle_t* pi = s->part + i;
    
    //pi->rho += ( 315.0/64.0/M_PI ) * s->mass / h3;

    //#pragma omp task firstprivate(buckets)
    //{
    
        // Retrieve neighboring buckets for particle pi
        unsigned num_buckets = particle_neighborhood(buckets, pi, h);
    
        //#pragma omp parallel for schedule(dynamic)
        for (unsigned b = 0; b < num_buckets; ++b) {
        
            unsigned bucket_id = buckets[b];
            particle_t* pj = hash[bucket_id];
            
            
            // Traverse the linked list of particles in the bucket
            while (pj != nullptr) 
            {
                int j = pj - s->part; // Calculate the index of pj
                if (j > i) // Ensure each pair is processed only once
                { 
                    float r2 = vec3_dist2(pi->x, pj->x);
                    float z  = h2-r2;
                    if (z > 0) 
                    {
                        float rho_ij = C*z*z*z;
                        //#pragma omp critical
                        {
			            #pragma omp atomic
                        pi->rho += rho_ij;
                        #pragma omp atomic
			            pj->rho += rho_ij;
                        }
                    }
                }

                pj = pj->next;
            }
        }
    //}
    time_end = omp_get_wtime();
    *dt_compute_par += time_end - time_start;

    }
    


    /* END TASK */
#else
    for (int i = 0; i < n; ++i) {
        particle_t* pi = s->part+i;
        pi->rho += ( 315.0/64.0/M_PI ) * s->mass / h3;
        for (int j = i+1; j < n; ++j) {
            particle_t* pj = s->part+j;
            update_density(pi, pj, h2, C);
        }
    }
#endif
}


/*@T
 * \subsection{Computing forces}
 * 
 * The acceleration is computed by the rule
 * \[
 *   \bfa_i = \frac{1}{\rho_i} \sum_{j \in N_i} 
 *     \bff_{ij}^{\mathrm{interact}} + \bfg,
 * \]
 * where the pair interaction formula is as previously described.
 * Like [[compute_density]], the [[compute_accel]] routine takes
 * advantage of the symmetry of the interaction forces
 * ($\bff_{ij}^{\mathrm{interact}} = -\bff_{ji}^{\mathrm{interact}}$)
 * but it does a very expensive brute force search for neighbors.
 *@c*/

inline
void update_forces(particle_t* pi, particle_t* pj, float h2,
                   float rho0, float C0, float Cp, float Cv)
{
    float dx[3];
    vec3_diff(dx, pi->x, pj->x);
    float r2 = vec3_len2(dx);
    if (r2 < h2) {
        const float rhoi = pi->rho;
        const float rhoj = pj->rho;
        float q = sqrt(r2/h2);
        float u = 1-q;
        float w0 = C0 * u/rhoi/rhoj;
        float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
        float wv = w0 * Cv;
        float dv[3];
        vec3_diff(dv, pi->v, pj->v);

        // Equal and opposite pressure forces
        vec3_saxpy(pi->a,  wp, dx);
        vec3_saxpy(pj->a, -wp, dx);
        
        // Equal and opposite viscosity forces
        vec3_saxpy(pi->a,  wv, dv);
        vec3_saxpy(pj->a, -wv, dv);
    }
}

void compute_accel(sim_state_t* state, sim_param_t* params, 
    double* dt_dense, double* dt_hash, double* dt_forc, double* dt_pn, double* dt_rest, double* dt_compute, double* dt_compute_par)
{
    //printf("\nEntering Compute Acceleration\n");

    double t_start, t_end, time_start, time_end;

    // Unpack basic parameters
    const float h    = params->h;
    const float rho0 = params->rho0;
    const float k    = params->k;
    const float mu   = params->mu;
    const float g    = params->g;
    const float mass = state->mass;
    const float h2   = h*h;

    // Unpack system state
    particle_t* p = state->part;
    particle_t** hash = state->hash;
    int n = state->n;

    // Rehash the particles
    t_start = omp_get_wtime();
    hash_particles(state, h);
    t_end   = omp_get_wtime();
    *dt_hash += t_end - t_start;
    *dt_compute += t_end - t_start;

    // Compute density and color
    t_start = omp_get_wtime();
    compute_density(state, params, dt_compute, dt_compute_par);
    t_end   = omp_get_wtime();
    *dt_dense += t_end - t_start;

    time_start = omp_get_wtime();

    // Start with gravity and surface forces
    for (int i = 0; i < n; ++i)
        vec3_set(p[i].a,  0, -g, 0);

    // Constants for interaction term
    float C0 = 45 * mass / M_PI / ( (h2)*(h2)*h );
    float Cp = k/2;
    float Cv = -mu;


    // Accumulate forces
    t_start = omp_get_wtime();
    #ifdef USE_BUCKETING
        /* BEGIN TASK */
        const int MAX_NEIGHBORS = 27; // 3x3x3 neighboring buckets
        unsigned buckets[MAX_NEIGHBORS];
        
        double pntime = 0.0, looptime=0.0, cptime = 0.0;  // Variable to store total time for particle_neighborhood

        time_end = omp_get_wtime();
        *dt_compute += time_end - time_start;
        #pragma omp parallel for private(buckets) reduction(+:pntime,looptime,cptime)
        for (int i = 0; i < n; ++i) {
            time_start = omp_get_wtime();
            particle_t* pi = state->part + i;

            // Retrieve neighboring buckets for particle pi
            double t_start_3 = omp_get_wtime();
            unsigned num_buckets = particle_neighborhood(buckets, pi, h);
            double t_end_3 = omp_get_wtime();
            pntime += t_end_3 - t_start_3;

            //#pragma omp parallel for schedule(dynamic)
            double t_start_2 = omp_get_wtime();
            for (unsigned b = 0; b < num_buckets; ++b) {
                unsigned bucket_id = buckets[b];
                particle_t* pj = hash[bucket_id];

                // Traverse the linked list of particles in the bucket
                while (pj != nullptr) {
                    int j = pj - state->part; // Calculate the index of pj
                    if (j > i) {   
                        float dx[3];
                        vec3_diff(dx, pi->x, pj->x);
                        float r2 = vec3_len2(dx);
                        if (r2 < h2) {
                            const float rhoi = pi->rho;
                            const float rhoj = pj->rho;
                            float q = sqrt(r2 / h2);
                            float u = 1 - q;
                            float w0 = C0 * u / rhoi / rhoj;
                            float wp = w0 * Cp * (rhoi + rhoj - 2 * rho0) * u / q;
                            float wv = w0 * Cv;
                            float dv[3];
                            vec3_diff(dv, pi->v, pj->v);

                            #pragma omp critical
                            {
                                // Equal and opposite pressure forces
                                vec3_saxpy(pi->a, wp, dx);
                                vec3_saxpy(pj->a, -wp, dx);

                                // Equal and opposite viscosity forces
                                vec3_saxpy(pi->a, wv, dv);
                                vec3_saxpy(pj->a, -wv, dv);
                            }
                        }
                    }

                    pj = pj->next;
                }
            }
            double t_end_2 = omp_get_wtime();
            looptime += t_end_2 - t_start_2;
            time_end = omp_get_wtime();
            cptime += time_end - time_start;

        }

        *dt_compute_par += cptime;
        *dt_rest += looptime;
        *dt_pn += pntime;

        /* END TASK */
        #else
            printf("Using the O(N^2) Algo");
            for (int i = 0; i < n; ++i) {
                particle_t* pi = p+i;
                for (int j = i+1; j < n; ++j) {
                    particle_t* pj = p+j;
                    update_forces(pi, pj, h2, rho0, C0, Cp, Cv);
                }
            }
        #endif
        t_end   = omp_get_wtime();
        *dt_forc += t_end - t_start;

    }


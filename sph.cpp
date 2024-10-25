#include <cstdio>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

#include "vec3.hpp"
#include "io.hpp"
#include "params.hpp"
#include "state.hpp"
#include "binhash.hpp"
#include "interact.hpp"
#include "leapfrog.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \section{Initialization}
 *
 * We've hard coded the computational domain to a unit box, but we'd prefer
 * to do something more flexible for the initial distribution of fluid.
 * In particular, we define the initial geometry of the fluid in terms of an
 * {\em indicator function} that is one for points in the domain occupied
 * by fluid and zero elsewhere.  A [[domain_fun_t]] is a pointer to an
 * indicator for a domain, which is a function that takes three floats and
 * returns 0 or 1.  Two examples of indicator functions are a little box
 * of fluid in the corner of the domain and a spherical drop.
 *@c*/
typedef int (*domain_fun_t)(float, float, float);

int box_indicator(float x, float y, float z)
{
    return (x < 0.5) && (y < 0.75) && (z < 0.5);
}

int circ_indicator(float x, float y, float z)
{
    float dx = (x-0.5);
    float dy = (y-0.5);
    float dz = (z-0.5);
    float r2 = dx*dx + dy*dy + dz*dz;
    return (r2 < 0.25*0.25*0.25);
}

/*@T
 *
 * The [[place_particles]] routine fills a region (indicated by the
 * [[indicatef]] argument) with fluid particles.  The fluid particles
 * are placed at points inside the domain that lie on a regular mesh
 * with cell sizes of $h/1.3$.  This is close enough to allow the
 * particles to overlap somewhat, but not too much.
 *@c*/
sim_state_t* place_particles(sim_param_t* param, 
                             domain_fun_t indicatef)
{
    float h  = param->h;
    float hh = h/1.3;

    // Count mesh points that fall in indicated region.
    int count = 0;
    for (float x = 0; x < 1; x += hh)
        for (float y = 0; y < 1; y += hh)
        	for (float z = 0; z < 1; z += hh)
        		count += indicatef(x,y,z);

    // Populate the particle data structure
    sim_state_t* s = alloc_state(count);
    int p = 0;
    for (float x = 0; x < 1; x += hh) {
        for (float y = 0; y < 1; y += hh) {
            for (float z = 0; z < 1; z += hh) {
                if (indicatef(x,y,z)) {
                    vec3_set(s->part[p].x, x, y, z);
                    vec3_set(s->part[p].v, 0, 0, 0);
                    ++p;
                }
            }
        }
    }
    return s;    
}

/*@T
 *
 * The [[place_particle]] routine determines the initial particle
 * placement, but not the desired mass.  We want the fluid in the
 * initial configuration to exist roughly at the reference density.
 * One way to do this is to take the volume in the indicated body of
 * fluid, multiply by the mass density, and divide by the number of
 * particles; but that requires that we be able to compute the volume
 * of the fluid region.  Alternately, we can simply compute the
 * average mass density assuming each particle has mass one, then use
 * that to compute the particle mass necessary in order to achieve the
 * desired reference density.  We do this with [[normalize_mass]].
 * 
 * @c*/
void normalize_mass(sim_state_t* s, sim_param_t* param)
{
    s->mass = 1;
    //printf("\nHashing:\n");
    hash_particles(s, param->h);
    //printf("\nhashing for the 2nd time:\n");
    //hash_particles(s, param->h);
    double dum1 = 0.0, dum2 = 0.0;
    compute_density(s, param, &dum1, &dum2);
    float rho0 = param->rho0;
    float rho2s = 0;
    float rhos  = 0;
    for (int i = 0; i < s->n; ++i) {
        rho2s += (s->part[i].rho)*(s->part[i].rho);
        rhos  += s->part[i].rho;
    }
    s->mass *= ( rho0*rhos / rho2s );
}

sim_state_t* init_particles(sim_param_t* param)
{
    printf("Initialization of the Particles \n");
    sim_state_t* s = place_particles(param, box_indicator);
    normalize_mass(s, param);
    return s;
}

/*@T
 * \section{The [[main]] event}
 *
 * The [[main]] routine actually runs the time step loop, writing
 * out files for visualization every few steps.  For debugging
 * convenience, we use [[check_state]] before writing out frames,
 * just so that we don't spend a lot of time on a simulation that
 * has gone berserk.
 *@c*/

void check_state(sim_state_t* s)
{
    for (int i = 0; i < s->n; ++i) {
        float xi = s->part[i].x[0];
        float yi = s->part[i].x[1];
        float zi = s->part[i].x[2];
        assert( xi >= 0 || xi <= 1 );
        assert( yi >= 0 || yi <= 1 );
        assert( zi >= 0 || zi <= 1 );
    }
}

int main(int argc, char** argv)
{
    int n_thrd = 6;
    double rn_thrd = n_thrd;
    omp_set_num_threads(n_thrd);
    
        
    sim_param_t params;
    if (get_params(argc, argv, &params) != 0)
        exit(-1);
    sim_state_t* state = init_particles(&params);
    
    
    int nframes = params.nframes;
    int npframe = params.npframe;
    float dt    = params.dt;
    int n       = state->n;

    std::string filename     = "N="+std::to_string(n)+"_"+params.fname;    
    std::string filename_ke  = "N="+std::to_string(n)+"_KE.out";    

    FILE* fp    = std::fopen(filename.c_str(), "w");
    FILE* fp_ke = std::fopen(filename_ke.c_str(), "w");
    
    double t_start = omp_get_wtime();
    //write_header(fp, n);
    write_header(fp, n, nframes, params.h);
    write_frame_data(fp, n, state, NULL);
    write_frame_ke(fp_ke, n, 0, state, NULL);
    
    double dt_compute_accel=0.0, dt_leapfrog=0.0, dt_check=0.0, dt_write=0.0, dt_compute=0.0, dt_compute_par=0.0;
    double dt_forces=0.0, dt_density=0.0, dt_hashing=0.0, dt_PN=0.0, dt_Rest=0.0, dt_synch = 0.0;
    double time_start, time_end;

    compute_accel(state, &params, &dt_density, &dt_hashing, &dt_forces, &dt_PN, &dt_Rest, &dt_compute, &dt_compute_par);
    time_start = omp_get_wtime();
    leapfrog_start(state, dt);
    check_state(state);
    time_end = omp_get_wtime();
    dt_compute += time_end - time_start;
    
    for (int frame = 1; frame < nframes; ++frame) {
            
        for (int i = 0; i < npframe; ++i) {
            
            time_start = omp_get_wtime();
            compute_accel(state, &params, &dt_density, &dt_hashing, &dt_forces, 
                &dt_PN, &dt_Rest, &dt_compute, &dt_compute_par);
            time_end = omp_get_wtime();
            dt_compute_accel += (time_end - time_start);

            time_start = omp_get_wtime();
            leapfrog_step(state, dt);
            time_end = omp_get_wtime();
            dt_leapfrog += (time_end - time_start);
            dt_compute += (time_end - time_start);

            time_start = omp_get_wtime();
            check_state(state);
            time_end = omp_get_wtime();
            dt_check += (time_end - time_start);
            dt_compute += (time_end - time_start);

        }
        printf("Frame: %d of %d - %2.1f%%\n",frame, nframes, 
               100*(float)frame/nframes);
            
        
        
        time_start = omp_get_wtime();
        write_frame_data(fp, n, state, NULL);
        write_frame_ke(fp_ke, n, frame, state, NULL);
        time_end = omp_get_wtime();
        dt_write += (time_end - time_start);
        dt_compute += (time_end - time_start);
    
    }



    double t_end = omp_get_wtime();
    dt_PN = dt_PN/rn_thrd;
    dt_Rest = dt_Rest/rn_thrd;
    double tot_compute = dt_compute + dt_compute_par/rn_thrd;
    dt_synch = t_end - t_start - tot_compute;


    printf("thread, N_p, total, compute, sync, comp_unp, comp_par (seconds)\n");
    printf(" %d, %d, %g, %g, %g, %g, %g \n", n_thrd,n,t_end-t_start,tot_compute,dt_synch,dt_compute,dt_compute_par/rn_thrd);

    // printf("Ran in %g seconds\n", t_end-t_start);
    // printf("Total Compute   =  %g seconds\n", dt_compute_accel);
    // printf("Total Synch   =  %g seconds\n", dt_synch);
    // printf("my compute   =  %g seconds\n", dt_compute);
    // printf("my compute par   =  %g seconds\n", dt_compute_par/rn_thrd);
    // printf("Compute Density =  %g seconds\n", dt_density);
    // printf("Compute Hashing =  %g seconds\n", dt_hashing);
    // printf("Compute Forces  =  %g seconds\n", dt_forces);
    // printf("Forces PN       =  %g seconds\n", dt_PN);
    // printf("Forces Rest     =  %g seconds\n", dt_Rest);
        

    fprintf(fp_ke, "Time=%e %e %e %e %e %e %e %e %e %e",
        t_end-t_start,dt_compute_accel,dt_density, dt_hashing, dt_forces, dt_PN, dt_Rest,
        dt_leapfrog,dt_check, dt_write);
    fclose(fp);
    fclose(fp_ke);
    
    free_state(state);
        
    
}

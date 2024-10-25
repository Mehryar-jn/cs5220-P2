#ifndef INTERACT_HPP
#define INTERACT_HPP

#include "params.hpp"
#include "state.hpp"

void compute_density(sim_state_t* s, sim_param_t* params, double* dt_compute, double* dt_compute_par);
void compute_accel(sim_state_t* state, sim_param_t* params,
	double* dt_dense, double* dt_hash, double* dt_forc, double* dt_pn, double* dt_rest, double* dt_compute, double* dt_compute_par);

#endif /* INTERACT_HPP */

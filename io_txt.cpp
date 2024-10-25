#include <stdio.h>
#include "io.hpp"

#ifndef IO_OUTBIN

#define VERSION_TAG "SPHView00 "


void write_header(FILE* fp, int n, int framecount, float h)
{
    fprintf(fp, "%s%d %d %g\n", VERSION_TAG, n, framecount, h);
}


void write_frame_data(FILE* fp, int n, sim_state_t* s, int* c)
{
    particle_t* p = s->part;
    for (int i = 0; i < n; ++i, ++p)
        fprintf(fp, "%e %e %e\n", p->x[0], p->x[1], p->x[2]);
}
void write_frame_ke(FILE* fp, int n, int framecount, sim_state_t* s, int* c)
{
    particle_t* p = s->part;
    double ke_sum = 0.0;
    for (int i = 0; i < n; ++i, ++p)
        ke_sum += ((p->v[0])*(p->v[0]))+((p->v[1])*(p->v[1]))+((p->v[2])*(p->v[2]));
    fprintf(fp, "%d %e\n", framecount, ke_sum);
}

#endif /* IO_OUTBIN */

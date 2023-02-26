#ifndef GENERATE_MATRIX_H
#define GENERATE_MATRIX_H

#include "mesh.h"

void generate_matrix(int nx, int ny, int nz, struct mesh **A, float **x, float **b, double **xexact, int use_7pt_stencil);
#endif

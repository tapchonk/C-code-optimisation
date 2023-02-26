#ifndef SILO_WRITER_H
#define SILO_WRITER_H

#include <silo.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include "mesh.h"

int writeTimestep(char* dir, int *timestep, struct mesh * matrix, float* p, float* r, float* Ap, const float *const b, float *const x);

#endif
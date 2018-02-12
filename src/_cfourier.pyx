from __future__ import division, absolute_import
import numpy as np

cimport numpy as np

cdef extern from "fourier.h":
    void fourier(int q_len, const double *qvalues, double *pg, int z_len,
              const double *z, const double *y_dash, const double bin_width)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def ft(np.ndarray q_values, np.ndarray z, np.ndarray y_dash, bin_width):
    cdef int q_len = q_values.size
    cdef int z_len = z.size
    cdef np.ndarray p = np.zeros_like(q_values, DTYPE)

    fourier(q_len,
             <const double*>q_values.data,
             <double*>p.data,
             z_len,
             <const double*>z.data,
             <const double*>y_dash.data,
            bin_width)

    return p
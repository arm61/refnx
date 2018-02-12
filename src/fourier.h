#ifdef __cplusplus
extern "C" {
#endif


void fourier(int q_len, const double *qvalues, double *pg, int z_len,
              const double *z, const double *y_dash, const double bin_width);

#ifdef __cplusplus
}
#endif
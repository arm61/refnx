/*
    fourier.h

    *Calculates the specular (Neutron or X-ray) reflectivity from a stratified
    series of layers.
    @Copyright, Andrew McCluskey 2018.
 */


#include "fourier.h"
#include <math.h>
#include "MyComplex.h"

using namespace MyComplexNumber;

void fourier(int q_len, const double *qvalues, double *pg, int z_len,
              const double *z, const double *y_dash, const double bin_width)
{
    int i;
    MyComplex p[q_len];
    for (i = 0; i < q_len; i++)
    {
        p[i].re = 0;
        p[i].im = 0;
    }
    for (i = 0; i < q_len; i++)
    {
        int j;
        for (j = 0; j < z_len; j++)
        {
            MyComplex qc, zc, oj, ydc;
            qc.re = qvalues[i];
            qc.im = 0;
            zc.re = z[j];
            zc.im = 0;
            oj.re = 0;
            oj.im = -1;
            ydc.re = y_dash[j];
            ydc.im = y_dash[j];
            p[i] = p[i] + ydc * compexp(oj * zc * qc) * bin_width;
        }
    }
    for (i = 0; i < q_len; i++)
    {
        pg[i] = p[i].re;
    }
}
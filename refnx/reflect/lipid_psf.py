import numpy as np
from refnx.analysis import (Parameters, Parameter, possibly_create_parameter)
from scipy.interpolate import InterpolatedUnivariateSpline

_FWHM = 2 * np.sqrt(2 * np.log(2.))


class LipidPSF(object):
    def __init__(self, lipidstructure, scale=1, bkg=1e-7, name='', dq=5.):
        """
        :param structure: refnx.reflect.LipidStructure
            the structure of the lipids of interest
        :param scale: float
            scale factor by which the calculated ref is scaled
        :param bkg: float
            linear background added to all the models
        :param name: str
            name of the model
        :param dq: float
            the resolution function for the instrument
        """
        self.name = name
        self._parameters = None
        self._scale = possibly_create_parameter(scale, name='scale')
        self._bkg = possibly_create_parameter(bkg, name='bkg')
        self._dq = possibly_create_parameter(dq, name='dq - resolution')
        self._lipidstructure = None
        self.lipidstructure = lipidstructure

    def __call__(self, x, x_err=None):
        return self.model(x, x_err=x_err)

    @property
    def dq(self):
        """
        Returns
        -------
        dq : Parameter
            If `dq.value == 0` then no resolution smearing is employed.
            If `dq.value > 0`, then a constant dQ/Q resolution smearing is
            employed.  For 5% resolution smearing supply 5. However, if
            `x_err` is supplied to the `model` method, then that overrides any
            setting reported here.
        """
        return self._dq

    @dq.setter
    def dq(self, value):
        self._dq.value = value

    @property
    def scale(self):
        """
        Returns
        -------
        scale : Parameter
            scale factor. All model values are multiplied by this value before
            the background is added.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale.value = value

    @property
    def bkg(self):
        """
        Returns
        -------
        bkg : Parameter
            linear background added to all model values.
        """
        return self._bkg

    @bkg.setter
    def bkg(self, value):
        self._bkg.value = value


    def model(self, x, x_err=None):
        if x_err is None:
            x_err = float(self.dq)

        return reflectivity(x, self.lipidstructure, scale=self.scale.value,
                                  bkg=self.bkg.value, dq=x_err)

    def lnprob(self):
        """
        Additional log-probability terms for the reflectivity model. Do not
        include log-probability terms for model parameters, these are
        automatically calculated elsewhere.

        Returns
        -------
        lnprob : float
            log-probability of structure.
        """
        return self.lipidstructure.lnprob()

    @property
    def lipidstructure(self):
        """
        Returns
        -------
        structure : Structure
            Structure objects describe the interface of a reflectometry sample.
        """
        return self._lipidstructure

    @lipidstructure.setter
    def lipidstructure(self, lipidstructure):
        self._lipidstructure = lipidstructure
        p = Parameters(name='instrument parameters')
        p.extend([self.scale, self.bkg, self.dq])

        self._parameters = Parameters(name=self.name)
        self._parameters.extend([p, lipidstructure.parameters])

    @property
    def parameters(self):
        self.lipidstructure = self._lipidstructure
        return self._parameters

def reflectivity(x, lipidstructure, scale=1., bkg=1e-7, dq=5.):
    return _smearing_constant(x, lipidstructure, scale, bkg, dq)
    #return refcalc(x, lipidstructure, scale, bkg)

import time

def refcalc(x, lipidstructure, scale=1., bkg=1e-7):
    if len(lipidstructure.numberdensity()[0]) != len(lipidstructure.z):
        print(lipidstructure.z, lipidstructure.numberdensity()[0])
    tail_nddash = analytical_derivative(lipidstructure.z, lipidstructure.numberdensity()[0])
    head_nddash = analytical_derivative(lipidstructure.z, lipidstructure.numberdensity()[1])
    solv_nddash = analytical_derivative(lipidstructure.z, lipidstructure.numberdensity()[2])
    z_ft = lipidstructure.z[0:-1]
    tail_ft = fourier_transform(x, z_ft, tail_nddash)
    head_ft = fourier_transform(x, z_ft, head_nddash)
    solv_ft = fourier_transform(x, z_ft, solv_nddash)
    htt = np.power(np.absolute(tail_ft), 2)
    hhh = np.power(np.absolute(head_ft), 2)
    hss = np.power(np.absolute(solv_ft), 2)
    hts = np.sqrt(htt * hss) * np.cos(x * lipidstructure.tailsolv_sep.value)
    hth = np.sqrt(htt * hhh) * np.sin(x * lipidstructure.tailhead_sep.value)
    hhs = np.sqrt(hhh * hss) * np.sin(x * lipidstructure.headsolv_sep.value)
    like_terms = ((lipidstructure.head_b.value ** 2 * hhh) + (lipidstructure.tail_b.value ** 2 * htt) +
                  (lipidstructure.solv_b.value ** 2 * hss))
    cross_terms = (2 * lipidstructure.tail_b.value * lipidstructure.head_b.value * hth) * \
                  (2 * lipidstructure.solv_b.value * lipidstructure.head_b.value * hhs) * \
                  (2 * lipidstructure.tail_b.value * lipidstructure.solv_b.value * hts)
    r = (16. * np.pi ** 2 * (like_terms + cross_terms)) / (x ** 4)
    r = r * scale + bkg
    return r

def _smearing_constant(q, lipidstructure, scale=1., bkg=1e-7, dq=5.):
    if dq < 0.5:
        return refcalc(q, lipidstructure, scale, bkg)

    dq /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    def gauss(x, s):
        return 1. / s / np.sqrt(2 * np.pi) * np.exp(-0.5 * np.power(x, 2) / s / s)

    lowq = np.min(q)
    highq = np.max(q)
    if lowq <= 0.:
        lowq = 1e-6

    start = np.log10(lowq) - 6 * dq / _FWHM
    finish = np.log10(highq * (1 + 6 * dq / _FWHM))
    interpnum = np.round(np.abs(1 * (np.abs(start - finish)) /
                                (1.7 * dq / _FWHM / gaussgpoint)))
    xtemp = np.linspace(start, finish, int(interpnum))
    xlin = np.power(10., xtemp)

    gauss_x = np.linspace(-1.7 * dq, 1.7 * dq, gaussnum)
    gauss_y = gauss(gauss_x, dq / _FWHM)

    rvals = refcalc(xlin, lipidstructure, scale, bkg)
    smeared_rvals = np.convolve(rvals, gauss_y, mode='same')
    interpolator = InterpolatedUnivariateSpline(xlin, smeared_rvals)

    smeared_output = interpolator(q)
    smeared_output *= gauss_x[1] - gauss_x[0]
    return smeared_output


def analytical_derivative(x, y):
    ydash = np.zeros_like(y)[0:-1]
    for i in range(0, len(ydash)):
        ydash[i] = ((y[i+1] - y[i]) / (x[i+1] - x[i]))
    return ydash

from multiprocessing import Pool
from functools import partial

def fruit_loops(ydash, z, q_values):
    p = np.zeros_like(q_values, dtype=complex)
    for i, q in enumerate(q_values):
        p[i] = np.sum(np.multiply(ydash, np.exp(np.multiply(np.multiply(-1j, z), q))))
    return p

def fourier_transform(q_values, z, ydash):
    cores = 7
    chunks = [q_values[i::cores] for i in range(cores)]
    pool = Pool(processes=cores)
    func = partial(fruit_loops, ydash, z)
    p = pool.map(func, chunks)
    p_ret = []
    for i in range(0, len(chunks[0])):
        for j in range(0, len(chunks)):
            if i == len(chunks[j]):
                break
            p_ret.append(p[j][i])
    pool.close()
    return p_ret

def slowfour(q_values, z, ydash):
    p = np.zeros_like(q_values, dtype=complex)
    for i, q in enumerate(q_values):
        p[i] = np.sum(np.multiply(ydash, np.exp(np.multiply(np.multiply(-1j, z), q))))
    return p

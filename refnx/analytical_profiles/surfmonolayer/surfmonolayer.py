"""
Analytic profile for studying surfactant/lipid monolayers at an interface
"""

import numpy as np
from refnx.analysis import Parameter, possibly_create_parameter
from refnx.reflect import SLD, Component
from numpy.testing import assert_


class SurfMono(Component):
    """
    A surfactant monolayer component, consisting of a series of contrasts where the lipid structure is fixed
    across the different contrasts
    """
    def __init__(self, headScatLen, tailScatLen, subPhaseSLD, superPhaseSLD, thick, apm, name=''):
        """
        Parameters
        ----------
        headScatLen: array-like
            the scattering lengths of the head groups of the different species contrasts
        tailScatLen: array-like
            the scattering lengths of the tail groups of the different species contrasts
        subPhaseSLD: array-like
            the SLD of the subphase for each of the contrasts
        superPhaseSLD: array-like
            the SLD of the superphase for each of the contrasts
        thick: float[2]
            the head[0] and the tail[1] estimated thicknesses
        apm: float
            an estimate of the system area per molecule
        name : str
            Name of this lipid component
        """
        super(SurfMono, self).__init__()
        n = len(headScatLen)
        if any(len(x) != n for x in [headScatLen, tailScatLen, subPhaseSLD, superPhaseSLD]):
            raise ValueError("The number of different contrasts is inconsistant!")
        if len(thick) != 2:
            raise ValueError("Both the head and tail layer thicknesses should be estimated")
        self.numberofcontrasts = len(headScatLen)
        self.headScatLen = {}
        self.tailScatLen = {}
        self.subPhaseSLD = {}
        self.superPhaseSLD = {}
        for i in range(0, self.numberofcontrasts):
            self.headScatLen['contrast%s' % i] = headScatLen[i]
            self.tailScatLen['contrast%s' % i] = tailScatLen[i]
            self.subPhaseSLD['contrast%s' % i] = subPhaseSLD[i]
            self.superPhaseSLD['contrast%s' % i] = superPhaseSLD[i]
        self.head_thick = possibly_create_parameter(thick[0], name='%s - head layer thickness' % name)
        self.tail_thick = possibly_create_parameter(thick[1], name='%s - tail layer thickness' % name)
        self.apm = apm
        self.tailSLDs = {}
        self.headSLDs = {}
        self.subSLDs = {}
        self.superSLDs = {}
        self.structures = {}
        self.tails = {}
        self.heads = {}
        self.subs = {}
        self.supers = {}
        self.head_rough = 0
        self.tail_rough = 0
        self.water_rough = 0
        self.head_layers = {}
        self.tail_layers = {}
        self.sub_layers = {}


    def guessSLD(self):
        a = self.tailScatLen['contrast0'] / (self.tail_thick.value * self.apm) * 1E6
        return a


    def setSLD(self):
        self.tailSLDs['contrast0'] = Parameter(self.guessSLD(), 'tail_layer_contrast0',
                                               bounds=(self.guessSLD() - (0.5 * self.guessSLD()), self.guessSLD() +
                                                       (0.5 * self.guessSLD())), vary=True)
        self.headSLDs['contrast0'] = Parameter(1., 'head_layer_contrast0')
        for i in range(1, self.numberofcontrasts):
            self.tailSLDs['contrast%s' % i] = Parameter(1, 'tail_layer_contrast%s' % i)
            self.headSLDs['contrast%s' % i] = Parameter(1, 'head_layer_contrast%s' % i)
            self.subSLDs['contrast%s' % i] = Parameter(self.subPhaseSLD['contrast%s' % i], 'sub%s' % i)
            self.superSLDs['contrast%s' % i] = Parameter(self.superPhaseSLD['contrast%s' % i], 'super%s' % i)
        for i in range(0, self.numberofcontrasts):
            self.tails['contrast%s' % i] = SLD(self.tailSLDs['contrast%s' % i], name='tail_contrast%s' % i)
            self.heads['contrast%s' % i] = SLD(self.headSLDs['contrast%s' % i], name='head_contrast%s' % i)
            self.subs['contrast%s' % i] = SLD(self.subPhaseSLD['contrast%s' % i], name='sub_contrast%s' % i)
            self.supers['contrast%s' % i] = SLD(self.superPhaseSLD['contrast%s' % i], name='super_contrast%s' % i)


    def setThickRough(self):
        self.head_thick.setp(bounds=(self.head_thick.value - (0.25 * self.head_thick.value),
                                     self.head_thick.value + (0.25 * self.head_thick.value)), vary=True)
        self.tail_thick.setp(bounds=(self.tail_thick.value - (0.25 * self.tail_thick.value),
                                     self.tail_thick.value + (0.25 * self.tail_thick.value)), vary=True)
        self.head_rough = Parameter(0.2 * self.head_thick.value, 'head_layer_rough',
                                    bounds=(0, 0.5 * self.head_thick.value), vary=True)
        self.tail_rough = Parameter(0.2 * self.tail_thick.value, 'tail_layer_rough',
                                    bounds=(0, 0.5 * self.tail_thick.value), vary=True)
        self.water_rough = Parameter(3.1, 'subphase_layer_rough')


    def setLayers(self):
        self.setSLD()
        self.setThickRough()
        for i in range(0, self.numberofcontrasts):
            self.head_layers['contrast%s' % i] = self.heads['contrast%s' % i](self.head_thick, self.head_rough)
            self.tail_layers['contrast%s' % i] = self.tails['contrast%s' % i](self.tail_thick, self.tail_rough)
            self.sub_layers['contrast%s' % i] = self.subs['contrast%s' % i](0., self.water_rough)


    def setConstraints(self):
        self.setLayers()
        vguess = 1 - (self.head_layers['contrast0'].sld.real.value * self.head_thick.value) / \
                     (self.headScatLen['contrast0'] * self.apm)
        self.head_layers['contrast0'].vfsolv.setp(vguess, bounds = (0., 0.999999), vary=True)
        for i in range(1, self.numberofcontrasts):
            self.head_layers['contrast%s' % i].vfsolv.constraint = self.head_layers['contrast0'].vfsolv
        if self.numberofcontrasts == 1:
            self.head_layers['contrast0'].sld.real.constraint = \
                (self.tail_layers['contrast0'].sld.real * self.tail_thick * self.headScatLen['contrast0']) / \
                (self.head_thick * self.tailScatLen['contrast0'] * (Parameter(1, '1') -
                                                                    self.head_layers['contrast0'].vfsolv))
        if self.numberofcontrasts == 2:
            self.head_layers['contrast0'].sld.real.constraint = \
                (self.tail_layers['contrast0'].sld.real * self.tail_thick * self.headScatLen['contrast0']) / \
                (self.head_thick * self.tailScatLen['contrast0'] * (Parameter(1, '1') -
                                                                    self.head_layers['contrast0'].vfsolv))
            self.head_layers['contrast1'].sld.real.constraint = \
                (self.head_layers['contrast0'].sld.real * self.headScatLen['contrast1']) / \
                (self.headScatLen['contrast0'])
            self.tail_layers['contrast1'].sld.real.constraint = \
                (self.head_layers['contrast1'].sld.real * self.head_thick * self.tailScatLen['contrast1'] *
                 (Parameter(1, '1') - self.head_layers['contrast1'].vfsolv)) / (self.tail_thick *
                                                                                self.headScatLen['contrast1'])
        if self.numberofcontrasts > 2:
            if self.numberofcontrasts % 2 != 0:
                for i in range(0, self.numberofcontrasts - 2, 2):
                    a = str(i)
                    b = str(i + 1)
                    c = str(i + 2)
                    self.head_layers['contrast%s' % a].sld.real.constraint = \
                        (self.tail_layers['contrast%s' % a].sld.real * self.tail_thick *
                         self.headScatLen['contrast%s' % a]) / (self.head_thick * self.tailScatLen['contrast%s' % a] *
                                                                (Parameter(1, '1') -
                                                                 self.head_layers['contrast%s' % a].vfsolv))
                    self.head_layers['contrast%s' % b].sld.real.constraint = \
                        (self.head_layers['contrast%s' % a].sld.real * self.headScatLen['contrast%s' % b]) / \
                        (self.headScatLen['contrast%s' % a])
                    self.tail_layers['contrast%s' % b].sld.real.constraint = \
                        (self.head_layers['contrast%s' % b].sld.real * self.head_thick *
                         self.tailScatLen['contrast%s' % b] * (Parameter(1, '1') -
                                                               self.head_layers['contrast%s' % b].vfsolv)) / \
                        (self.tail_thick * self.headScatLen['contrast%s' % b])
                    self.tail_layers['contrast%s' % c].sld.real.constraint = \
                        (self.tail_layers['contrast%s' % b].sld.real * self.tailScatLen['contrast%s' % c]) / \
                        (self.tailScatLen['contrast%s' % b])
                self.head_layers['contrast%s' % str(self.numberofcontrasts-1)].sld.real.constraint = \
                    (self.tail_layers['contrast%s' % str(self.numberofcontrasts-1)].sld.real * self.tail_thick *
                     self.headScatLen['contrast%s' % str(self.numberofcontrasts-1)]) / \
                    (self.head_thick * self.tailScatLen['contrast%s' % str(self.numberofcontrasts-1)] *
                     (Parameter(1, '1') - self.head_layers['contrast%s' % str(self.numberofcontrasts-1)].vfsolv))
            else:
                for i in range(0, self.numberofcontrasts - 2, 2):
                    a = str(i)
                    b = str(i + 1)
                    c = str(i + 2)
                    self.head_layers['contrast%s' % a].sld.real.constraint = \
                        (self.tail_layers['contrast%s' % a].sld.real * self.tail_thick *
                         self.headScatLen['contrast%s' % a]) / (self.head_thick * self.tailScatLen['contrast%s' % a] *
                                                                (Parameter(1, '1') -
                                                                 self.head_layers['contrast%s' % a].vfsolv))
                    self.head_layers['contrast%s' % b].sld.real.constraint = \
                        (self.head_layers['contrast%s' % a].sld.real * self.headScatLen['contrast%s' % b]) / \
                        (self.headScatLen['contrast%s' % a])
                    self.tail_layers['contrast%s' % b].sld.real.constraint = \
                        (self.head_layers['contrast%s' % b].sld.real * self.head_thick *
                         self.tailScatLen['contrast%s' % b] * (Parameter(1, '1') -
                                                               self.head_layers['contrast%s' % b].vfsolv)) / \
                        (self.tail_thick * self.headScatLen['contrast%s' % b])
                    self.tail_layers['contrast%s' % c].sld.real.constraint = \
                        (self.tail_layers['contrast%s' % b].sld.real * self.tailScatLen['contrast%s' % c]) / \
                        (self.tailScatLen['contrast%s' % b])
                self.head_layers['contrast%s' % str(self.numberofcontrasts-2)].sld.real.constraint = \
                    (self.tail_layers['contrast%s' % str(self.numberofcontrasts-2)].sld.real * self.tail_thick *
                     self.headScatLen['contrast%s' % str(self.numberofcontrasts-2)]) / \
                    (self.head_thick * self.tailScatLen['contrast%s' % str(self.numberofcontrasts-2)] *
                     (Parameter(1, '1') - self.head_layers['contrast%s' % str(self.numberofcontrasts-2)].vfsolv))
                self.head_layers['contrast%s' % str(self.numberofcontrasts-1)].sld.real.constraint = \
                    (self.head_layers['contrast%s' % str(self.numberofcontrasts-2)].sld.real *
                     self.headScatLen['contrast%s' % str(self.numberofcontrasts-1)]) / \
                    (self.headScatLen['contrast%s' % str(self.numberofcontrasts-2)])
                self.tail_layers['contrast%s' % str(self.numberofcontrasts-1)].sld.real.constraint = \
                    (self.head_layers['contrast%s' % str(self.numberofcontrasts-1)].sld.real *
                     self.head_thick * self.tailScatLen['contrast%s' % str(self.numberofcontrasts-1)] *
                     (Parameter(1, '1') - self.head_layers['contrast%s' % str(self.numberofcontrasts-1)].vfsolv)) / \
                    (self.tail_thick * self.headScatLen['contrast%s' % str(self.numberofcontrasts-1)])


    def getStructures(self):
        self.setConstraints()
        self.tail_layers['contrast0'].sld.real.setp(vary = True, bounds=(self.guessSLD() - (0.25 * self.guessSLD()),
                                                                         self.guessSLD() + (0.25 * self.guessSLD())))
        for i in range(0, self.numberofcontrasts):
            self.structures['contrast%s' %i] = (self.supers['contrast%s' % i] | self.tail_layers['contrast%s' % i] |
                                                self.head_layers['contrast%s' % i] | self.sub_layers['contrast%s' % i])

    @property
    def apmCalc(self):
        apmc = self.tailScatLen['contrast0'] / (self.tail_layers['contrast0'].sld.real.value * 1E-6 *
                                      self.tail_layers['contrast0'].thick.value)
        apmcerr = (self.tailScatLen['contrast0'] * self.tail_layers['contrast0'].thick.stderr) / \
                  (self.tail_layers['contrast0'].thick.value * self.tail_layers['contrast0'].thick.value *
                   self.tail_layers['contrast0'].sld.real.value * 1E-6)
        return apmc, apmcerr

    @property
    def molecularVolumes(self):
        head = self.headScatLen['contrast0'] / (self.head_layers['contrast0'].sld.real.value * 1E-6 *
                                      (1 - self.head_layers['contrast0'].vfsolv.value))
        tail = self.tailScatLen['contrast0'] / (self.tail_layers['contrast0'].sld.real.value * 1E-6)
        total = head + tail
        a = self.headScatLen[0] / (self.structures['contrast0'][1].sld.real.value * 1E-6 *
                                   self.structures['contrast0'][1].thick.value) * \
            self.structures['contrast0'][2].thick.stderr
        b = (self.structures['contrast0'][2].thick.value * self.tailScatLen[0]) / \
            ((self.structures['contrast0'][1].sld.real.value * 1E-6) ** 2 *
             self.structures['contrast0'][1].thick.value) * self.structures['contrast0'][1].sld.real.stderr * 1E-6
        c = (self.structures['contrast0'][2].thick.value * self.tailScatLen[0]) / \
            (self.structures['contrast0'][1].sld.real.value * 1E-6 *
             (self.structures['contrast0'][1].thick.value ** 2)) * self.structures['contrast0'][1].thick.stderr
        headerr = np.sqrt(a**2 + b**2 + c**2)
        tailerr = self.tailScatLen[0] / ((self.structures['contrast0'][1].sld.real.value * 1E-6) ** 2) * \
                  (self.structures['contrast0'][1].sld.real.stderr * 1E-6)
        totalerr = np.sqrt((headerr ** 2) + (tailerr ** 2))
        return head, headerr, tail, tailerr, total, totalerr
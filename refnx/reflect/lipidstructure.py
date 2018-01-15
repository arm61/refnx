import numpy as np
from six.moves import UserList
from refnx.analysis import Parameters, Parameter, possibly_create_parameter


class LipidStructure(UserList):
    def __init__(self, scatlens, widths, sep, apm, solv_number_density=0.0334277, name=''):
        """
        :param name: str
            name for the structure
        :param scatlens: array of floats
            scattering lengths for the different components of the structure
            [tail, head, solvent]
        :param widths: array of floats
            initial guesses are the widths of each functional form
            [tail, head, solvent]
        :param sep: array of floats
            initial guesses are the separation between the components
            [tail-solv, tail-head]
        :param apm: float
            initial guess at the area per molecule of the system
        """
        super(LipidStructure, self).__init__()
        self.name = name
        self.tail_b = possibly_create_parameter(scatlens[0], name='tail b')
        self.head_b = possibly_create_parameter(scatlens[1], name='head b')
        self.solv_b = possibly_create_parameter(scatlens[2], name='solvent b')
        self.tail_width = possibly_create_parameter(widths[0], name='tail width')
        self.head_width = possibly_create_parameter(widths[1], name='head width')
        self.solvent_width = possibly_create_parameter(widths[2], name='solvent width')
        self.tailsolv_sep = possibly_create_parameter(sep[0], name='tail-solv separation')
        self.tailhead_sep = possibly_create_parameter(sep[0], name='tail-head separation')
        self.apm = possibly_create_parameter(apm, name='area per molecule')
        self.solv_number_density = possibly_create_parameter(solv_number_density, name='solvent number density')

        self.tail_width.setp(widths[0], vary=True, bounds=(widths[0] - (0.25 * widths[0]),
                                                           widths[0] + (0.25 * widths[0])))
        self.head_width.setp(widths[1], vary=True, bounds=(widths[1] - (0.25 * widths[1]),
                                                           widths[1] + (0.25 * widths[1])))
        self.solvent_width.setp(widths[2], vary=True, bounds=(widths[2] - (0.25 * widths[2]),
                                                           widths[2] + (0.25 * widths[2])))
        self.tailsolv_sep.setp(float(sep[0]), vary=True, bounds=(sep[0] - (0.25 * sep[0]),
                                                          sep[0] + (0.25 * sep[0])))
        self.tailhead_sep.setp(float(sep[1]), vary=True, bounds=(sep[1] - (0.25 * sep[1]),
                                                          sep[1] + (0.25 * sep[1])))
        self.apm.setp(apm, vary=True, bounds=(apm - (0.25 * apm),
                                              apm + (0.25 * apm)))
        self.headsolv_sep = possibly_create_parameter(sep[0]-sep[1], name='head-solv separation')
        self.headsolv_sep.constraint = self.tailsolv_sep - self.tailhead_sep
        self.z = np.asarray([])

    @property
    def parameters(self):
        p = Parameters(name='Structure - {0}'.format(self.name))
        p_tail = Parameters(name='Tail parameters')
        p_tail.extend([self.tail_b, self.tail_width, self.tailsolv_sep, self.tailhead_sep])
        p_head = Parameters(name='Head parameters')
        p_head.extend([self.head_b, self.head_width, self.headsolv_sep])
        p_solv = Parameters(name='Solvent parameters')
        p_solv.extend([self.solv_b, self.solvent_width])
        p_apm = Parameters(name='System parameters')
        p_apm.extend([self.apm])
        p.extend([p_tail, p_head, p_solv, p_apm])
        return p

    def numberdensity(self):
        maximum = self.tail_width.value * 10
        self.z = np.arange(-maximum, maximum, 0.01)
        tail_nd = gaussian(gaussian_height(1 / self.apm.value, self.tail_width.value), self.z, self.tail_width.value, self.tailhead_sep.value)
        head_nd = gaussian(gaussian_height(1 / self.apm.value, self.head_width.value), self.z, self.head_width.value, 0)
        solv_nd = tanh(self.solv_number_density.value, self.z, self.solvent_width.value, self.headsolv_sep.value)
        return tail_nd, head_nd, solv_nd

    @property
    def components(self):
        return self.data

    def lnprob(self):
        """
        log-probability for the interfacial structure. Note that if a given
        component is present more than once in a Structure then it's log-prob
        will be counted twice.

        Returns
        -------
        lnprob : float
            log-prior for the Structure.
        """
        lnprob = 0
        for component in self.components:
            lnprob += component.lnprob()

        return lnprob


def gaussian(height, x, width, offset):
    return height * np.exp((-4 * np.square(x + offset)) / np.square(width))

def gaussian_height(area, width):
    return (2 * area) / (np.sqrt(np.pi) * width)

def tanh(height, x, width, offset):
    return height * (0.5 + (0.5 * np.tanh(np.divide(x - offset, width))))

import torch
from basis_func import real_sph_harm, bessel_basis
import sympy as sym

# Init: num_sph: dimension to expand form $\theta$
# Input: a tensor of theta value, calculated from arctan function(to prevent numerical problem(leading to **nan**))
# Output: a tensor(matrix) of expanded theta

# note: this expansion consider **only** the angular between two vector, 
#     so there is no need to envelop it.

class AngularBasisLayer(torch.nn.Module):
    def __init__(self, num_sph):
        super().__init__()
        self.num_sph = num_sph
        self.sph_formulas = real_sph_harm(num_sph)
        self.sph_funcs = []
        theta = sym.symbols('theta')
        self.modules = {'sin': torch.sin, 'cos': torch.cos}

        for i in range(num_sph):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_formulas[i][0], self.modules)(0)
                self.sph_funcs.append(lambda tensor: torch.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_formulas[i][0], self.modules))
            
    def forward(self, Angles):
        cbf = [f(Angles) for f in self.sph_funcs]
        cbf = torch.stack(cbf, dim= 0)

        return cbf.T

def AngularBasisLayer_func(Angles, num_sph = 16):
    sph_formulas = real_sph_harm(num_sph)
    sph_funcs = []
    theta = sym.symbols('theta')
    modules = {'sin': torch.sin, 'cos': torch.cos}
    for i in range(num_sph):
        if i == 0:
            first_sph = sym.lambdify([theta], sph_formulas[i][0], modules)(0)
            sph_funcs.append(lambda tensor: torch.zeros_like(tensor) + first_sph)
        else:
            sph_funcs.append(sym.lambdify([theta], sph_formulas[i][0], modules))

    cbf = [f(Angles) for f in sph_funcs]
    cbf = torch.stack(cbf, dim= 0)
    return cbf.T

from envelop import poly_envelop
class F_B_2D(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5):
        super(F_B_2D, self).__init__()

        assert num_radial <= 64

        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = poly_envelop(cutoff = 5.0, exponent=envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(num_spherical, num_radial)
        self.sph_harm_formulas = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify([theta], self.sph_harm_formulas[i][0], modules)(0)
                self.sph_funcs.append(lambda tensor: torch.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify([theta], self.sph_harm_formulas[i][0], modules))
            for j in range(num_radial):
                self.bessel_funcs.append(sym.lambdify([x], self.bessel_formulas[i][j], modules))

    def forward(self, d, Angles, edge_index_1):
        d_scaled = d / self.cutoff
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = torch.stack(rbf, dim=1)

        d_cutoff = self.envelope(d)
        rbf_env = d_cutoff[:, None] * rbf
        rbf_env = rbf_env[edge_index_1.long()]

        cbf = [f(Angles) for f in self.sph_funcs]
        cbf = torch.stack(cbf, dim=1)
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)

        return rbf_env * cbf

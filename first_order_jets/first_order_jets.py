"""
This is a library for 1st-order jets.
They make forward automatic differentiation possible.
We think of a jet-value as storing some information about
a differentiable function
f: R^num_parameters -> R at some point x in R^num_parameters,
namely the jet contains one scalar and one vector:
the value of f at the point x;
and the first partial derivatives of f at x (the gradient vector).
Given such information about f at x, and also given such information
about a differentiable function g at x, one can deduce such
value and 1st partial derivative at x information about
h := f + g, h:= f*g, h:= f / g,  h := exp(f), h := cos(f), etc.
and the forward automatic differentiation of any function f
that you know how to build from simple operations such as + - * / ^ sin cos etc.
"""
import numpy as np


class Jet1(object):
    def __init__(self, num_parameters):
        """
        say that f: R^num_parameters --> R
        is a real-valued expression / function
        of num_parameters real inputs.
        Then a Jet1 is used to store the value of a f
        at some value x of those parameters as well as
        the gradient of f at x.
        """
        self.num_parameters = num_parameters
        self.v = 0.0
        self.G = np.zeros(shape=(num_parameters,), dtype=np.double)


# NO ALLOCATION ALLOWED AFTER HERE:


def set_to_jet_of_parameter(jet, current_parameter_values, parameter_index):
    """
    Makes the jet be the parameter_index-ith parameter,
    with value current_parameter_values[parameter_index]
    """
    assert isinstance(jet, Jet1)
    assert isinstance(current_parameter_values, np.ndarray)
    assert current_parameter_values.ndim == 1
    assert jet.num_parameters == current_parameter_values.shape[0]
    num_parameters = current_parameter_values.shape[0]
    assert 0 <= parameter_index < num_parameters

    jet.v = current_parameter_values[parameter_index]
    # the gradient is all zeros save for 1
    jet.G *= 0.0
    jet.G[parameter_index] = 1.0


def set_jet_to_zero(j):
    assert isinstance(j, Jet1)
    j.v = 0.0
    j.G[:] = 0.0


def set_jet_to_constant(j, c):
    assert isinstance(j, Jet1)
    assert not isinstance(c, Jet1)
    j.v = c
    j.G[:] = 0.0


def add_jets(result, a, b):
    result.v = a.v + b.v
    result.G[:] = a.G + b.G


def jet_plus_equals_jet(victim_jet, summand):
    victim_jet.v += summand.v
    victim_jet.G += summand.G


def jet_minus_equals_jet(victim_jet, subtrahend):
    victim_jet.v -= subtrahend.v
    victim_jet.G -= subtrahend.G


def subtract_jets(result, a, b):
    result.v = a.v - b.v
    result.G[:] = a.G - b.G


def add_constant_to_jet(result, j, c):
    assert isinstance(j, Jet1)
    assert not isinstance(c, Jet1)
    result.v = j.v + c
    result.G[:] = j.G


def subtract_constant_from_jet(result, j, c):
    assert isinstance(j, Jet1)
    assert not isinstance(c, Jet1)
    result.v = j.v - c
    result.G[:] = j.G


def subtract_jet_from_constant(result, c, j):
    assert isinstance(j, Jet1)
    assert not isinstance(c, Jet1)
    result.v = c - j.v
    result.G[:] = - j.G


def scalar_multiply_jet(result, c, j):
    assert isinstance(j, Jet1)
    assert not isinstance(c, Jet1)
    result.v = c * j.v
    result.G[:] = c * j.G


def multiply_jets(result, a, b):
    """
    suppose h = f*g
    then h_x = f_x * g + f * g_x
    """
    result.v = a.v * b.v
    result.G[:] = b.v * a.G + a.v * b.G


def jet_times_equals_jet(a, b):
    """
    The jet version of a *= b
    """
    # In Python these assignments happen simultaneously:
    a.v, a.G[:] = (
        a.v * b.v,
        b.v * a.G + a.v * b.G,
    )


def multiply_3_jets(result, a, b, c):
    """
    result = a * b * c
    """
    set_jet_to_constant(result, 1.0)
    jet_times_equals_jet(result, a)
    jet_times_equals_jet(result, b)
    jet_times_equals_jet(result, c)


def divide_jets(result, a, b):
    """
    suppose h = f / g
    then h_x = f_x * g - f / g**2 * g_x = (f_x * g -  f * g_x) / (g ** 2)
    """
    assert b.v != 0
    result.v = a.v / b.v
    result.G[:] = (b.v * a.G - a.v * b.G) / b.v ** 2


def square_jet(result, j):
    """
    suppose h = f**2
    then h_x = 2 *  f * f_x
    """
    result.v = j.v * j.v
    result.G[:] = 2 * j.v * j.G


def square_root_jet(result, j):
    """
    suppose h = f ** 0.5
    then h_x = 0.5 * f ^ -0.5 * f_x
    """
    result.v = j.v ** 0.5
    result.G[:] = 0.5 * j.v ** (-0.5) * j.G


def sin_jet(result, j):
    """
    suppose h = sin(f)
    then h_x = cos(f) * f_x
    """
    result.v = np.sin(j.v)
    result.G[:] = np.cos(j.v) * j.G


def cos_jet(result, j):
    """
    suppose h = cos(f)
    then h_x = - sin(f) * f_x
    """
    result.v = np.cos(j.v)
    result.G[:] = -np.sin(j.v) * j.G


def jet_SO3_from_rodrigues(
    rodrigues_x,
    rodrigues_y,
    rodrigues_z,
    summand,
    norm_squared,
    angle,
    ux,
    uy,
    uz,
    cosine,
    one_minus_cos,
    sine,
    anti10,
    anti02,
    anti21,
    sym00,
    sym01,
    sym02,
    sym11,
    sym12,
    sym22,
    R00,
    R01,
    R02,
    R10,
    R11,
    R12,
    R20,
    R21,
    R22,
):
    """
    Given jet-valued Rodrigues vector coordinates
    rodrigues_x, rodrigues_y, rodrigues_z
    
    returns the
    9 jet-valued 
    SO3 = cos(angle) * I_3x3
    + 
    sin(angle) * [0  - uz   uy]
                 [uz    0  -ux]
                 [0    ux  -uy]
    +
    (1-cos(angle)) * [ux ux  ux uy  ux uz]
                     [uy ux  uy uy  uy uz]
                     [uz ux  uz uy  uz uz]
    ]
    """

    set_jet_to_zero(norm_squared)
    square_jet(summand, rodrigues_x)
    jet_plus_equals_jet(norm_squared, summand)

    square_jet(summand, rodrigues_y)
    jet_plus_equals_jet(norm_squared, summand)

    square_jet(summand, rodrigues_z)
    jet_plus_equals_jet(norm_squared, summand)

    square_root_jet(angle, norm_squared)

    # u = rodrigues / angle  # the unit length version is the rotation axis
    divide_jets(ux, rodrigues_x, angle)
    divide_jets(uy, rodrigues_y, angle)
    divide_jets(uz, rodrigues_z, angle)

    cos_jet(cosine, angle)
    sin_jet(sine, angle)

    """
    Basically this is what is happening:
    I_3x3 = np.eye(3)
    anti = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0],])
    SO3 = cosine * I_3x3 + sine * anti + (1 - cosine) * np.outer(u, u)
    """

    set_jet_to_zero(R00)
    set_jet_to_zero(R01)
    set_jet_to_zero(R02)
    set_jet_to_zero(R10)
    set_jet_to_zero(R11)
    set_jet_to_zero(R12)
    set_jet_to_zero(R20)
    set_jet_to_zero(R21)
    set_jet_to_zero(R22)

    # only the diagonals get this:
    jet_plus_equals_jet(R00, cosine)
    jet_plus_equals_jet(R11, cosine)
    jet_plus_equals_jet(R22, cosine)
    """
    np.sin(angle) * [  0  - uz   uy]
                    [ uz     0  -ux]
                    [-uy    ux    0]
    """

    multiply_jets(anti10, sine, uz)
    multiply_jets(anti02, sine, uy)
    multiply_jets(anti21, sine, ux)

    # only off-diagonals get this:
    jet_minus_equals_jet(R01, anti10)
    jet_plus_equals_jet(R10, anti10)

    jet_minus_equals_jet(R20, anti02)
    jet_plus_equals_jet(R02, anti02)

    jet_minus_equals_jet(R12, anti21)
    jet_plus_equals_jet(R21, anti21)

    subtract_jet_from_constant(one_minus_cos, 1.0, cosine)

    multiply_3_jets(sym00, one_minus_cos, ux, ux)
    multiply_3_jets(sym01, one_minus_cos, ux, uy)
    multiply_3_jets(sym02, one_minus_cos, ux, uz)
    multiply_3_jets(sym11, one_minus_cos, uy, uy)
    multiply_3_jets(sym12, one_minus_cos, uy, uz)
    multiply_3_jets(sym22, one_minus_cos, uz, uz)

    jet_plus_equals_jet(R00, sym00)
    jet_plus_equals_jet(R01, sym01)
    jet_plus_equals_jet(R02, sym02)

    jet_plus_equals_jet(R10, sym01)
    jet_plus_equals_jet(R11, sym11)
    jet_plus_equals_jet(R12, sym12)

    jet_plus_equals_jet(R20, sym02)
    jet_plus_equals_jet(R21, sym12)
    jet_plus_equals_jet(R22, sym22)

    # the SO3 is "returned" in these 9 output arguments:
    # R00 R01 R02
    # R10 R11 R12
    # R20 R21 R22

from autodiffpy.forward import log, sin, cos
from autodiffpy.demo import newtons_method
from pytest import approx
import numpy as np


def test_newtons_method_simple():
    f = lambda x: x

    zero = newtons_method(f, 10)
    assert zero == approx(0)


def test_newtons_method_more_complex():
    f = lambda x: (x - 2) * (x + 3)

    zero1 = newtons_method(f, 2.45)
    assert zero1 == approx(2)

    zero2 = newtons_method(f, -20)
    assert zero2 == approx(-3)


def test_newtons_method_weirdness():
    f = lambda x: log(sin(cos(x ** x))) + 1

    zero = newtons_method(f, 1)

    assert zero == approx(1.1575451)


def test_newtons_method_log():
    f = lambda x: log(x) - 1

    zero = newtons_method(f, 2)

    assert zero == approx(np.e)

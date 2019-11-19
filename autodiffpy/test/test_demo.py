from autodiffpy import Var, Log, Sin, Cos
from autodiffpy.demo import newtons_method
from pytest import approx
import numpy as np


def test_newtons_method_simple():
    f = Var("x")

    zero = newtons_method(f, 10)
    assert zero == approx(0)
    assert f(x=zero) == approx(0)


def test_newtons_method_more_complex():
    x = Var("x")

    f = (x - 2) * (x + 3)

    zero1 = newtons_method(f, 2.45)
    assert zero1 == approx(2)
    assert f(x=zero1) == approx(0)

    zero2 = newtons_method(f, -20)
    assert zero2 == approx(-3)
    assert f(x=zero2) == approx(0)


def test_newtons_method_weirdness():
    x = Var("x")

    f = Log(Sin(Cos(x ** x))) + 1

    zero = newtons_method(f, 1)

    assert zero == approx(1.1575451)
    assert f(x=zero) == approx(0)


def test_newtons_method_log():
    f = Log("x") - 1

    zero = newtons_method(f, 2)

    assert zero == approx(np.e)
    assert f(x=zero) == approx(0)

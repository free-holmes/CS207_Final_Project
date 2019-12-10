from autodiffpy.reverse import (
    Reverse,
    sin,
    cos,
    tan,
    sec,
    csc,
    cot,
    arcsin,
    arccos,
    arctan,
    exp,
    sinh,
    cosh,
    tanh,
    sech,
    csch,
    coth,
    ln,
    log2,
    log10,
    log,
    sqrt,
    rVector,
)
from pytest import approx, raises
import numpy as np


def test_eq_ne():
    x = Reverse(5)
    f = sin(x) + 2
    g = sin(x) + 2
    h = cos(x) + 2
    f.gradient_value = 1.0
    g.gradient_value = 1.0
    h.gradient_value = 1.0
    assert x == 5
    assert x != 7
    assert f == g
    assert f != h


def test_str():
    x = Reverse(5)
    f = sin(x) + 2
    f.gradient_value = 1.0
    assert str(f) == "value = 1.0410757253368614, gradient_value = 1.0"


def test_single_var():
    x = Reverse(1) + 2
    f = x
    f.gradient_value = 1.0
    assert f.value == approx(3)
    assert x.get_gradient() == approx(1)


def test_multiplication():
    x = Reverse(1)
    f = 4 * x
    f.gradient_value = 1.0
    assert f.value == approx(4)
    assert x.get_gradient() == approx(4)

    x = Reverse(1)
    f = x * 4
    f.gradient_value = 1.0
    assert f.value == approx(4)
    assert x.get_gradient() == approx(4)


def test_multiplication_by_variable():
    x = Reverse(1)
    y = Reverse(2)
    f = x * y
    f.gradient_value = 1.0
    assert f.value == approx(2)
    assert x.get_gradient() == approx(2)
    assert y.get_gradient() == approx(1)


def test_division():
    x = Reverse(2)
    f = 4 / x
    f.gradient_value = 1.0
    assert f.value == approx(2)
    assert x.get_gradient() == approx(-1)

    x = Reverse(5)
    f = x / 4
    f.gradient_value = 1.0
    assert f.value == approx(5 / 4)
    assert x.get_gradient() == approx(1 / 4)


def test_power():
    # Variable root
    x = Reverse(3)
    f = x ** 5
    f.gradient_value = 1.0
    assert f.value == approx(3 ** 5)
    assert x.get_gradient() == approx(5 * 3 ** 4)

    x = Reverse(2)
    f = x ** 2.5
    f.gradient_value = 1.0
    assert f.value == approx(2 ** 2.5)
    assert x.get_gradient() == approx(2.5 * 2 ** 1.5)

    # Numeric root
    x = Reverse(3)
    f = 5 ** x
    f.gradient_value = 1.0
    assert f.value == approx(5 ** 3)
    assert x.get_gradient() == approx(5 ** 3 * np.log(5))


def test_addition():
    x = Reverse(3)
    y = Reverse(3)
    f = 4 * x + y * -6
    f.gradient_value = 1.0
    assert f.value == approx(4 * 3 + 3 * -6)
    assert x.get_gradient() == approx(4)
    assert y.get_gradient() == approx(-6)

    x = Reverse(3)
    f = 4 + x ** 2
    f.gradient_value = 1.0
    assert f.value == approx(4 + 3 ** 2)
    assert x.get_gradient() == approx(2 * 3)


def test_subtraction():
    x = Reverse(5)
    f = 4 * x ** 2.3 - 3 / x
    f.gradient_value = 1.0
    assert f.value == approx(4 * 5 ** 2.3 - 3 / 5)
    assert x.get_gradient() == approx(4 * 2.3 * 5 ** 1.3 + 3 / (5 ** 2))

    x = Reverse(2)
    f = 4 - x
    f.gradient_value = 1.0
    assert f.value == approx(2)
    assert x.get_gradient() == approx(-1)


def test_weirdness():
    x = Reverse(2)
    f = x ** x
    f.gradient_value = 1.0
    assert f.value == approx(4)
    assert x.get_gradient() == approx(4 + np.log(16))


def test_sin():
    x = Reverse(5)
    f = sin(x) + sin(5)
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(5) + np.sin(5))
    assert x.get_gradient() == approx(np.cos(5))

    # Nested case
    x = Reverse(5)
    f = sin(2 * (-x))
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(-2 * 5))
    assert x.get_gradient() == approx(-2 * np.cos(-2 * 5))


def test_cos():
    x = Reverse(5)
    y = Reverse(5)
    f = sin(x) * cos(y) + cos(3)
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(5) * np.cos(5) + np.cos(3))
    assert y.get_gradient() == approx((-np.sin(5)) * np.sin(5))

    # Nested case
    x = Reverse(4)
    y = Reverse(3)
    f = sin(cos(x * y))
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(np.cos(3 * 4)))
    assert x.get_gradient() == approx(-3 * np.sin(12) * np.cos(np.cos(12)))


def test_arc_trig_funcs():
    x = Reverse(0.5)
    f = arcsin(x) + arcsin(0.5)
    f.gradient_value = 1.0
    assert f.value == approx(np.arcsin(0.5) + np.arcsin(0.5))
    assert x.get_gradient() == 1 / np.sqrt(1 - 0.5 ** 2)

    x = Reverse(0.5)
    f = arccos(x) + arccos(1)
    f.gradient_value = 1.0
    assert f.value == approx(np.arccos(0.5) + np.arccos(1))
    assert x.get_gradient() == -1 / np.sqrt(1 - 0.5 ** 2)

    x = Reverse(0.2)
    f = arctan(x) + arctan(0.1)
    f.gradient_value = 1.0
    assert f.value == approx(np.arctan(0.2) + np.arctan(0.1))
    assert x.get_gradient() == 1 / (1 + 0.2 ** 2)


def test_log():
    x = Reverse(4)
    f = log(x) + log(5) + ln(5)
    f.gradient_value = 1.0
    assert f.value == approx(np.log(4) + 2 * np.log(5))
    assert x.get_gradient() == approx(1 / 4)

    # Harder case
    x = Reverse(3)
    y = Reverse(7)
    f = log(x ** y)
    f.gradient_value = 1.0
    assert f.value == approx(np.log(3 ** 7))
    assert x.get_gradient() == approx((7 * (3 ** 6)) * 1 / (3 ** 7))

    # log2 and log10
    x = Reverse(3)
    f = log2(x) + log10(x)
    f.gradient_value = 1.0
    assert f.value == approx(np.log2(3) + np.log10(3))
    assert x.get_gradient() == approx(1 / (3 * np.log(10)) + 1 / (3 * np.log(2)))


def test_tan():
    x = Reverse(2)
    f = tan(x)
    f.gradient_value = 1.0
    assert f.value == approx(np.tan(2))
    assert x.get_gradient() == approx(1 / np.cos(2) ** 2)


def test_cot():
    x = Reverse(5)
    f = cot(x)
    f.gradient_value = 1.0
    assert f.value == approx(1 / np.tan(5))
    assert x.get_gradient() == approx(-1 / np.sin(5) ** 2)


def test_sec():
    x = Reverse(2)
    f = sec(x)
    f.gradient_value = 1.0
    assert f.value == approx(1 / np.cos(2))
    assert x.get_gradient() == approx(np.tan(2) / np.cos(2))


def test_cosec():
    x = Reverse(-1)
    f = csc(x)
    f.gradient_value = 1.0
    assert f.value == approx(1 / np.sin(-1))
    assert x.get_gradient() == approx(-1 / np.tan(-1) / np.sin(-1))


def test_exp():
    x = Reverse(0)
    f = exp(x) + exp(0)
    f.gradient_value = 1.0
    assert f.value == approx(2)
    assert x.get_gradient() == approx(1)


def test_sqrt():
    x = Reverse(2)
    f = sqrt(x)
    f.gradient_value = 1.0
    assert f.value == approx(2 ** 0.5)
    assert x.get_gradient() == approx(0.5 * (2 ** (-0.5)))


def test_other_trig_funcs():
    x = Reverse(2)
    f = sinh(x) + sinh(2)
    f.gradient_value = 1.0
    assert f.value == approx(2 * np.sinh(2))
    assert x.get_gradient() == approx(np.cosh(2))

    x = Reverse(2)
    f = cosh(x) + cosh(2)
    f.gradient_value = 1.0
    assert f.value == approx(2 * np.cosh(2))
    assert x.get_gradient() == approx(np.sinh(2))

    x = Reverse(2)
    f = tanh(x) + tanh(2)
    f.gradient_value = 1.0
    assert f.value == approx(2 * np.tanh(2))
    assert x.get_gradient() == approx(
        (np.cosh(2) ** 2 - np.sinh(2) ** 2) / np.cosh(2) ** 2
    )

    x = Reverse(2)
    f = sech(x) + sech(2)
    f.gradient_value = 1.0
    assert f.value == approx(2 * 1 / np.cosh(2))
    assert x.get_gradient() == approx((-1 / np.cosh(2)) * (np.sinh(2) / np.cosh(2)))

    x = Reverse(2)
    f = csch(x) + csch(2)
    f.gradient_value = 1.0
    assert f.value == approx(2 * 1 / np.sinh(2))
    assert x.get_gradient() == approx((-1 / np.sinh(2)) * (np.cosh(2) / np.sinh(2)))

    x = Reverse(2)
    f = coth(x) + coth(2)
    f.gradient_value = 1.0
    assert f.value == approx(2 * (np.cosh(2) / np.sinh(2)))
    assert x.get_gradient() == approx(-1 / (np.sinh(2) ** 2))


def test_vector():
    x = Reverse(1)
    y = Reverse(2)
    functions = [x * 2 * y + y ** 3, 2 * x ** 2 * y, 3 * y]
    vector = rVector(functions)

    # Check for get_values()
    assert vector.values == [12, 4, 6]
    # Check for get_gradients()
    assert vector.get_gradients(x) == [4.0, 8.0, 0]
    assert vector.get_gradients(y) == [14.0, 2.0, 3.0]

from autodiffpy.forward import (
    Forward,
    fVector,
    sin,
    cos,
    tan,
    sec,
    csc,
    cot,
    arcsin,
    arccos,
    arctan,
    arcsec,
    arccsc,
    arccot,
    exp,
    sqrt,
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
    logistic,
)
from pytest import approx, raises
import numpy as np


def test_constant():
    f = Forward(1)

    assert f.value == approx(1)
    assert f.get_gradient("x") == approx(0)
    assert f.get_gradient("y") == approx(0)


def test_var():
    f = Forward("x", 0)

    assert f.value == approx(0)
    assert f.get_gradient("x") == approx(1)

    f = Forward("x", -10)
    assert f.get_gradient("x") == approx(1)
    assert f.get_gradient("y") == approx(0)


def test_multi_dim_var():
    f = Forward("x", 0) * Forward("y", 0)

    assert f.value == approx(0)
    assert f.get_gradient("x") == approx(0)
    assert f.get_gradient("y") == approx(0)

    f = Forward("x", 2) * Forward("y", -2)

    assert f.value == approx(-4)
    assert f.get_gradient("x") == approx(-2)
    assert f.get_gradient("y") == approx(2)


def test_multiplication():
    f = 4 * Forward("x", 1)

    assert f.value == approx(4)
    assert f.get_gradient("x") == approx(4)

    f = Forward("x", 2) * 4

    assert f.value == approx(8)
    assert f.get_gradient("x") == approx(4)


def test_multiplication_by_variable():
    x = Forward("x", 2)
    f = x * x

    assert f.value == approx(4)
    assert f.get_gradient("x") == approx(4)


def test_division():
    x = Forward("x", -2)
    f = 4 / x

    assert f.value == approx(4 / -2)
    assert f.get_gradient("x") == approx(-1 * 4 / (-2) ** 2)

    f = x / 4

    assert f.value == approx(-2 / 4)

    assert f.get_gradient("x") == approx(1 / 4)


def test_power():
    x = Forward("x", 5)
    f = x ** 5

    assert f.value == approx(5 ** 5)
    assert f.get_gradient("x") == approx(5 * 5 ** 4)

    x = Forward("x", 0)
    f = x ** 5

    assert f.value == approx(0)
    assert f.get_gradient("x") == approx(0)

    f = Forward("x", -5) ** 2

    assert f.value == approx((-5) ** 2)
    assert f.get_gradient("x") == approx(2 * -5)


def test_power_weird():
    f = 5 ** Forward("x", 4)

    assert f.value == approx(5 ** 4)
    assert f.get_gradient("x") == approx(5 ** 4 * np.log(5))


def test_addition():
    x = Forward("x", 2)
    f = 4 * x + x * -6

    assert f.value == approx(4 * 2 + 2 * -6)
    assert f.get_gradient("x") == approx(4 + -6)


def test_radd():
    f = 4 + Forward("x", 2) ** 2

    assert f.value == approx(4 + 2 ** 2)
    assert f.get_gradient("x") == approx(2 * 2)


def test_subtraction():
    y = Forward("y", 3)
    f = 4 * y ** 2.3 - 3 / y

    assert f.value == approx(4 * 3 ** 2.3 - 3 / 3)
    assert f.get_gradient("y") == approx(4 * 2.3 * 3 ** 1.3 + 3 / 3 ** 2)


def test_right_subtraction():
    f = 4 - Forward("x", 3)

    assert f.value == approx(1)
    assert f.get_gradient("x") == approx(-1)


def test_weirdness():
    x = Forward("x", 2)
    f = x ** x

    assert f.value == approx(4)
    assert f.get_gradient("x") == approx(4 + np.log(16))

    x = Forward("x", -1.2)
    f = x ** x

    assert f.value == approx((-1.2) ** (-1.2))

    with raises(ValueError):
        f.get_gradient("x")


def test_missing_value():
    with raises(ValueError):
        f = Forward("x")


def test_sin():
    x = Forward("x", 2)
    f = sin(x)

    assert f.value == approx(np.sin(2))
    assert f.get_gradient("x") == approx(np.cos(2))


def test_sin_chain():
    val = np.pi + 1
    x = Forward("x", val)
    f = sin(2 * x)

    assert f.value == approx(np.sin(2 * val))
    assert f.get_gradient("x") == approx(2 * np.cos(2 * val))


def test_cos():

    f = cos(Forward("y", -2))

    assert f.value == approx(np.cos(-2))
    assert f.get_gradient("y") == approx(-np.sin(-2))


def test_nested_sin_cos():
    x = Forward("x", 2)
    y = Forward("y", 5)
    f = sin(cos(x * y))

    assert f.value == approx(np.sin(np.cos(2 * 5)))
    assert f.get_gradient("x") == approx(-5 * np.sin(10) * np.cos(np.cos(10)))
    assert f.get_gradient("y") == approx(-2 * np.sin(10) * np.cos(np.cos(10)))


def test_log():
    x = Forward("x", 2)
    f = log(x)

    assert f.value == approx(np.log(2))
    assert f.get_gradient("x") == approx(1 / 2)


def test_log_weird():
    x = Forward("x", 2)
    y = Forward("y", 5)

    f = log(x ** y)

    assert f.value == approx(np.log(2 ** 5))
    assert f.get_gradient("x") == approx(5 / 2)
    assert f.get_gradient("y") == approx(np.log(2))
    assert f.get_gradient("z") == approx(0)


def test_log_negative():
    x = Forward("x", -2)

    f = log(x)

    assert np.isnan(f.value)
    assert f.get_gradient("x") == approx(1 / -2)


def test_sin_of_const():
    x = Forward(2)
    f = sin(x)

    assert f.value == approx(np.sin(2))
    assert f.get_gradient("x") == approx(0)
    assert f.get_gradient("asdf") == approx(0)


def test_bad_coercion():
    with raises(ValueError):
        None * sin("x")


def test_vector():
    x = Forward("x", 2)
    y = Forward("y", -3)
    f = fVector([x, y])

    assert f.value == approx([2, -3])
    assert f.get_gradient("x") == approx([1, 0])
    assert f.get_gradient("y") == approx([0, 1])
    assert f.get_gradient("z") == approx([0, 0])


def test_vector_more():
    x = Forward("x", -3.5)

    f = fVector([x, 2 * x ** x])

    assert f.value == approx([-3.5, 2 * (-3.5) ** (-3.5)])
    with raises(ValueError):
        f.get_gradient("x")


def test_tan():
    x = Forward("x", 2)
    f = tan(x)

    assert f.value == approx(np.tan(2))
    assert f.get_gradient("x") == approx(1 / np.cos(2) ** 2)


def test_sec():
    x = Forward("x", 2)
    f = sec(x)

    assert f.value == approx(1 / np.cos(2))
    assert f.get_gradient("x") == approx(np.tan(2) / np.cos(2))


def test_cosec():
    x = Forward("x", -1)
    f = csc(x)

    assert f.value == approx(1 / np.sin(-1))
    assert f.get_gradient("x") == approx(-1 / np.tan(-1) / np.sin(-1))


def test_cot():
    f = cot(Forward("x", 5))

    assert f.value == approx(1 / np.tan(5))
    assert f.get_gradient("x") == approx(-1 / np.sin(5) ** 2)


def test_eq():
    x = Forward("x", 23)
    f = x == x

    assert f

    f = Forward("x", 1) == Forward("y", 2)

    assert not f

    f = Forward("x", 1) == Forward("y", 1)
    assert f


def test_lt():
    f = Forward("x", 1) < Forward("y", 2)
    assert f

    f = Forward("x", 2) < Forward("y", 1)
    assert not f

    f = Forward("x", 1) < Forward("y", 1)
    assert not f


def test_gt():
    f = Forward("x", 24) > Forward("y", 1)
    assert f

    f = Forward("x", 1) > Forward("y", 24)
    assert not f

    f = Forward("x", 2) > Forward("y", 2)
    assert not f


def test_lte():
    f = Forward("x", -24) <= Forward("y", 45)
    assert f

    f = Forward("x", 36) <= Forward("y", -2)
    assert not f

    f = Forward("x", 37) <= Forward("x", 37)
    assert f


def test_gte():
    f = Forward("x", 24) >= Forward("y", 2)
    assert f

    f = Forward("x", -2) >= Forward("y", 45)
    assert not f

    f = Forward("x", 3) >= Forward("y", 3)
    assert f


def test_ne():
    f = Forward("x", 48) != Forward("y", 1)
    assert f

    f = Forward("x", 24) != Forward("y", 24)
    assert not f

    f = Forward("x", -1) != Forward("y", 23)
    assert f


def test_comparison_with_math():
    x = Forward("x", 2)

    f = x ** 2 < x ** 3
    assert f

    f = x ** x ** 2 >= x ** 3
    assert f

    f = x ** x ** 3 <= x ** 0.5
    assert not f


def test_exp():
    x = Forward("x", 2)
    f = exp(x) + exp(0)
    assert f.value == approx(np.exp(2) + 1)
    assert f.get_gradient("x") == approx(np.exp(2))


def test_sqrt():
    x = Forward("x", 2)
    f = sqrt(x)
    assert f.value == approx(2 ** 0.5)
    assert f.get_gradient("x") == approx(0.5 * (2 ** (-0.5)))


def test_sqrt_imaginary():
    x = Forward("y", -3)
    f = sqrt(x)
    assert f.value == approx((-3) ** 0.5)
    assert f.get_gradient("y") == approx(0.5 * (-3) ** (-0.5))


def test_other_trig_funcs():
    x = Forward("x", 2)
    f = sinh(x) + sinh(2)
    assert f.value == approx(2 * np.sinh(2))
    assert f.get_gradient("x") == approx(np.cosh(2))

    f = cosh(x) + cosh(2)
    assert f.value == approx(2 * np.cosh(2))
    assert f.get_gradient("x") == approx(np.sinh(2))

    f = tanh(x) + tanh(2)
    assert f.value == approx(2 * np.tanh(2))
    assert f.get_gradient("x") == approx(
        (np.cosh(2) ** 2 - np.sinh(2) ** 2) / np.cosh(2) ** 2
    )

    f = sech(x) + sech(2)
    assert f.value == approx(2 * 1 / np.cosh(2))
    assert f.get_gradient("x") == approx((-1 / np.cosh(2)) * (np.sinh(2) / np.cosh(2)))

    f = csch(x) + csch(2)
    assert f.value == approx(2 * 1 / np.sinh(2))
    assert f.get_gradient("x") == approx((-1 / np.sinh(2)) * (np.cosh(2) / np.sinh(2)))

    f = coth(x) + coth(2)
    assert f.value == approx(2 * (np.cosh(2) / np.sinh(2)))
    assert f.get_gradient("x") == approx(-1 / (np.sinh(2) ** 2))


def test_neg():
    x = Forward("x", 3)
    f = -x

    assert f.value == approx(-3)
    assert f.get_gradient("x") == approx(-1)


def test_pos():
    x = Forward("x", -3)
    f = +x

    assert f.value == approx(-3)
    assert f.get_gradient("x") == approx(1)


def test_ln():
    x = Forward("x", 0.5)
    f = ln(x) ** 2

    assert f.value == approx(np.log(0.5) ** 2)
    assert f.get_gradient("x") == approx(2 * np.log(0.5) / 0.5)


def test_log2():
    x = Forward("x", 3)
    f = log2(x) ** 3

    assert f.value == approx(np.log2(3) ** 3)
    assert f.get_gradient("x") == approx(1 / np.log(2) ** 3 / 3 * np.log(3) ** 2 * 3)


def test_log10():
    x = Forward("x", 3)
    f = log10(x) * 2

    assert f.value == approx(np.log10(3) * 2)
    assert f.get_gradient("x") == approx(1 / np.log(10) * 2 / 3)


def test_arcsin():
    x = Forward("x", 0.4)
    f = arcsin(2 * x)

    assert f.value == approx(np.arcsin(2 * 0.4))
    assert f.get_gradient("x") == approx(1 / np.sqrt(1 - 0.8 ** 2) * 2)


def test_arccos():
    x = Forward("x", 0.1)
    f = arccos(2 * x)

    assert f.value == approx(np.arccos(2 * 0.1))
    assert f.get_gradient("x") == approx(-1 / np.sqrt(1 - 0.2 ** 2) * 2)


def test_arctan():
    x = Forward("x", 3.5)
    f = arctan(3 * x)

    assert f.value == approx(np.arctan(3 * 3.5))
    assert f.get_gradient("x") == approx(1 / (1 + (3 * 3.5) ** 2) * 3)


def test_arcsec():
    x = Forward("x", 3)
    f = arcsec(x ** 2)

    assert f.value == approx(np.arccos(1 / 9))
    assert f.get_gradient("x") == approx(1 / (abs(9) * np.sqrt(81 - 1)) * 2 * 3)


def test_arccsc():
    x = Forward("x", -4)
    f = arccsc(2 * x)

    assert f.value == approx(np.arcsin(1 / (2 * -4)))
    y = 2 * -4
    assert f.get_gradient("x") == approx(-1 / (abs(y) * np.sqrt(y ** 2 - 1)) * 2)


def test_arccot():
    x = Forward("x", 0.5)

    f = arccot(x)

    assert f.value == approx(np.arctan(1 / 0.5))
    assert f.get_gradient("x") == approx(-1 / (1 + 0.5 ** 2))


def test_sondak():
    x = Forward("x", 10)

    f = cos(x) ** 2

    assert f.value == approx(np.cos(10) ** 2)
    assert f.get_gradient("x") == approx(2 * np.cos(10) * -np.sin(10))


def test_logistic():
    l = lambda x: 1 / (1 + np.e ** -x)

    x = Forward("x", 0)
    f = logistic(x)

    assert f.value == approx(0.5)
    assert f.get_gradient("x") == approx(l(0) * (1 - l(0)))

    x = Forward("x", -5)
    f = logistic(x)

    assert f.value == approx(l(-5))
    assert f.get_gradient("x") == approx(l(-5) * (1 - l(-5)))

    x = Forward("x", 20)
    f = logistic(x)

    assert f.value == approx(l(20))
    assert f.get_gradient("x") == approx(l(20) * (1 - l(20)))


def test_invalid_creation():
    with raises(ValueError):
        _x = Forward("x")

    with raises(ValueError):
        Forward("x", "y")

    with raises(ValueError):
        Forward(2, 3)

    with raises(ValueError):
        Forward(2, 3, 4)

    with raises(ValueError):
        Forward()

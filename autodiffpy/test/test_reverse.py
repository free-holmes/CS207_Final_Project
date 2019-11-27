from autodiffpy import Reverse, sin, cos, tan, sec, csc, cot, exp, sinh, cosh, tanh, sech, csch, coth, ln, log, sqrt
from pytest import approx, raises
import numpy as np

def test_single_var():
    x = Reverse(1)
    f = x
    f.gradient_value = 1.0
    assert x.value == approx(1)
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
    assert f == approx(4)
    assert x.get_gradient() == approx(4)


def test_multiplication_by_variable():
    x = Reverse(1)
    y = Reverse(2)
    f = x * y
    f.gradient_value = 1.0
    assert f.value = 2
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
    x = Reverse(3)
    f = x ** 5
    f.gradient_value = 1.0
    assert f.value == approx(3 ** 5)
    assert x.get_gradient() == approx(5 * 3 ** 4)

    x = Reverse(2)
    f = x ** 2.5
    f.gradient_value = 1.0
    assert f == approx(2 ** 2.5)
    assert x.get_gradient() == approx(2.5 * 2 ** 1.5)


def test_power_weird():
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


# def test_radd():
#     f = 4 + Var("x") ** 2

#     assert f(x=2) == approx(4 + 2 ** 2)
#     assert f.derivative("x", x=3) == approx(2 * 3)


# def test_subtraction():
#     f = 4 * Var("y") ** 2.3 - 3 / Var("y")

#     assert f(y=3) == approx(4 * 3 ** 2.3 - 3 / 3)
#     assert f.derivative("y", y=5) == approx(4 * 2.3 * 5 ** 1.3 + 3 / 5 ** 2)


# def test_right_subtraction():
#     f = 4 - Var("x")

#     assert f(x=3) == approx(1)
#     assert f.derivative("x", x=2) == approx(-1)


# def test_weirdness():
#     f = Var("x") ** Var("x")

#     assert f(x=2) == approx(4)
#     assert f.derivative("x", x=2) == approx(4 + np.log(16))


# def test_implicit_string():
#     f = Var("x") * "x"

#     assert f(x=5) == approx(5 ** 2)
#     assert f.derivative("x", x=2) == approx(2 * 2)


# def test_sin():
#     f = Sin("x")

#     assert f(x=2) == approx(np.sin(2))
#     assert f.derivative("x", x=5) == approx(np.cos(5))


# def test_sin_chain():
#     f = Sin(2 * Var("x"))

#     assert f(x=2) == approx(np.sin(2 * 2))
#     assert f.derivative("x", x=5) == approx(2 * np.cos(2 * 5))


# def test_cos():
#     f = Cos("y")

#     assert f(y=-2) == approx(np.cos(-2))
#     assert f.derivative("y", y=5) == approx(-np.sin(5))


# def test_nested_sin_cos():
#     f = Sin(Cos(Var("x") * Var("y")))

#     assert f(x=2, y=5) == approx(np.sin(np.cos(2 * 5)))
#     assert f.derivative("x", x=4, y=3) == approx(-3 * np.sin(12) * np.cos(np.cos(12)))


# def test_log():
#     f = Log("x")

#     assert f(x=2) == approx(np.log(2))
#     assert f.derivative("x", x=4) == approx(1 / 4)


# def test_log_weird():
#     f = Log(Var("x") ** Var("y"))

#     assert f(x=2, y=5) == approx(np.log(2 ** 5))
#     assert f.derivative("x", x=3, y=72) == approx(24)


# def test_sin_of_const():
#     f = Sin(2)

#     assert f(x=2) == approx(np.sin(2))
#     assert f.derivative("x") == approx(0)


# def test_bad_coercion():
#     with raises(ValueError):
#         _f = None * Sin("x")


# def test_vector():
#     f = Vector([Var("x"), Var("y")])

#     assert f(x=2, y=3) == approx([2, 3])
#     assert f.derivative("x", x=-2, y=5) == approx([1, 0])


# def test_function_of_vector():
#     f = Log(Vector([Var("x"), Var("x") ** 2]))

#     assert f(x=2) == approx([np.log(2), np.log(4)])
#     assert f.derivative("x", x=5) == approx([1 / 5, 2 / 5])


# def test_tan():
#     f = Tan("x")

#     assert f(x=2) == approx(np.tan(2))
#     assert f.derivative("x", x=2) == approx(1 / np.cos(2) ** 2)


# def test_sec():
#     f = Sec("x")

#     assert f(x=2) == approx(1 / np.cos(2))
#     assert f.derivative("x", x=2) == approx(np.tan(2) / np.cos(2))


# def test_cosec():
#     f = Csc("x")

#     assert f(x=-1) == approx(1 / np.sin(-1))
#     assert f.derivative("x", x=-1) == approx(-1 / np.tan(-1) / np.sin(-1))


# def test_cot():
#     f = Cot("x")

#     assert f(x=5) == approx(1 / np.tan(5))
#     assert f.derivative("x", x=5) == approx(-1 / np.sin(5) ** 2)


# def test_missing_arg():
#     f = Var("x") ** 2

#     with raises(ValueError):
#         f()

#     with raises(ValueError):
#         f.derivative("x")
from autodiffpy.forward import (
    Var,
    Constant,
    Log,
    Vector,
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,
    Arcsin,
    Arccos,
    Arctan,
    Arcsec,
    Arccsc,
    Arccot,
    Exp,
    Sqrt,
    Sinh,
    Cosh,
    Tanh,
    Sech,
    Csch,
    Coth,
    Ln,
    Log2,
    Log10,
)
from pytest import approx, raises
import numpy as np


def test_constant():
    f = Constant(1)

    assert f(x=0) == approx(1)
    assert f.derivative("x") == approx(0)
    assert f.derivative("y") == approx(0)


def test_var():
    f = Var("x")

    assert f(x=0) == approx(0)
    assert f(x=10) == approx(10)

    assert f.derivative("x") == approx(1)
    assert f.derivative("y") == approx(0)


def test_multi_dim_var():
    f = Var("x") * Var("y")

    assert f(x=0, y=0) == approx(0)
    assert f(x=2, y=5) == approx(10)

    assert f.derivative("x", x=2, y=2) == approx(2)
    assert f.derivative("y", x=3, y=5) == approx(3)


def test_multiplication():
    f = 4 * Var("x")

    assert f(x=1) == approx(4)
    assert f.derivative("x", x=2) == approx(4)

    f = Var("x") * 4

    assert f(x=2) == approx(8)
    assert f.derivative("x", x=0) == approx(4)


def test_multiplication_by_variable():
    f = Var("x") * Var("x")

    assert f(x=2) == approx(4)
    assert f.derivative("x", x=3) == approx(6)


def test_division():
    f = 4 / Var("x")

    assert f(x=3) == approx(4 / 3)
    assert f.derivative("x", x=2) == approx(-1)

    f = Var("x") / 4

    assert f(x=5) == approx(5 / 4)

    assert f.derivative("x", x=5) == approx(1 / 4)


def test_power():
    f = Var("x") ** 5

    assert f(x=5) == approx(5 ** 5)
    assert f.derivative("x", x=3) == approx(5 * 3 ** 4)

    f = Var("x") ** 2.5

    assert f(x=2) == approx(2 ** 2.5)
    assert f.derivative("x", x=2) == approx(2.5 * 2 ** 1.5)


def test_power_weird():
    f = 5 ** Var("x")

    assert f(x=4) == approx(5 ** 4)
    assert f.derivative("x", x=3) == approx(5 ** 3 * np.log(5))


def test_addition():
    f = 4 * Var("x") + Var("x") * -6

    assert f(x=2) == approx(4 * 2 + 2 * -6)
    assert f.derivative("x", x=3) == approx(4 + -6)


def test_radd():
    f = 4 + Var("x") ** 2

    assert f(x=2) == approx(4 + 2 ** 2)
    assert f.derivative("x", x=3) == approx(2 * 3)


def test_subtraction():
    f = 4 * Var("y") ** 2.3 - 3 / Var("y")

    assert f(y=3) == approx(4 * 3 ** 2.3 - 3 / 3)
    assert f.derivative("y", y=5) == approx(4 * 2.3 * 5 ** 1.3 + 3 / 5 ** 2)


def test_right_subtraction():
    f = 4 - Var("x")

    assert f(x=3) == approx(1)
    assert f.derivative("x", x=2) == approx(-1)


def test_weirdness():
    f = Var("x") ** Var("x")

    assert f(x=2) == approx(4)
    assert f.derivative("x", x=2) == approx(4 + np.log(16))


def test_implicit_string():
    f = Var("x") * "x"

    assert f(x=5) == approx(5 ** 2)
    assert f.derivative("x", x=2) == approx(2 * 2)


def test_sin():
    f = Sin("x")

    assert f(x=2) == approx(np.sin(2))
    assert f.derivative("x", x=5) == approx(np.cos(5))


def test_sin_chain():
    f = Sin(2 * Var("x"))

    assert f(x=2) == approx(np.sin(2 * 2))
    assert f.derivative("x", x=5) == approx(2 * np.cos(2 * 5))


def test_cos():
    f = Cos("y")

    assert f(y=-2) == approx(np.cos(-2))
    assert f.derivative("y", y=5) == approx(-np.sin(5))


def test_nested_sin_cos():
    f = Sin(Cos(Var("x") * Var("y")))

    assert f(x=2, y=5) == approx(np.sin(np.cos(2 * 5)))
    assert f.derivative("x", x=4, y=3) == approx(-3 * np.sin(12) * np.cos(np.cos(12)))


def test_log():
    f = Log("x")

    assert f(x=2) == approx(np.log(2))
    assert f.derivative("x", x=4) == approx(1 / 4)


def test_log_weird():
    f = Log(Var("x") ** Var("y"))

    assert f(x=2, y=5) == approx(np.log(2 ** 5))
    assert f.derivative("x", x=3.0, y=72) == approx(24)


def test_sin_of_const():
    f = Sin(2)

    assert f(x=2) == approx(np.sin(2))
    assert f.derivative("x") == approx(0)


def test_bad_coercion():
    with raises(ValueError):
        _f = None * Sin("x")


def test_vector():
    f = Vector([Var("x"), Var("y")])

    assert f(x=2, y=3) == approx([2, 3])
    assert f.derivative("x", x=-2, y=5) == approx([1, 0])


def test_function_of_vector():
    f = Log(Vector([Var("x"), Var("x") ** 2]))

    assert f(x=2) == approx([np.log(2), np.log(4)])
    assert f.derivative("x", x=5) == approx([1 / 5, 2 / 5])


def test_tan():
    f = Tan("x")

    assert f(x=2) == approx(np.tan(2))
    assert f.derivative("x", x=2) == approx(1 / np.cos(2) ** 2)


def test_sec():
    f = Sec("x")

    assert f(x=2) == approx(1 / np.cos(2))
    assert f.derivative("x", x=2) == approx(np.tan(2) / np.cos(2))


def test_cosec():
    f = Csc("x")

    assert f(x=-1) == approx(1 / np.sin(-1))
    assert f.derivative("x", x=-1) == approx(-1 / np.tan(-1) / np.sin(-1))


def test_cot():
    f = Cot("x")

    assert f(x=5) == approx(1 / np.tan(5))
    assert f.derivative("x", x=5) == approx(-1 / np.sin(5) ** 2)


def test_missing_arg():
    f = Var("x") ** 2

    with raises(ValueError):
        f()

    with raises(ValueError):
        f.derivative("x")


def test_eq():
    f = Var("x") == Var("x")

    assert f(x=1)
    assert f(x=45)

    f = Var("x") == Var("y")

    assert not f(x=1, y=2)
    assert f(x=1, y=1)


def test_lt():
    f = Var("x") < Var("y")

    assert f(x=1, y=2)
    assert not f(x=2, y=1)
    assert not f(x=1, y=1)


def test_gt():
    f = Var("x") > Var("y")

    assert f(x=24, y=2)
    assert not f(x=2, y=24)
    assert not f(x=2, y=2)


def test_lte():
    f = Var("x") <= Var("y")

    assert f(x=-24, y=45)
    assert not f(x=36, y=-2)

    assert f(x=2, y=2)


def test_gte():
    f = Var("x") >= Var("y")

    assert f(x=24, y=2)
    assert not f(x=-2, y=45)
    assert f(x=4, y=4)


def test_ne():
    f = Var("x") != Var("y")

    assert f(x=2, y=3)
    assert f(x=3, y=2)
    assert not f(x=2, y=2)


def test_exp():
    f = Exp("x") + Exp(0)
    assert f(x=2) == approx(np.exp(2) + 1)
    assert f.derivative("x", x=2) == approx(np.exp(2))


def test_sqrt():
    f = Sqrt("x")
    assert f(x=2) == approx(2 ** 0.5)
    assert f.derivative("x", x=2) == approx(0.5 * (2 ** (-0.5)))


def test_other_trig_funcs():
    f = Sinh("x") + Sinh(2)
    assert f(x=2) == approx(2 * np.sinh(2))
    assert f.derivative("x", x=2) == approx(np.cosh(2))

    f = Cosh("x") + Cosh(2)
    assert f(x=2) == approx(2 * np.cosh(2))
    assert f.derivative("x", x=2) == approx(np.sinh(2))

    f = Tanh("x") + Tanh(2)
    assert f(x=2) == approx(2 * np.tanh(2))
    assert f.derivative("x", x=2) == approx(
        (np.cosh(2) ** 2 - np.sinh(2) ** 2) / np.cosh(2) ** 2
    )

    f = Sech("x") + Sech(2)
    assert f(x=2) == approx(2 * 1 / np.cosh(2))
    assert f.derivative("x", x=2) == approx(
        (-1 / np.cosh(2)) * (np.sinh(2) / np.cosh(2))
    )

    f = Csch("x") + Csch(2)
    assert f(x=2) == approx(2 * 1 / np.sinh(2))
    assert f.derivative("x", x=2) == approx(
        (-1 / np.sinh(2)) * (np.cosh(2) / np.sinh(2))
    )

    f = Coth("x") + Coth(2)
    assert f(x=2) == approx(2 * (np.cosh(2) / np.sinh(2)))
    assert f.derivative("x", x=2) == approx(-1 / (np.sinh(2) ** 2))


def test_neg():
    f = -Var("x")

    assert f(x=2) == approx(-2)
    assert f.derivative("x", x=1) == approx(-1)


def test_pos():
    f = +Var("x")

    assert f(x=3) == approx(3)
    assert f.derivative("x", x=3) == approx(1)


def test_ln():
    f = Ln("x") ** 2

    assert f(x=3) == approx(np.log(3) ** 2)
    assert f.derivative("x", x=4) == approx(2 * np.log(4) / 4)


def test_log2():
    f = Log2("x") ** 3

    assert f(x=3) == approx(np.log2(3) ** 3)
    assert f.derivative("x", x=4) == approx(1 / np.log(2) ** 3 / 4 * np.log(4) ** 2 * 3)


def test_log10():
    f = Log10("x") * 2

    assert f(x=3) == approx(np.log10(3) * 2)
    assert f.derivative("x", x=4) == approx(1 / np.log(10) * 2 / 4)


def test_arcsin():
    f = Arcsin(2 * Var("x"))

    assert f(x=0.4) == approx(np.arcsin(2 * 0.4))
    assert f.derivative("x", x=0.4) == approx(1 / np.sqrt(1 - 0.8 ** 2) * 2)


def test_arccos():
    f = Arccos(2 * Var("x"))

    assert f(x=0.1) == approx(np.arccos(2 * 0.1))
    assert f.derivative("x", x=0.1) == approx(-1 / np.sqrt(1 - 0.2 ** 2) * 2)


def test_arctan():
    f = Arctan(3 * Var("x"))

    assert f(x=3.5) == approx(np.arctan(3 * 3.5))
    assert f.derivative("x", x=3.5) == approx(1 / (1 + (3 * 3.5) ** 2) * 3)


def test_arcsec():
    f = Arcsec(Var("x") ** 2)

    assert f(x=3) == approx(np.arccos(1 / 9))
    assert f.derivative("x", x=3) == approx(1 / (abs(9) * np.sqrt(81 - 1)) * 2 * 3)


def test_arccsc():
    f = Arccsc(2 * Var("x"))

    assert f(x=-4) == approx(np.arcsin(1 / (2 * -4)))
    x = 2 * -4
    assert f.derivative("x", x=-4) == approx(-1 / (abs(x) * np.sqrt(x ** 2 - 1)) * 2)


def test_arccot():
    f = Arccot("x")

    assert f(x=0.5) == approx(np.arctan(1 / 0.5))
    assert f.derivative("x", x=0.5) == approx(-1 / (1 + 0.5 ** 2))

from autodiffpy import Reverse, sin, cos, tan, sec, csc, cot, exp, sinh, cosh, tanh, sech, csch, coth, ln, log, sqrt
from pytest import approx, raises
import numpy as np
import numpy as np
import math

# class Reverse:
#     def __init__(self, val):
#         self.value = val
#         self.children = []
#         self.gradient_value = None

#     def get_gradient(self):
#         if self.gradient_value is None:
#             self.gradient_value = sum(weight * var.get_gradient()
#                                       for weight, var in self.children)
#         return self.gradient_value

#     def __add__(self, other):
#         try:
#             z = Reverse(self.value + other.value)
#             self.children.append((1, z))
#             other.children.append((1, z))
#             return z
#         except AttributeError:
#             z = Reverse(self.value + other)
#             self.children.append((1, z))
#             return z

#     def __radd__(self, other):
#         return self.__add__(other)

#     def __sub__(self, other):
#         return self.__add__(-other)

#     def __rsub__(self, other):
#         return self.__neg__() + other

#     def __mul__(self, other):
#         try:
#             z = Reverse(self.value * other.value)
#             self.children.append((other.value, z))
#             other.children.append((self.value, z))
#             return z
#         except AttributeError:
#             z = Reverse(self.value * other)
#             self.children.append((other, z))
#             return z

#     def __rmul__(self, other):
#         return self.__mul__(other)

#     def __truediv__(self, other):
#         return self.__mul__(other**(-1))

#     def __rtruediv__(self, other):
#         return self.__pow__(-1) * other

#     def __pow__(self, other):
#         try:
#             z = Reverse(self.value ** other.value)
#             self.children.append((other.value * self.value**(other.value-1), z))
#             other.children.append((self.value**other.value * np.log(self.value), z))
#             return z
#         except AttributeError:
#             z = Reverse(self.value ** other)
#             self.children.append((other * self.value**(other-1), z))
#             return z

#     def __rpow__(self, other):
#         z = Reverse(other ** self.value)
#         self.children.append((other**self.value * np.log(other), z))
#         return z

#     def __neg__(self):
#         return self.__mul__(-1)

# def sin(x):
#     try:
#         z = Reverse(np.sin(x.value))
#         x.children.append((np.cos(x.value), z))
#         return z
#     except AttributeError:
#         return np.sin(x)

# def cos(x):
#     try:
#         z = Reverse(np.cos(x.value))
#         x.children.append((-1*np.sin(x.value), z))
#         return z
#     except AttributeError:
#         return np.cos(x)

# def tan(x):
#     return sin(x)/cos(x)

# def sec(x):
#     return 1 / cos(x)

# def csc(x):
#     return 1 / sin(x)

# def cot(x):
#     return 1 / tan(x)

# def exp(x):
#     try:
#         z = Reverse(np.exp(x.value))
#         x.children.append((np.exp(x.value), z))
#         return z
#     except AttributeError:
#         return np.exp(x)

# def sinh(x):
#     return (exp(x)-exp(-x))/2

# def cosh(x):
#     return (exp(x)+exp(-x))/2

# def tanh(x):
#     return sinh(x)/cosh(x)

# def sech(x):
#     return 1 / cosh(x)

# def csch(x):
#     return 1 / sinh(x)

# def coth(x):
#     return 1 / tanh(x)

# def ln(x):
#     return log(x)

# def log(x, base=np.exp(1)):
#     # default log is natural log
#     try:
#         z = Reverse(math.log(x.value, base))
#         x.children.append((1/(np.log(base)*x.value), z))
#         return z
#     except AttributeError:
#         return math.log(x, base)

# def sqrt(x):
#     return x**(1/2)






def test_single_var():
    x = Reverse(1)
    f = x
    f.gradient_value = 1.0
    assert f.value == approx(1)
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
    f = sin(x)
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(5))
    assert x.get_gradient() == approx(np.cos(5))

    # Simple Nested case
    x = Reverse(5)
    f = sin(2 * x)
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(2 * 5))
    assert x.get_gradient() == approx(2 * np.cos(2 * 5))


def test_cos():
    x = Reverse(5)
    y = Reverse(5)
    f = sin(x) * cos(y)
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(5) * np.cos(5))
    assert y.get_gradient() == approx((-np.sin(5)) * np.sin(5))

    # Nested case
    x = Reverse(4)
    y = Reverse(3)
    f = sin(cos(x*y))
    f.gradient_value = 1.0
    assert f.value == approx(np.sin(np.cos(3 * 4)))
    assert x.get_gradient() == approx(-3 * np.sin(12) * np.cos(np.cos(12)))



def test_log():
    x = Reverse(4)
    f = log(x) + log(5)
    f.gradient_value = 1.0
    assert f.value == approx(np.log(4) + np.log(5))
    assert x.get_gradient() == approx(1 / 4)

    # Harder case
    x = Reverse(3)
    y = Reverse(7)
    f = log(x ** y)
    f.gradient_value = 1.0
    assert f.value == approx(np.log(3 ** 7))
    assert x.get_gradient() == approx((7 * (3 ** 6)) * 1/(3**7))


# def test_sin_of_const():
#     f = Sin(2)

#     assert f(x=2) == approx(np.sin(2))
#     assert f.derivative("x") == approx(0)


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

# test_single_var()
# test_multiplication()
# test_multiplication_by_variable()
# test_division()
# test_power()
# test_addition()
# test_subtraction()
# test_weirdness()
# test_sin()
# test_cos()
# test_log()

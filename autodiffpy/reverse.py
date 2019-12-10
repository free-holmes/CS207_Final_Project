import numpy as np
import math


# class for automatic differentiation reverse mode
class Reverse:
    def __init__(self, val):
        self.value = val
        self.children = []
        self.gradient_value = None

    def reset_gradient(self):
        self.gradient_value = None
        for i in self.children:
            i[1].reset_gradient()

    def get_gradient(self):
        if self.gradient_value is None:
            self.gradient_value = sum(
                weight * child.get_gradient() for weight, child in self.children
            )
        return self.gradient_value

    def __str__(self):
        return f'value = {self.value}, gradient_value = {self.gradient_value}'

    def __add__(self, other):
        try:
            z = Reverse(self.value + other.value)
            self.children.append((1, z))
            other.children.append((1, z))
            return z
        except AttributeError:
            z = Reverse(self.value + other)
            self.children.append((1, z))
            return z

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__neg__() + other

    def __mul__(self, other):
        try:
            z = Reverse(self.value * other.value)
            self.children.append((other.value, z))
            other.children.append((self.value, z))
            return z
        except AttributeError:
            z = Reverse(self.value * other)
            self.children.append((other, z))
            return z

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(other ** (-1))

    def __rtruediv__(self, other):
        return self.__pow__(-1) * other

    def __pow__(self, other):
        try:
            z = Reverse(self.value ** other.value)
            self.children.append((other.value * self.value ** (other.value - 1), z))
            other.children.append((self.value ** other.value * np.log(self.value), z))
            return z
        except AttributeError:
            z = Reverse(self.value ** other)
            self.children.append((other * self.value ** (other - 1), z))
            return z

    def __rpow__(self, other):
        z = Reverse(other ** self.value)
        self.children.append((other ** self.value * np.log(other), z))
        return z

    def __neg__(self):
        return self.__mul__(-1)

    def __eq__(self, other):
        try:
            return self.value == other.value and self.gradient_value == other.gradient_value
        except AttributeError:
            return self.value == other

    def __ne__(self, other):
        try:
            return self.value != other.value or self.gradient_value != other.gradient_value
        except AttributeError:
            return self.value != other

def sin(x):
    try:
        z = Reverse(sin(x.value))
        x.children.append((cos(x.value), z))
        return z
    except AttributeError:
        return np.sin(x)


def cos(x):
    try:
        z = Reverse(cos(x.value))
        x.children.append((-1 * sin(x.value), z))
        return z
    except AttributeError:
        return np.cos(x)


def tan(x):
    return sin(x) / cos(x)


def sec(x):
    return 1 / cos(x)


def csc(x):
    return 1 / sin(x)


def cot(x):
    return 1 / tan(x)

def arcsin(x):
    try:
        z = Reverse(arcsin(x.value))
        x.children.append((1/(sqrt(1-x.value**2)), z))
        return z
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    try:
        z = Reverse(arccos(x.value))
        x.children.append((-1/(sqrt(1-x.value**2)), z))
        return z
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    try:
        z = Reverse(arctan(x.value))
        x.children.append((1/(1+x.value**2), z))
        return z
    except AttributeError:
        return np.arctan(x)

def exp(x):
    try:
        z = Reverse(exp(x.value))
        x.children.append((exp(x.value), z))
        return z
    except AttributeError:
        return np.exp(x)


def sinh(x):
    try:
        z = Reverse(sinh(x.value))
        x.children.append((cosh(x.value), z))
        return z
    except AttributeError:
        return (exp(x) - exp(-x)) / 2


def cosh(x):
    try:
        z = Reverse(cosh(x.value))
        x.children.append((sinh(x.value), z))
        return z
    except AttributeError:
        return (exp(x) + exp(-x)) / 2


def tanh(x):
    return sinh(x) / cosh(x)


def sech(x):
    return 1 / cosh(x)


def csch(x):
    return 1 / sinh(x)


def coth(x):
    return 1 / tanh(x)


def ln(x):
    return log(x)


def log2(x):
    return log(x, 2)


def log10(x):
    return log(x, 10)


def log(x, base=np.exp(1)):
    # default log is natural log
    try:
        z = Reverse(log(x.value, base))
        x.children.append((1 / (log(base) * x.value), z))
        return z
    except AttributeError:
        return math.log(x, base)


def sqrt(x):
    return x**(1/2)


class rVector():
    def __init__(self, functions: list, variables: list):
        # INPUT: functions in a list
        # INPUT : variables in a list
        # example:
        # x = Reverse(1)
        # y = Reverse(2)
        # functions = [x*2*y+y**3, 2*x**2*y, 3*y]
        # variables = [x, y]
        # vector = rVector(functions, variables)
        # >>> [12, 4, 6]
        # >>> [4.0, 8.0, 0]
        # >>> [14.0, 2.0, 3.0]

        self.functions = functions
        self.variables = variables
        self.values = []
        self.gradients = {}
        self.find_gradients()

    def find_gradients(self):
        for i, j in enumerate(self.functions):
            for f in self.functions:
                f.gradient_value = 0
            j.gradient_value = 1.0

            self.values.append(j.value)

            for v in self.variables:
                if id(v) not in self.gradients.keys():
                    self.gradients[id(v)] = [v.get_gradient()]
                else:
                    self.gradients[id(v)].append(v.get_gradient())
            for v in self.variables:
                v.reset_gradient()

    def get_gradients(self, variable):
        return self.gradients[id(variable)]

    def get_values(self):
        return self.values

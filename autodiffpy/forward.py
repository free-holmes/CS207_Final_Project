import numpy as np


class AutoDiff:
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        """Not implemented"""

    def derivative(self, *variables, **kwargs):
        """Not implemented"""

    @classmethod
    def coerce(self, thing):
        if isinstance(thing, AutoDiff):
            return thing

        if isinstance(thing, (float, int)):
            return Constant(thing)

        if isinstance(thing, str):
            return Var(thing)

        raise ValueError

    def __add__(self, other):
        return Add(self, AutoDiff.coerce(other))

    def __radd__(self, other):
        return Add(AutoDiff.coerce(other), self)

    def __sub__(self, other):
        return Sub(self, AutoDiff.coerce(other))

    def __rsub__(self, other):
        return Sub(AutoDiff.coerce(other), self)

    def __mul__(self, other):
        return Mul(self, AutoDiff.coerce(other))

    def __rmul__(self, other):
        return Mul(AutoDiff.coerce(other), self)

    def __truediv__(self, other):
        return Div(self, AutoDiff.coerce(other))

    def __rtruediv__(self, other):
        return Div(AutoDiff.coerce(other), self)

    def __pow__(self, other):
        return Pow(self, AutoDiff.coerce(other))

    def __rpow__(self, other):
        return Pow(AutoDiff.coerce(other), self)

    def __eq__(self, other):
        return Eq(self, AutoDiff.coerce(other))

    def __lt__(self, other):
        return Lt(self, AutoDiff.coerce(other))

    def __gt__(self, other):
        return Gt(self, AutoDiff.coerce(other))

    def __le__(self, other):
        return Le(self, AutoDiff.coerce(other))

    def __ge__(self, other):
        return Ge(self, AutoDiff.coerce(other))

    def __ne__(self, other):
        return Ne(self, AutoDiff.coerce(other))

    def __neg__(self):
        return Sub(AutoDiff.coerce(0), self)

    def __pos__(self):
        return Add(AutoDiff.coerce(0), self)


class Eq(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) == self.right(**kwargs)


class Lt(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) < self.right(**kwargs)


class Gt(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) > self.right(**kwargs)


class Le(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) <= self.right(**kwargs)


class Ge(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) >= self.right(**kwargs)


class Ne(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) != self.right(**kwargs)


class Add(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) + self.right(**kwargs)

    def derivative(self, *args, **kwargs):
        return self.left.derivative(*args, **kwargs) + self.right.derivative(
            *args, **kwargs
        )


class Sub(AutoDiff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) - self.right(**kwargs)

    def derivative(self, *args, **kwargs):
        return self.left.derivative(*args, **kwargs) - self.right.derivative(
            *args, **kwargs
        )


class Mul(AutoDiff):
    def __init__(self, left, right):
        super().__init__()

        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) * self.right(**kwargs)

    def derivative(self, *args, **kwargs):
        return self.left(**kwargs) * self.right.derivative(
            *args, **kwargs
        ) + self.left.derivative(*args, **kwargs) * self.right(**kwargs)


class Div(AutoDiff):
    def __init__(self, top, bottom):
        super().__init__()

        self.top = top
        self.bottom = bottom

    def __call__(self, **kwargs):
        return self.top(**kwargs) / self.bottom(**kwargs)

    def derivative(self, *args, **kwargs):
        g = self.bottom(**kwargs)
        return (
            self.top.derivative(*args, **kwargs) * g
            - self.top(**kwargs) * self.bottom.derivative(*args, **kwargs)
        ) / g ** 2


class Constant(AutoDiff):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, **kwargs):
        return self.value

    def derivative(self, *args, **kwargs):
        return 0


class Var(AutoDiff):
    def __init__(self, variable_name):
        super().__init__()
        self.variable_name = variable_name

    def __call__(self, **kwargs):
        if self.variable_name not in kwargs:
            raise ValueError(f"Variable {self.variable_name} not provided!")
        return kwargs[self.variable_name]

    def derivative(self, *args, **kwargs):
        if self.variable_name in args:
            return 1

        return 0


class Pow(AutoDiff):
    def __init__(self, base, exp):
        super().__init__()
        self.base = base
        self.exp = exp

    def __call__(self, **kwargs):
        return self.base(**kwargs) ** self.exp(**kwargs)

    def derivative(self, *args, **kwargs):
        a, b = self.base(**kwargs), self.base.derivative(*args, **kwargs)
        c, d = self.exp(**kwargs), self.exp.derivative(*args, **kwargs)
        return a ** c * (d * np.log(a) + b * c / a)


class Sin(AutoDiff):
    def __init__(self, val):
        super().__init__()
        self.val = AutoDiff.coerce(val)

    def __call__(self, **kwargs):
        return np.sin(self.val(**kwargs))

    def derivative(self, *args, **kwargs):
        return self.val.derivative(*args, **kwargs) * np.cos(self.val(**kwargs))


class Cos(AutoDiff):
    def __init__(self, val):
        super().__init__()
        self.val = AutoDiff.coerce(val)

    def __call__(self, **kwargs):
        return np.cos(self.val(**kwargs))

    def derivative(self, *args, **kwargs):
        return -self.val.derivative(*args, **kwargs) * np.sin(self.val(**kwargs))


class _Log(AutoDiff):
    # default log is natural log
    def __init__(self, val):
        self.val = AutoDiff.coerce(val)

    def __call__(self, **kwargs):
        return np.log(self.val(**kwargs))

    def derivative(self, *args, **kwargs):
        return self.val.derivative(*args, **kwargs) / self.val(**kwargs)


def Log(x, base=np.e):
    return _Log(x) / _Log(base)


class Vector(AutoDiff):
    def __init__(self, values):
        super().__init__()
        self.values = [AutoDiff.coerce(val) for val in values]

    def __call__(self, **kwargs):
        return np.array([val(**kwargs) for val in self.values])

    def derivative(self, *args, **kwargs):
        return np.array([val.derivative(*args, **kwargs) for val in self.values])


def Tan(x):
    return Sin(x) / Cos(x)


def Sec(x):
    return 1 / Cos(x)


def Csc(x):
    return 1 / Sin(x)


def Cot(x):
    return 1 / Tan(x)


def Exp(x):
    return AutoDiff.coerce(np.e) ** AutoDiff.coerce(x)


def Sinh(x):
    x = AutoDiff.coerce(x)
    return (Exp(x) - Exp(-x)) / 2


def Cosh(x):
    x = AutoDiff.coerce(x)
    return (Exp(x) + Exp(-x)) / 2


def Tanh(x):
    return Sinh(x) / Cosh(x)


def Sech(x):
    return 1 / Cosh(x)


def Csch(x):
    return 1 / Sinh(x)


def Coth(x):
    return 1 / Tanh(x)


def Ln(x):
    return Log(x)


def Log2(x):
    return Log(x, 2)


def Log10(x):
    return Log(x, 10)


def Sqrt(x):
    return AutoDiff.coerce(x) ** (1 / 2)

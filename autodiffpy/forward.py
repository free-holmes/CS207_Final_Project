import numpy as np


class AutoDiff:
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        """Not implemented"""

    def derivative(self, *variables, **kwargs):
        """Not implemented"""

    @classmethod
    def _coerce(self, thing):
        if isinstance(thing, AutoDiff):
            return thing

        if isinstance(thing, (float, int)):
            return Constant(thing)

        if isinstance(thing, str):
            return Var(thing)

        raise ValueError

    def __add__(self, other):
        return Add(self, AutoDiff._coerce(other))

    def __radd__(self, other):
        return Add(AutoDiff._coerce(other), self)

    def __sub__(self, other):
        return Sub(self, AutoDiff._coerce(other))

    def __rsub__(self, other):
        return Sub(AutoDiff._coerce(other), self)

    def __mul__(self, other):
        return Mul(self, AutoDiff._coerce(other))

    def __rmul__(self, other):
        return Mul(AutoDiff._coerce(other), self)

    def __truediv__(self, other):
        return Div(self, AutoDiff._coerce(other))

    def __rtruediv__(self, other):
        return Div(AutoDiff._coerce(other), self)

    def __pow__(self, other):
        return Pow(self, AutoDiff._coerce(other))

    def __rpow__(self, other):
        return Pow(AutoDiff._coerce(other), self)


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
        self.val = AutoDiff._coerce(val)

    def __call__(self, **kwargs):
        return np.sin(self.val(**kwargs))

    def derivative(self, *args, **kwargs):
        return self.val.derivative(*args, **kwargs) * np.cos(self.val(**kwargs))


class Cos(AutoDiff):
    def __init__(self, val):
        super().__init__()
        self.val = AutoDiff._coerce(val)

    def __call__(self, **kwargs):
        return np.cos(self.val(**kwargs))

    def derivative(self, *args, **kwargs):
        return -self.val.derivative(*args, **kwargs) * np.sin(self.val(**kwargs))


class Log(AutoDiff):
    def __init__(self, val):
        super().__init__()
        self.val = AutoDiff._coerce(val)

    def __call__(self, **kwargs):
        return np.log(self.val(**kwargs))

    def derivative(self, *args, **kwargs):
        return self.val.derivative(*args, **kwargs) / self.val(**kwargs)


class Vector(AutoDiff):
    def __init__(self, values):
        super().__init__()
        self.values = [AutoDiff._coerce(val) for val in values]

    def __call__(self, **kwargs):
        return np.array([val(**kwargs) for val in self.values])

    def derivative(self, *args, **kwargs):
        return np.array([val.derivative(*args, **kwargs) for val in self.values])

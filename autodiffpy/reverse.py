import numpy as np
import math


# class for automatic differentiation reverse mode
class Reverse:
    def __init__(self, val):
        self.value = val
        self.children = []
        self.gradient_value = None

    def get_gradient(self):
        if self.gradient_value is None:
            self.gradient_value = sum(weight * child.get_gradient() for weight, child in self.children)
        return self.gradient_value

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
        try:
            z = Reverse(self.value - other.value)
            self.children.append((1, z))
            other.children.append((1, z))
            return z
        except AttributeError:
            z = Reverse(self.value - other)
            self.children.append((1, z))
            return z

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
        return self.__mul__(other**(-1))

    def __rtruediv__(self, other):
        return self.__pow__(-1) * other

    def __pow__(self, other):
        try:
            z = Reverse(self.value ** other.value)
            self.children.append((other.value * self.value**(other.value-1), z))
            other.children.append((self.value**other.value * np.log(self.value), z))
            return z
        except AttributeError:
            z = Reverse(self.value ** other)
            self.children.append((other * self.value**(other-1), z))
            return z

    def __rpow__(self, other):
        z = Reverse(other ** self.value)
        self.children.append((other**self.value * np.log(other), z))
        return z

    def __neg__(self):
        return self.__mul__(-1)


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
        x.children.append((-1*sin(x.value), z))
        return z
    except AttributeError:
        return np.cos(x)

def tan(x):
    return sin(x)/cos(x)

def sec(x):
    return 1/cos(x)

def csc(x):
    return 1/sin(x)

def cot(x):
    return 1/tan(x)

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
        return (exp(x)-exp(-x))/2

def cosh(x):
    try:
        z = Reverse(cosh(x.value))
        x.children.append((sinh(x.value), z))
        return z
    except AttributeError:
        return (exp(x)+exp(-x))/2

def tanh(x):
    return sinh(x)/cosh(x)

def sech(x):
    return 1/cosh(x)

def csch(x):
    return 1/sinh(x)

def coth(x):
    return 1/tanh(x)

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
        x.children.append((1/(log(base)*x.value), z))
        return z
    except AttributeError:
        return math.log(x, base)

def sqrt(x):
    return x**(1/2)


def create_function():
    # declare variables that will be used in expression
    while True:
        try:
            var_list = input("Enter the variable names separated by a space:")
            var_list = list(var_list.split(' '))
            for i in var_list:
                assert i[0].isalpha()
            break
        except AssertionError:
            print(f'INVALID INPUT: {var_list}\nAll variable names must be alphanumeric.')

    # set value for declared variables
    counter = 0
    while counter < len(var_list):
        var_name = var_list[counter]
        val = input(f'Enter the value of {var_name}:')
        try:
            vars()[var_name] = Reverse(float(val))
            counter += 1
        except ValueError:
            print(f'INVALID INPUT: {val}\nValue for {var_name} must be a real number.')

    # input the expression
    expr = input("Enter the function:")

    # evaluate the expression
    eval_expr = eval(expr)
    try:
        eval_expr.gradient_value = 1
        print(f'expression = {eval_expr.value}')
        for i in var_list:
            print(f'{i} gradient = {vars()[i].get_gradient()}')
    except AttributeError:
        print(f'There are no variables in the expression you entered: {expr}')
        print(f'expression = {eval_expr}')


if __name__ == "__main__":
    create_function()

import numpy as np
import math


# class for automatic differentiation reverse mode
class Reverse:
    def __init__(self, val):
        self.value = val
        self.children = []
        self.gradient_value = None

    def reset(self):
        self.children = []
        self.gradient_value = None

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
    def __init__(self, functions: list, variables: dict):
        # INPUT: functions is a list
        # functions must be a string
        # INPUT : variables is a dictionary
        # keys are the variable name (string)
        # key values are the values of the variables
        # example:
        # func = ['x*2*y+y**3', '2*x**2*y', '3*y']
        # vars_dict = {'x': 1, 'y': 2}
        # vector = rVector(func, vars_dict)

        self.functions = functions
        self.variables = variables
        self.values = None
        self.gradients = None
        self.find_gradients()

    def __str__(self):
        # returns a string of the class in a user friendly form
        # example:
        # func = ['x*2*y+y**3', '2*x**2*y', '3*y']
        # vars_dict = {'x': 1, 'y': 2}
        # vector = rVector(func, vars_dict)
        # print(vector)
        # >>> x=1
        # >>> y=2
        # >>> x*2*y+y**3=12.0  Df(x)=4.0  Df(y)=14.0
        # >>> 2*x**2*y=4.0  Df(x)=8.0  Df(y)=2.0
        # >>> 3*y=6.0  Df(x)=0  Df(y)=3

        printout = ''.join([f"{k}={v}\n" for k,v in self.variables.items()])
        for i, j in zip(self.gradients.items(), self.values.items()):
            printout += f'{j[0]}={j[1]}  ' + ''.join([f"Df({k})={v}  " for k,v in i[1].items()]) + '\n'
        return printout.rstrip()

    def find_gradients(self, functions: list = None, variables: dict = None):
        # rules for inputs are the same as __init__

        # overwrites functions and variables from initialization if provided in function call
        if variables is not None:
            self.variables = variables
        if functions is not None:
            self.functions = functions

        # create Reverse objects from variables
        for k, v in self.variables.items():
            try:
                assert k[0].isalpha()
                vars()[k] = Reverse(float(v))
            except (AttributeError, AssertionError) as e:
                raise AttributeError(f'INVALID VARIABLE: {k} in {self.variables.keys()}, All variable names must be alphanumeric.')
            except ValueError as e:
                raise ValueError(f'INVALID VALUE: {k} = {v}, assigned value must be a real number.')

        # evaluate function and gradients
        self.values = {}
        self.gradients = {}
        for f in self.functions:
            try:
                eval_func = eval(f)
                eval_func.gradient_value = 1
                self.values[f] = eval_func.value
                var_results = {}
                for i in self.variables.keys():
                    var_results[i] = vars()[i].get_gradient()
                    vars()[i].reset()
                if len(self.variables.items()) > 0:
                    self.gradients[f] = var_results
            except AttributeError as e:
                self.values[f] = eval(f)
            except (SyntaxError, NameError) as e:
                raise Exception(f'INVALID FUNCTION: {f}, {e}')
            except TypeError as e:
                raise TypeError(f'INVALID FUNCTION: Function {f} must be input as a string, {e}')

        # returns dictionary of functions with their gradients
        return self.gradients


    def get_gradients(self, func_num: int = None, var_name: str = None):
        # gets the gradient values for one function and/or key
        # example:
        # func = ['x*2*y+y**3', '2*x**2*y', '3*y']
        # vars_dict = {'x': 1, 'y': 2}
        # vector = rVector(func, vars_dict)
        # vector.get_gradients(func_num=0, var_name='x')
        # >>> 4.0
        # vector.get_gradients(func_num=0)
        # >>> {'x': 4.0, 'y': 14.0}
        # vector.get_gradients(var_name='x')
        # >>> [4.0, 8.0, 0]
        # vector.get_gradients()
        # >>> [{'x': 4.0, 'y': 14.0}, {'x': 8.0, 'y': 2.0}, {'x': 0, 'y': 3}]

        # returns gradient value of all functions if no func_num or var_name provided
        if var_name is None and func_num is None:
            return list(self.gradients.values())
        # returns gradient value of var_name for all functions if no func_num provided
        elif func_num is None:
            return [i[var_name] for i in list(self.gradients.values())]
        # returns gradient value for all variables of function if no var_name provided
        elif var_name is None:
            return list(self.gradients.values())[func_num]
        # returns the gradient value for the provided func_num and var_name
        else:
            return list(self.gradients.values())[func_num][var_name]

import numpy as np
import math


""" Class for automatic differentiation using reverse mode. Represents a node in the computation graph.

Attributes:
    value {Float} -- value of the node in the computation graph
    children {[Reverse]} -- Array of Reverse nodes whose value is dependent upon this node
    gradient_value -- gradient of the node in the computation graph
"""


class Reverse:
    def __init__(self, val):
        """Initializes a Reverse object with value, empty children array, and a blank gradient. Children are nodes whose values are dependent upon this node.

        Arguments:
            val {Float} -- stores function value at this point of reverse graph

        Returns:
            None
        """
        self.value = val
        self.children = []
        self.gradient_value = None

    def reset_gradient(self):
        self.gradient_value = None
        for i in self.children:
            i[1].reset_gradient()

    def get_gradient(self):
        """Returns gradient value. Calculates gradient value if undefined.

            Returns:
                Float -- gradient value
        """
        if self.gradient_value is None:
            self.gradient_value = sum(
                weight * child.get_gradient() for weight, child in self.children
            )
        return self.gradient_value

    def __str__(self):
        """Sets string output for Revese object

        Returns:
            String -- displays value and gradient_value
        """
        return f"value = {self.value}, gradient_value = {self.gradient_value}"

    def __add__(self, other):
        """Adds Reverse object to another value and appends result to children

        Arguments:
            other {Reverse, Float} -- determines type then adds to self and appends to children

        Returns:
            Reverse -- sum of self and other object
        """
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
        """Calculates reverse add with self and other

        Arguments:
            other {Reverse, Float} -- calls as input to Reverse.__add__

        Returns:
            Reverse -- sum of self and other object
        """
        return self.__add__(other)

    def __sub__(self, other):
        """Returns difference between self and other object by negating other then adding to self

        Arguments:
            other {Reverse, Float} -- negates then calls as input to Reverse.__add__

        Returns:
            Reverse -- difference of self and other
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """Returns difference between else and other object by negating self then adding to other

        Arguments:
            other {Reverse, Float} -- adds to self after negating self

        Returns:
            Reverse -- difference of self and other
        """
        return self.__neg__() + other

    def __mul__(self, other):
        """Calculates the product of self and other and appends result to children

        Arguments:
            other {Reverse, Float} -- determines type then multiplies with self and appends to children

        Returns:
            Reverse -- product of self and other
        """
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
        """Calculates product of self and other

        Arguments:
            other {Reverse, Float} -- calls as input to Reverse.__mul__

        Returns:
            Reverse -- product of self and other
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divides self by other using __mul__ and inverse of other

        Arguments:
            other {Reverse, Float} -- divisor

        Returns:
            Reverse -- self divided by other
        """
        return self.__mul__(other ** (-1))

    def __rtruediv__(self, other):
        """Divides other by self using __mul__ and inverse of self

        Arguments:
            other {Reverse, Float} -- dividend

        Returns:
            Reverse -- other divided by self
        """
        return self.__pow__(-1) * other

    def __pow__(self, other):
        """Raises self to the other power and appends result to childen

        Arguments:
            other {Reverse, Float} -- power to which user will raise self

        Returns:
            Reverse -- self raised to the other
        """
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
        """Calculates other raised to self and appends to children

        Arguments:
            other {Reverse, Float} -- value being raised to the self power

        Returns:
            Reverse -- other raised to self
        """
        z = Reverse(other ** self.value)
        self.children.append((other ** self.value * np.log(other), z))
        return z

    def __neg__(self):
        """Negates self

        Returns:
            Reverse -- self negated
        """
        return self.__mul__(-1)

    def __eq__(self, other):
        """Calculates whether self is equal to other

        Arguments:
            other {Reverse, Float} -- value being compared to self

        Returns:
            Bool -- true if self is equal to other
        """
        try:
            return (
                self.value == other.value
                and self.gradient_value == other.gradient_value
            )
        except AttributeError:
            return self.value == other

    def __ne__(self, other):
        """Calculates whether self is not equal to other

        Arguments:
            other {Reverse, Float} -- value being compared to self

        Returns:
            Bool -- true if self is not equal to other
        """
        try:
            return (
                self.value != other.value or self.gradient_value != other.gradient_value
            )
        except AttributeError:
            return self.value != other


def sin(x):
    """Returns sin of x. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to sin function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    try:
        z = Reverse(sin(x.value))
        x.children.append((cos(x.value), z))
        return z
    except AttributeError:
        return np.sin(x)


def cos(x):
    """Returns cos of x. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to cos function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    try:
        z = Reverse(cos(x.value))
        x.children.append((-1 * sin(x.value), z))
        return z
    except AttributeError:
        return np.cos(x)


def tan(x):
    """Returns tan of x using sin and cos Reverse methods. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to tan function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return sin(x) / cos(x)


def sec(x):
    """Returns sec of x using inverse of cos Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to sec function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return 1 / cos(x)


def csc(x):
    """Returns csc of x using inverse of sin Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to csc function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return 1 / sin(x)


def cot(x):
    """Returns cot of x using inverse of tan Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to cot function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return 1 / tan(x)


def arcsin(x):
    try:
        z = Reverse(arcsin(x.value))
        x.children.append((1 / (sqrt(1 - x.value ** 2)), z))
        return z
    except AttributeError:
        return np.arcsin(x)


def arccos(x):
    try:
        z = Reverse(arccos(x.value))
        x.children.append((-1 / (sqrt(1 - x.value ** 2)), z))
        return z
    except AttributeError:
        return np.arccos(x)


def arctan(x):
    try:
        z = Reverse(arctan(x.value))
        x.children.append((1 / (1 + x.value ** 2), z))
        return z
    except AttributeError:
        return np.arctan(x)


def exp(x):
    """Returns e^x. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- value to which e is raised

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    try:
        z = Reverse(exp(x.value))
        x.children.append((exp(x.value), z))
        return z
    except AttributeError:
        return np.exp(x)


def sinh(x):
    """Calculates hyperbolic sin of x. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to sinh function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    try:
        z = Reverse(sinh(x.value))
        x.children.append((cosh(x.value), z))
        return z
    except AttributeError:
        return (exp(x) - exp(-x)) / 2


def cosh(x):
    """Calculates hyperbolic cos of x. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to cosh function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    try:
        z = Reverse(cosh(x.value))
        x.children.append((sinh(x.value), z))
        return z
    except AttributeError:
        return (exp(x) + exp(-x)) / 2


def tanh(x):
    """Calculates hyperbolic tan of x using sinh and cosh Reverse methods. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to tanh function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return sinh(x) / cosh(x)


def sech(x):
    """Calculates hyperbolic sec of x using inverse of cosh Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to sech function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return 1 / cosh(x)


def csch(x):
    """Calculates hyperbolic csc of x using inverse of sinh Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to csch function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return 1 / sinh(x)


def coth(x):
    """Calculates hyperbolic cot of x using inverse of tanh Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- input to coth function

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return 1 / tanh(x)


def log(x, base=np.exp(1)):
    """Calculates base log of x. Defaults to natural log. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- Value to calculate log.
        base (default: e) {Float} -- log base

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    try:
        z = Reverse(log(x.value, base))
        x.children.append((1 / (log(base) * x.value), z))
        return z
    except AttributeError:
        return math.log(x, base)


def ln(x):
    """Calculates natural log of x using log Reverse method. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- Value to calculate natural log.

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return log(x)


def log2(x):
    """Calculates log2 of x using log Reverse method with 2 base. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- Value to calculate log2.

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return log(x, 2)


def log10(x):
    """Calculates log10 of x using log Reverse method with 10 base. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- Value to calculate log10.

    Returns:
        {Reverse, Float} -- Only returns Reverse if x is a Reverse object. Else float.
    """
    return log(x, 10)


def sqrt(x):
    """ Calculates square root of x. Appends result to x.children if x is a Reverse object.

    Arguments:
        x {Reverse, Float} -- Value to calculate square root for.

    Returns:
        Reverse -- Input raised to the 0.5
    """
    return x ** (1 / 2)


class rVector:
    def __init__(self, functions: list):
        # INPUT: functions in a list
        # example:
        # x = Reverse(1)
        # y = Reverse(2)
        # functions = [x*2*y+y**3, 2*x**2*y, 3*y]
        # vector = rVector(functions)
        # print(vector.values)
        # print(vector.get_gradients(x))
        # print(vector.get_gradients(y))
        # >>> [12, 4, 6]
        # >>> [4.0, 8.0, 0]
        # >>> [14.0, 2.0, 3.0]

        self.functions = functions
        self.values = []
        self._find_values()

    def _find_values(self):
        for i in self.functions:
            self.values.append(i.value)

    def get_gradients(self, variable):
        gradients = []
        for i in self.functions:
            for f in self.functions:
                f.gradient_value = 0
            i.gradient_value = 1.0

            gradients.append(variable.get_gradient())
            variable.reset_gradient()

        return gradients

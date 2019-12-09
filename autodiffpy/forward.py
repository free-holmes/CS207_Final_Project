"""Implements the forward mode of automatic differentiation

This module implements the forward mode of automatic differentiation. It does this by
overloading the many math dunder methods that are provided in Python such as __add__,
__sub__, __mul__, etc. In addition to these primitive functions, this module also
defines the elementary functions such as the trigonometric functions, hyperbolic
functions, logarithms, exponentials, and logistic function.

To create a new function of a variable, you can instantiate a variable with an
initial value using the constructor Forward. For example, Forward('x', 1) creates a
variable named 'x' with initial value 1. If we use this value in mathematical
operations, then we will create more Forward objects that we can access.

To get the results of computation, we can access the actual value by doing .value on
the Forward object at hand. To get the gradient with respect to a certain variable,
we can call .get_gradient(variable_name) on the Forward object.
"""
import numpy as np

np.seterr(all="ignore")


def _coerce(arg):
    """Private function which does the actual coercion of a type into a
    Forward object.
    """
    if isinstance(arg, Forward):
        return arg

    # we support complex numbers too!
    if isinstance(arg, (float, int, complex)):
        return Forward(arg)

    # otherwise raise ValueError cause we don't support
    raise ValueError(type(arg))


def coerce(fun):
    """Decorates a function and coerces each of the inputs of the function into a
    Forward object.

    Many of our functions would like to operate on Forward objects instead of raw
    values. For example, __add__ might get an integer in the case of Forward('x', 5)
    + 2, but we want the 2 to be wrapped in a Forward object. This decorator
    automates that process and cleans up the code.
    """

    def ret_f(*args):
        new_args = [_coerce(arg) for arg in args]

        return fun(*new_args)

    return ret_f


class Forward:
    """The primary class of the forward mode of automatic differentiation.

    The Forward class can be used to instantiate the forward mode of automatic
    differentiation. By overloading the many dunder methods of Python, this class
    enables the user to seamlessly define the forward computation of a function while
    simultaneously deriving the gradient.

    The result of the computation can be accessed via the .value attribute of the
    object. The gradient can be accessed by the .get_gradient(variable_name) method
    which returns the gradient with respect to a particular variable.

    The object can be instantiated with one or two arguments. If one argument is
    provided, it must be a numeric type. This represents a constant within automatic
    differentiation. If two arguments are provided, the first must be a string, which
    represents the variable name, and the second must be a numeric type, which
    represents the value of that variable.
    """

    def __init__(self, *args):
        if len(args) == 1:
            value = args[0]

            if not isinstance(value, (int, float, complex)):
                raise ValueError

            self.derivatives = {}
            self.value = value

        elif len(args) == 2:
            var_name, value = args

            if not isinstance(var_name, str):
                raise ValueError

            if not isinstance(value, (int, float, complex)):
                raise ValueError

            # initialize the variable to have derivative 1
            self.derivatives = {var_name: 1}

            self.value = value
        else:
            raise ValueError("Incorrect number of args")

    def get_gradient(self, var_name):
        """Gets the gradient with respect to a particular variable.

        Accesses the .derivatives dictionary of the Forward object which stores the
        results of the computations that were done by the duner methods during the
        computation of the result and gradient. If the variable name is not within
        the dictionary, this implies that the expression was constant and the
        derivative should be zero. If the stored value is nan, this means there was
        some error during the computation or the gradient does not exist at that
        point.
        """
        grad = self.derivatives.get(var_name, 0)
        # check to see if the computatino was nan, indicating that the gradient
        # most likely does not exist
        if np.isnan(grad):
            raise ValueError("Gradient does not exist!")

        return grad

    @classmethod
    def _with_derivatives(cls, value, derivatives):
        """Creates a Forward object with a particular set of derivatives and value.

        This is a convenience method that allows certain methods to create Forward
        objects with a specific value and derivative dictionary.
        """
        forward = cls(value)

        forward.derivatives = derivatives

        return forward

    def update_derivatives(self, update_fun):
        """Helper method to update each of the derivatives of the Forward object by
        some function update_fun.
        """
        updated = {}

        for key, value in self.derivatives.items():
            updated[key] = update_fun(value)

        return updated

    def binop(
        self, right_forward, left_update, right_update, derivative_update, result_func
    ):
        """Helper method that abstracts away the implementation of binary operators.

        This method reduces the amount of boilerplate necessary to define the various
        binary dunder methods that we must implement.

        Args:
            self - the current Forward object
            right_forward - the Forward object that is the other one in the binary operation
            left_update - the function to update the left derivative
            right_update - the function to update the right derivative
            derivative_update - the function to combine the left and right derivatives
            result_func - the function to compute the result of the function
        """

        # compute the new "derivatives"
        new_left_derivatives = self.update_derivatives(left_update)
        new_right_derivatives = right_forward.update_derivatives(right_update)

        # get all the variables that we can differentiate
        all_vars = self.derivatives.keys() | right_forward.derivatives.keys()

        updated = {}

        for var in all_vars:
            # get the value of derivative, fallback to 0
            l_der = new_left_derivatives.get(var, 0)
            r_der = new_right_derivatives.get(var, 0)

            # combine the derivatives
            updated[var] = derivative_update(l_der, r_der)

        return Forward._with_derivatives(
            result_func(self.value, right_forward.value), updated
        )

    @coerce
    def __add__(self, other):
        return self.binop(
            other, lambda x: x, lambda x: x, lambda x, y: x + y, lambda x, y: x + y
        )

    @coerce
    def __radd__(self, other):
        # the coercion allows us to fallback on __add__!
        return other + self

    @coerce
    def __sub__(self, other):
        return self.binop(
            other, lambda x: x, lambda x: x, lambda x, y: x - y, lambda x, y: x - y
        )

    @coerce
    def __rsub__(self, other):
        return other - self

    @coerce
    def __mul__(self, other):
        # here we have the first interesting use of binop where we define the
        # product rule of differentiation
        return self.binop(
            other,
            lambda x: x * other.value,
            lambda x: x * self.value,
            lambda x, y: x + y,
            lambda x, y: x * y,
        )

    @coerce
    def __rmul__(self, other):
        return other * self

    @coerce
    def __truediv__(self, other):
        # defining the quotient rule here
        return self.binop(
            other,
            lambda x: x * other.value,
            lambda x: x * self.value,
            # defining the overall fraction
            lambda x, y: (x - y) / other.value ** 2,
            lambda x, y: x / y,
        )

    @coerce
    def __rtruediv__(self, other):
        return other / self

    @coerce
    def __pow__(self, other):
        # we use the general power rule defined at
        # https://en.wikipedia.org/wiki/Differentiation_rules#Generalized_power_rule

        res = self.value ** other.value

        return self.binop(
            other,
            lambda x: x * self.value ** (other.value - 1) * other.value,
            lambda x: x * res * np.log(self.value),
            lambda x, y: x + y,
            lambda x, y: res,
        )

    @coerce
    def __rpow__(self, other):
        return other ** self

    @coerce
    def __eq__(self, other):
        return self.value == other.value

    @coerce
    def __lt__(self, other):
        return self.value < other.value

    @coerce
    def __gt__(self, other):
        return self.value > other.value

    @coerce
    def __le__(self, other):
        return self.value <= other.value

    @coerce
    def __ge__(self, other):
        return self.value >= other.value

    @coerce
    def __ne__(self, other):
        return self.value != other.value

    def __neg__(self):
        # define negation as literally just subtracting from zero
        return 0 - self

    def __pos__(self):
        # define pos as a nop
        return self

    def unop(self, value_fun, derivative_fun):
        """Convenience method for defining unary operators on Forward objects.

        This method deals with computing the gradient and result of a unary operator
        while simultaneously taking care of the chain rule.

        Args:
            value_fun - the function to compute the actual result
            derivative_fun - the function that computes the derivative
        """
        updated_ders = {}

        for var in self.derivatives.keys():
            der_value = self.derivatives.get(var, 0)
            updated_ders[var] = derivative_fun(self.value) * der_value

        return Forward._with_derivatives(value_fun(self.value), updated_ders)


@coerce
def sin(x):
    """Computes the sin of the input as well as its gradient"""
    return x.unop(np.sin, np.cos)


@coerce
def cos(x):
    """Computes the cosine of the input along with its gradient"""
    return x.unop(np.cos, lambda x: -np.sin(x))


@coerce
def tan(x):
    """Computes the tangent of the input along with its gradient"""
    return sin(x) / cos(x)


@coerce
def sec(x):
    """Computes the secant of the input"""
    return 1 / cos(x)


@coerce
def csc(x):
    """Computes the cosecant of the input"""
    return 1 / sin(x)


@coerce
def cot(x):
    """Computes the cotangent of the input"""
    return 1 / tan(x)


@coerce
def arcsin(x):
    """Computes the arcsine (inverse sine) of the input"""
    return x.unop(np.arcsin, lambda x: 1 / np.sqrt(1 - x ** 2))


@coerce
def arccos(x):
    """Computes the arccosine (inverse cosine) of the input"""
    return x.unop(np.arccos, lambda x: -1 / np.sqrt(1 - x ** 2))


@coerce
def arctan(x):
    """Computes the arctangent of the input"""
    return x.unop(np.arctan, lambda x: 1 / (1 + x ** 2))


@coerce
def arcsec(x):
    """Computes the arcsecant of the input"""
    return arccos(1 / x)


@coerce
def arccsc(x):
    """Computes the arccosecant of the input"""
    return arcsin(1 / x)


@coerce
def arccot(x):
    """Computes the arccotangent of the input"""
    return arctan(1 / x)


@coerce
def sinh(x):
    """Computes the hyperbolic sine of the input"""
    return (exp(x) - exp(-x)) / 2


@coerce
def cosh(x):
    """Computes the hyperbolic cosine of the input"""
    return (exp(x) + exp(-x)) / 2


@coerce
def tanh(x):
    """Computes the hyperbolic tangent of the input"""
    return sinh(x) / cosh(x)


@coerce
def sech(x):
    """Computes the hyperbolic secant of the input"""
    return 1 / cosh(x)


@coerce
def csch(x):
    """Computes the hyperbolic cosecant of the input"""
    return 1 / sinh(x)


@coerce
def coth(x):
    """Computes the hyperbolic cotangent of the input"""
    return 1 / tanh(x)


@coerce
def _log(x):
    """Computes the natural logarithm of the input along with its gradient"""
    return x.unop(np.log, lambda x: 1 / x)


@coerce
def log(x, base=np.e):
    """Computes the logarithm with specified base of the input.

    The default base is e.
    """
    return _log(x) / _log(base)


@coerce
def log2(x):
    """Computes the log base 2 of the input with its gradient"""
    return log(x, 2)


@coerce
def log10(x):
    """Computes the log base 10 of the input with its gradient"""
    return log(x, 10)


@coerce
def ln(x):
    """Computes the natural logarithm of the input along with its gradient"""
    return log(x)


@coerce
def exp(x):
    """Computes e raised to the input power"""
    return np.e ** x


@coerce
def sqrt(x):
    """Computes the square root of the input"""
    return x ** (1 / 2)


@coerce
def logistic(x):
    """Computes the logistic function of the input.

    The logistic function has the curious property that its gradient is given
    by logistic(x) * (1 - logistic(x)).
    """
    return 1 / (1 + exp(-x))


class fVector:
    """A class that allows users to create vector functions of multiple inputs.

    This class is really just a thin wrapper around the Forward class and just
    redefines the attribute .value and function .get_gradient in terms of the list
    of Forward objects that it was initialized with.
    """

    def __init__(self, values):
        self.values = [_coerce(value) for value in values]

    @property
    def value(self):
        """Allows users to get the result of the vector as an array.

        This function works by simply looping over all the Forward objects it
        contains and returning their values.
        """
        return [v.value for v in self.values]

    def get_gradient(self, var_name):
        """Returns the gradient of each of the vector components with respect
        to the given variable.

        This function also loops over each of the Forward objects and asks
        for the correct gradient.
        """
        return [v.get_gradient(var_name) for v in self.values]

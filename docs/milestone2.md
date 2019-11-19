# Group 13 - Milestone1

## Introduction

> Describe the problem the software solves and why it's important to solve that problem.

Our software aims to compute derivatives accurately and in a cost-efficient manner.
We will create a python package that implements the forward and reverse modes of automatic
differentiation. This is important because it overcomes the limitations of symbolic differentiation
(computationally intensive) and the finite difference method (accuracy/stability).

## Background

> Describe (briefly) the mathematical background and concepts as you see fit. You do not need
> to give a treatise on automatic differentation or dual numbers. Just give the essential ideas
> (e.g. the chain rule, the graph structure of calculations, elementary functions, etc).
> Do not copy and paste any of the lecture notes. We will easily be able to tell if you did
> this as it does not show that you truly understand the problem at hand.

Automatic differentiation (AD) breaks a function into a graph of elementary functions (ex: addition, multiplication, or log), and at each step, it calculates the value of both the function and its derivative.

Because the elemental functions are simple, we can calculate the derivative at each step to machine precision, thereby avoiding the stability issues with the numerical approach. Any elemental node simply uses the chain rule to determine its derivative, applying the derivatives from previous nodes along with the symbolic derivative of the elemental function. One should note that an elemental function’s derivative far simpler to define symbolically than that of the complete original function.

The decomposition of the original function into elemental functions is illustrated as a directed graph, with each node representing an elemental function and each edge carrying the results of one elemental function to be used as the inputs for another. As the graph is evaluated, the values are kept within a trace table, showing the function’s value and derivative at each step of the AD process.

Once the process has traversed across the entire graph, the final node should have both the value and the derivative of the function at the given point of evaluation. These values will also be available in the final row of the trace table.

## How to Use `autodiffpy`

> How to install? Even (especially) if the package isn't on PyPI, you should walk them through the creation of a virtual environment or some other kind of manual installation.

A user will install `autodiffpy` using `pip`.

```console
eholmes: pip install autodiffpy
```

A user will most likely import two important exports from our package

```python
from autodiffpy import Var, Vector
```

The `Var` class can be used to instantiate abstract variables that can be differentiated later. For example,

```python
f = Var(‘x’)
```

To create interesting functions, the user can just directly use the standard mathematical operations on our `Var` object.

```python
x = Var(‘x’)
f = x ** 2 + 3 * x - x / 4
```

If the user would like to use more special methods (like sin, cos, ln, anything that can’t be implemented with dunder methods), they will have to import special ones from our package

```python
from autodiffpy import Sin, Cos, Var
f = Cos(Sin(2 * Var(‘x’)))
```

The user can also have multiple variables

```python
x = Var(‘x’)
y = Var(‘y’)
f = x ** (2 * y)
```

If the user would like to have a multidimensional output, they can combine multiple functions using the `Vector` class

```python
x = Var(‘x’)
y = Var(‘y’)
f = Vector([y ** 2, x * y])
```

The type of the output of these combinations will always be an `AutoDiff` object (including `Vector` and `Var`). The user can then call the `AutoDiff` object directly to get the value of the function at a point. The arguments must be passed in as named parameters

```python
x = Var(‘x’)
y = Var(‘y’)
f = Vector([y ** 2, x * y])
f(x=1, y=2)
```

To get the derivative, the user can call `.derivative` on the `AutoDiff` object and pass in the variable to differentiate with respect to as well as the point they would like to get the derivative at.

```python
x = Var(‘x’)
y = Var(‘y’)
f = Vector([y ** 2, x * y])
f.derivative(‘x’, x=1, y=2)
f.derivative(‘y’, x=1, y=2)
```

The distinction between forward and backward mode will be hidden from the user and decided automatically depending on the number of inputs and outputs of the function.

## Software Organization

> Discuss how you plan on organizing your software package.
>
> What will the directory structure look like?
> What modules do you plan on including? What is their basic functionality?
> Where will your test suite live? Will you use TravisCI? CodeCov?
> How will you distribute your package (e.g. PyPI)?
> How will you package your software? Will you use a framework? If so, which one and why? If not, why not?
> Other considerations?

```
cs207-FinalProject/
    README.md
    LICENSE
    .travis.yml
    __init__.py
    docs/
        milestone1.ipynb
    source/
        __init__.py
        forward.py
        reverse.py
        test/
            __init__.py
            test_forward.py
            test_reverse.py
```

There are two main modules we plan to implement - `forward` and `reverse`. The `forward` module performs Automatic Differentiation in the forward mode, whereas the `reverse` module achieves the same purpose in the reverse mode. As demonstrated in the `How to use` section, each module is a class that can be initialized with a variable. The instance can then be used to construct more complicated equations while still being the same type. To access the function value and derivative at a specific value, we can access the `val` and `der` attributes of the instance.

Within each module, we overwrite all of the elementary functions including `add`, `mul`, and several other functions such as `sin` and `exp`. With the help of these functions, we can calculate function values and derivatives for a broad range of complex functions.

The test suite will live inside the `.source/test/` directory. It has two test files, each for one module, testing for code correctness and code coverage. For correctness, we are using the `pytest` package in conjunction with <font color="red">TravisCI</font>; for code coverage, we are using <font color="red">CodeCov</font>. And we plan to distribute our package using <font color="red">PyPI</font>.

We plan to package our software, but will not use any framework. We believe that the package itself is pretty straight forward, and we should be able to layout the package structure ourselves - it is also a good learning opportunity!

## Implementation

Implementation

> Discuss how you plan on implementing the forward mode of automatic differentiation.
>
> - What are the core data structures?
> - What classes will you implement?
> - What method and name attributes will your classes have?
> - What external dependencies will you rely on?
> - How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?
>
> Be sure to consider a variety of use cases. For example, don't limit your design to scalar
> functions of scalar values. Make sure you can handle the situations of vector functions of
> vectors and scalar functions of vectors. Don't forget that people will want to use your library
> in algorithms like Newton's method (among others).

We will define a class `Dual` that implements arithmetic over the dual numbers. To override the standard mathematical operations like +, -, /, and \*, we can use the dunder methods like `__add__`, `__mul__`, etc. For more complicated functions like sin, sqrt, log, etc, we will have to define our own special methods sin, sqrt, that can operate on Dual objects.

```python
class Dual:
    def __init__(self, a, b):
        ...

    def __add__(self, other):
        ...

    def __radd__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __rpow__(self, other):
        ...

def dual_sin(d):
    ...

def dual_cos(d):
    ...
```

We will define a high-level class `AutoDiff`. This will implement the dunder methods `__add__`, `__mul__`, etc to allow for composition of `AutoDiff` objects. We will also have to define special methods or classes like `Sin`, `Sqrt`, etc. that can operate on `AutoDiff` objects. The `AutoDiff` class will also implement `__call__` to allow users to call the resulting object on certain inputs. The `AutoDiff` class will also implement a method called `.derivative` that will allow users to differentiate with respect to a variable and point.

```python
class AutoDiff:
    def __init__(self, variables, expressions, operator):
        self.variables = variables
        self.expressions = expressions
        self.operator = operator
        ...

    def derivative(self, variable, **kwargs):
        ...

    def __call__(self, **kwargs):
        ...

    def __add__(self, other):
        ...

    def __radd__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __rsub__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __rtruediv__(self, other):
        ...

    def __pow__(self, other):
        ...

    def __rpow__(self, other):
        ...

class Sin(AutoDiff):
    ...

class Cosine(AutoDiff):
    ...

class Var(AutoDiff):
    ...

class Vector(AutoDiff):
    ...
```

The AutoDiff class will have the following attributes:

- `variables` - a set of all the variables that have been declared in the expression
- `expressions` - a list of all the expressions that are part of the node
- `operator` - stores the operator of the current AutoDiff object. Tells whether evaluation should use addition, multiplication, subtraction, etc…

Together these attributes will also implicitly define a computation graph with the AutoDiff objects acting as nodes.

We will use `twine` to distribute the package on PyPI. Otherwise, we do not expect to rely on other external dependencies.

## Additional Comments

> There is no need to have an implementation started for Milestone 1. You are currently in the
> planning phase of your project. This means that you should feel free to have a project_planning
> repo in your project organization for scratch work and code.
> The actual implementation of your package will start after Milestone 1.

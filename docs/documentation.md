# Group 13 - Documentation of `autodiffpy`

## Introduction

Our software aims to compute derivatives accurately and in a cost-efficient
manner. Our Python package, `autodiffpy`, implements the forward and reverse
modes of automatic differentiation. This computation is important because it
overcomes the limitations of symbolic differentiation (computationally
intensive) and the finite difference method (inaccurate/unstable).

Derivatives are highly important in numerical algebra. We can use derivatives
to find the roots of differentiable functions or the minima of other
functions. This second use case has become increasingly important in the
field of machine learning where optimizing and "learning" in deep neural
networks requires finding local minima of highly nested functions.

## Background

Automatic differentiation (AD) breaks a function into a graph of elementary functions (ex: addition, multiplication, or log), and at each step, it calculates the value of both the function and its derivative.

Because the elemental functions are simple, we can calculate the derivative at each step to machine precision, thereby avoiding the stability issues of the numerical approach. Any elemental node simply uses the chain rule to determine its derivative, applying the derivatives from previous nodes along with the symbolic derivative of the elemental function. One should note that an elemental function's derivative is far simpler to define symbolically than that of the complete original function.

The decomposition of the original function into elemental functions is illustrated as a directed graph, with each node representing an elemental function and each edge carrying the results of one elemental function to be used as the inputs for another. As the graph is evaluated, the values are kept within a trace table, showing the function's value and derivative at each step of the AD process.

Once the process has traversed across the entire graph, the final node should have both the value and the derivative of the function at the given point of evaluation. These values will also be available in the final row of the trace table.

The forward mode of automatic differentiation actually computes the product of the Jacobian matrix, $J$, where the $i,j^{th}$ entry is $J_{i,j} = \frac{\partial(f_i)}{\partial(x_j)}$ for function $f_i$ and variable $x_j$, and a seed vector, $p$, which represents the initial derivatives of each variable. Thus, if we have a function with two variables $x,y$ and we choose the seed vector $p=(1,0)$, then the forward mode of automatic differentiation would output $\frac{\partial(f)}{\partial(x)$.

To illustrate the forward mode, let's consider the function
$$f(x,y) = e^xsin(x+2y)$$
where the inputs are $x$ and $y$. We would like to evaluate the function and find its derivative at $x=0$ and $y = \frac{\pi}{2}$. Additionally, we will seed the derivatives with $(1,1)$, which indicates that we would like to find derivatives of both $x$ and $y$. First, we can draw the computational graph of this function.

![Trace Table](pics/milestone2-comp-graph.jpg)

The trace table is shown below.

![Trace Table](pics/milestone2-trace-table.png)

It is evident from the trace table that the final value of the function is 0, while the partial derivatives with respective to $x$ and $y$ are -1 and -2, respectively.

One should note that the forward mode of automatic differentiation is more efficient when the number of outputs far exceeds the number of inputs.

## How to Use `autodiffpy`

### Installation

You can install `autodiffpy` from `PyPI` by running `pip` from the command line.

```bash
pip install autodiffpy-free-holmes
```

Alternatively, you can install the package directly from our GitHub repository.

```bash
pip install git+git://github.com/free-holmes/cs207-FinalProject.git
```

### Basic Demo

This section assumes a user will be working with the forward mode of automatic differentiation. Further instructions on reverse mode can be found in the extension section.

To access the forward mode, a user can import the `Forward` class from the `autodiffpy.forward` module.

```python
from autodiffpy.forward import Forward
```

The `Forward` class can be used to instantiate a variable along with its
initial value. In our demo, suppose we want to evaluate and find the
derivative of the function f(x) = sin(x^2) at the point x = 2. Then we would need

```python
from autodiffpy.forward import Forward, sin
x = Forward('x', 2)
f = sin(x ** 2)

print(f.value) # get the value of the function at x = 2
print(f.get_gradient('x')) # get the derivative of the function at x = 2
```

Note the use of the special import `sin` from the module as well. There are many other elementary functions defined within the module that allow users to take the derivative of those functions. These will be elaborted on in the Implementation section.

If the user wants functions of multiple inputs and outputs, they can use the `fVector` constructor from the module in conjunction with multiple `Forward` object instances. For example,

```python
from autodiffpy.forward import Forward, fVector
x = Forward('x', 2)
y = Forward('y', -3)

f = fVector([x * y, x + y]) # represents f(x, y) = (xy, x + y)

print(f.value) # returns [-6, -1]
print(f.get_gradient('x')) # returns [-3, 1]
print(f.get_gradient('y')) # returns [2, 1]
```

## Software Organization

### Directory Structure

```
cs207-FinalProject/
    autodiffpy/
        __init__.py
        demo.py
        forward.py
        reverse.py
        test/
            test_demo.py
            test_forward.py
            test_reverse.py
    docs/
        documentation.md
        milestone1.md
        milestone2.md
        pics/
            milestone2-comp-graph.jpg
            milestone2-trace-table.png
    .gitignore
    .travis.yml
    LICENSE
    README.md
    setup.py
```

### Modules

The primary exported module is `autodiffpy` which lives in its entirety under the `autodiffpy` folder. This contains all the publicly accessible interfaces.

Under the `autodiffpy` folder is the `forward.py` file/module. This contains the logic and implementation of the forward mode of automatic differentiation. It defines the `Forward` class and elementary functions that can be used to chain together complicated and interesting functions while simultaneously computing their gradient.

The `autodiffpy` folder also contains the `reverse.py` file/module. This contains the logic and implementation of the reverse mode of automatic differentiation, which is our extension feature.

### Tests

Tests live under the `autodiffpy/test` folder. They can be run from the
top-level of the repository by doing the following:

```bash
pip install -r developer-requirements.txt
python -m pytest --cov=autodiffpy --cov-report=term-missing
```

Tests are run on every commit to the repository via `TravisCI`. Coverage information is
generated via the above call to `pytest`. Coverage information is stored via a call to `codecov`
during the Travis build.

### Deployment

The `autodiffpy-free-holmes` Python package is automatically uploaded to `PyPI` upon each push to master with a version bump.

## Implementation

### The `Forward` class

As mentioned in the Demo section, the main object of interest is the
`Forward` class defined in the `autodiffpy.forward` module. This class allows
us to construct variables that can then be chained together with standard Python mathematical operations as well as several elementary functions that we define.

If the `Forward` class is constructed with two arguments, the first must be the name of the variable and the second must be the initial value of the variable. As an example:

```python
from autodiffpy.forward import Forward
x = Forward('x', 2)
```

This defines a variable `x` with initial value `2`. If the `Forward` class is constructed with one argument, it must be a numeric type. This is how constants can be defined in our forward differentiation. For example

```python
c = Forward(3)
```

This creates a constant with value 3.

### Standard Python operators

A user can then use the standard Python operators to combine these `Forward` objects with other forward objects or numeric Python constants to actually compute the output of functions. All operators including `+`, `-` (both unary and binary), `*`, `/`, `**` are all supported.

```python
x = Forward('x', -2.5)
f = -(x - 2) * (x + 3) ** 4 / (2 ** x)
```

### Accessing the Result and Gradient

The resulting `f` above is also a `Forward` object. To actually access the result of the computation and the gradient with respect to `x`, we can look at the `.value` property and use the `.get_gradient` method.

```python
print(f.value)
>>> 1.590990257669732
print(f.get_gradient('x'))
>>> 11.271578259362466
```

Note that `.get_gradient` accepts as its parameter the name of the variable that we want the gradient with respect to.

### Comparisons

In addition to the mathematical Python operators, we also support the comparison of `Forward` objects via the standard comparison operators. This simply compares the output value of expressions.

```python
x = Forward('x', -3)
x == x # True
x != x # False
x ** 2 > x # True
```

### Elementary Functions

The `autodiffpy.forward` module also comes with many elementary functions including

- trigonometric functions and their inverses
- hyperbolic functions
- exponential function
- logistic function
- logarithms of any base
- square root function

These are exported from `uutodiffpy.forward` under the standard names like
`sin`, `cos`, `sinh`, `csch`, `logistic`, `exp`, `ln`, etc. We can apply
these elementary functions to the `Forward` objects to get more interesting
derivatives. As an example:

```python
from autodiffpy.forward import Forward, sin, logistic
x = Forward('x', -23)
f = sin(logistic(x))

print(f.value, f.get_gradient('x'))
>>> 1.0261879630648841e-10 1.0261879629595781e-10
```

### Functions of Multiple Inputs

By defining other `Forward` objects with different variable names, the user
can create functions that take in multiple inputs. The name can be anything. As an example:

```python
from autodiffpy.forward import Forward
x = Forward('x', -2)
y = Forward('y', 3)

f = x + y + x * y
print(f.value, f.get_gradient('x'), f.get_gradient('y'), f.get_gradient('z'))
>>> -5 4 -1 0
```

### Functions of Multiple Outputs

We can also define a function with multiple or vector output by using our `fVector` constructor that is included in `autodiffpy.forward`. The `fVector` accepts an array of expressions as its argument. To access the value is the same as the `Forward` object, simply use `.value` and `.get_gradient`.

```python
from autodiffpy.forward import fVector

x = Forward('x', -3)
y = Forward('y', 2)

f = fVector([x + y, x * y])

print(f.value, f.get_gradient('x'), f.get_gradient('y'))
>>> [-1, -6] [1, 2] [1, -3]
```

### External Dependencies

We only rely on `numpy` as our external dependency. We use `numpy` to compute
the values of our elementary functions including the trig and inverse trig
functions, log, exp, hyperbolic functions, and the logistic function.

## Extension: Reverse Mode

We implemented automatic differentiation reverse mode as our extension.
Reverse mode can be more efficient than forward mode when there are far more
inputs than outputs.

### Mathematical Background

In the forward mode of automatic differentiation, we start at the inputs and
apply the chain rule going forward through the implicit computational graph.
At each "node" in the graph, we take the derivative with respect to the input
parameter until we reach the final node in the graph. At this point we will
have our final derivative of interest.

The reverse mode, as the name implies, works in the opposite direction. We
start by initializing the final node in the graph with a gradient value of 1.
We then can, in a sense, work backwards from that node to derive the gradient
at all other nodes. This is relying on a form of the chain rule.

At each node, `w_i`, we compute the derivative `df/dw_i` by breaking this into

```
df/dw_i = df/dw_{i + 1} * dw_{i + 1}/dw_i
```

in the case that the node only has one child. This generalizes in the case where the node has two children. We
can continue to unwrap the value `df/dw_i` recurisvely until we reach `df/dw_n` which we know is 1.

Since the reverse mode can propagate backwards the gradient from a single
output to potentially multiple inputs, it can be efficient when there is one
output but multiple inputs.

### Implementation

The following functions are supported in reverse mode: `exp`, `ln`, `log`,
`log2`, `log10`, `sqrt`, `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `sinh`,
`cosh`, `tanh`, `sech`, `csch`, `coth`

The `log` function defaults to natural log but has an optional parameter `base` that the user can specify.

```python
from autodiffpy.reverse import Reverse, log

x = Reverse(9)
f = log(x, 3) # Specify 3 as the log base
f.gradient_value = 1.0 # Set seed value, very important!!!!!!
f.value # Returns 2.0
```

### How to use: Scalars

Non-vector operations in reverse mode are almost identical to those in the
forward mode. The primary difference is that we now utilize a `Reverse`
object instead of a `Forward` object. In addition, the user must seed the
gradient of the function, typically with a value of `1`.

Note that the derivative can be found by calling `get_gradient` in reverse
mode on the _variable_, while in the forward mode it was found by calling
`get_gradient` on the function result.

```python
from autodiffpy.reverse import Reverse, exp

# Declare reverse variables
x = Reverse(1)
y = Reverse(2)

# Declare function
func = x * y + exp(x*y)

# Set seed value
func.gradient_value = 1.0

# Get values and derivatives
print(func.value)
>>> 9.38905609893065
print(x.get_gradient())
>>> 16.7781121978613
print(y.get_gradient())
>>> 8.38905609893065
```

### How to use: Vectors

Vector operations in reverse mode are somewhat different from those in the
forward mode. Users should createa list of expressions using the `Reverse`
objects in conjunction with the standard Python mathematical operators.

To get the vector gradient, a user should call `get_gradients` and pass in
the input `Reverse` object that they want the gradient of. This will return a
list of values, which represent the gradient of the vector of functions with
respect to each element of the vector. To get the value of the vector, the
user can simply access `.values` on the vector object.

```python
from autodiffpy.reverse import Reverse, rVector

# Declare reverse variables
x = Reverse(1)
y = Reverse(2)

# Create list of functions
functions = [x * 2 * y + y ** 3, 2 * x ** 2 * y, 3 * y]
vector = rVector(functions)

# Get values and derivatives
print(vector.values)
print(vector.get_gradients(x))
print(vector.get_gradients(y))

>>> [12, 4, 6]
>>> [4.0, 8.0, 0]
>>> [14.0, 2.0, 3.0]
```

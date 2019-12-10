# `autodiffpy` - Group 13

[![Build Status](https://travis-ci.org/free-holmes/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/free-holmes/cs207-FinalProject.svg?branch=master) [![Coverage Status](https://codecov.io/gh/free-holmes/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/free-holmes/cs207-FinalProject)

`autodiffpy` is a library implementing automatic differentiation to make the calculation of gradients of complex functions fast and precise.

## Installation

```bash
pip install autodiffpy-free-holmes
```

## Getting Started with Forward mode

```python
from autodiffpy.forward import Forward

x = Forward('x', 3)
f = x ** 2

print(f.value, f.get_gradient('x'))
>>> 9 6
```

## Getting Started with Reverse mode

```python
from autodiffpy.reverse import Reverse

x = Reverse(3)
f = x ** 2

f.gradient_value = 1

print(f.value, x.get_gradient())
```

## Learn more

To learn more about the features of forward and reverse mode, including functions of multiple inputs and outputs and elementary functions, head to our documentation at [our docs](https://github.com/free-holmes/cs207-FinalProject/blob/master/docs/documentation.md).

## Authors

- Teddy Liu
- Tapley Stephenson
- Zihao Xu
- David Zheng

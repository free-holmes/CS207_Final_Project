import pytest
from source.forward import AutoDiffForward

def test_value():
    a = 2.0
    x = AutoDiffForward(a)
    alpha = 2.0
    beta = 3.0

    f = beta + alpha * x
    assert (f.val, f.der) == (7.0, 2.0)

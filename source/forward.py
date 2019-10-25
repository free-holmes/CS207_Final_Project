class AutoDiffForward:
    def __init__(self, v, d=1):
        self.val = v
        self.der = d

    def __add__(self, other):
        try:
            val = self.val + other.val
            der = self.der + other.der
        except AttributeError:
            val = self.val + other
            der = self.der

        return AutoDiffForward(val, der)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            val = self.val * other.val
            der = self.val*other.der + other.val*self.der
        except AttributeError:
            val = self.val * other
            der = self.der * other

        return AutoDiffForward(val, der)

    def __rmul__(self, other):
        return self.__mul__(other)

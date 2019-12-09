from autodiffpy.forward import Forward


def newtons_method(f, initial_guess=0):
    while True:
        x = Forward("x", initial_guess)
        updated = f(x)

        update = updated.value / updated.get_gradient("x")

        if abs(update) < 10 ** -12:
            break

        initial_guess -= update

    return initial_guess


if __name__ == "__main__":  # pragma: no cover

    def fun(x):
        return x ** x - 2

    zero = newtons_method(fun, 1)

    print(zero, fun(zero))

from autodiffpy.forward import Var


def newtons_method(f, initial_guess=0):
    while True:
        update = f(x=initial_guess) / f.derivative("x", x=initial_guess)

        if abs(update) < 10 ** -12:
            break

        initial_guess -= update

    return initial_guess


if __name__ == "__main__":  # pragma: no cover
    f = Var("x") ** Var("x") - 2

    zero = newtons_method(f, 1)

    print(zero, f(x=zero))

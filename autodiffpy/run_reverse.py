from reverse import *

def create_reverse_function():
    # declare variables that will be used in expression
    while True:
        try:
            var_list = input("Enter the variable names separated by a space:")
            var_list = list(var_list.split(' '))
            for i in var_list:
                assert i[0].isalpha()
            break
        except AssertionError:
            print(f'INVALID INPUT: {var_list}\nAll variable names must be alphanumeric.')

    # set value for declared variables
    counter = 0
    while counter < len(var_list):
        var_name = var_list[counter]
        val = input(f'Enter the value of {var_name}:')
        try:
            vars()[var_name] = Reverse(float(val))
            counter += 1
        except ValueError:
            print(f'INVALID INPUT: {val}\nValue for {var_name} must be a real number.')

    # input the expression
    expr = input("Enter the function:")

    # evaluate the expression
    eval_expr = eval(expr)
    try:
        eval_expr.gradient_value = 1
        print(f'expression = {eval_expr.value}')
        for i in var_list:
            print(f'{i} gradient = {vars()[i].get_gradient()}')
    except AttributeError:
        print(f'There are no variables in the expression you entered: {expr}')
        print(f'expression = {eval_expr}')


if __name__ == "__main__":
    create_reverse_function()

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.symbols('x')
y = sp.symbols('y')
f = sp.symbols('f')
alpha = sp.symbols('alpha')

arr_x = []
arr_y = []


def goldenSectionFirstStep(function, a, b, e):
    t = 1.618034
    length = abs(a - b)
    x11 = a + (1 - 1 / t) * length
    x12 = a + length / t
    x11value = function.evalf().subs({alpha: x11})
    x12value = function.evalf().subs({alpha: x12})
    if x11value < x12value:
        b = x12
        nextPoint = x11
        nextValue = x11value
    else:
        a = x11
        nextPoint = x12
        nextValue = x12value
    if abs(a - b) > e:
        return goldenSectionNextSteps(function, nextPoint, nextValue, a, b, e)
    else:
        return (a + b) / 2


def goldenSectionNextSteps(function, x1, x1value, a, b, e):
    x2 = a + b - x1
    if x2 < x1:
        temp = x2
        x2 = x1
        x1 = temp
        x2value = x1value
        x1value = function.evalf().subs({alpha: x1})
    else:
        x2value = function.evalf().subs({alpha: x2})
    if x1value < x2value:
        b = x2
        nextPoint = x1
        nextValue = x1value
    else:
        a = x1
        nextPoint = x2
        nextValue = x2value
    if abs(a - b) > e:
        return goldenSectionNextSteps(function, nextPoint, nextValue, a, b, e)
    else:
        return (a + b) / 2


def getFibValue(n):
    return int(math.pow((1 + math.sqrt(5)) / 2, n) - math.pow((1 - math.sqrt(5)) / 2, n)) / math.sqrt(5)


def fibonacci(function, a0, b0, e):
    n = (b0 - a0) / e

    it_number = 0

    num = 0
    while n < getFibValue(num) or n > getFibValue(num + 1):
        num += 1

    num += 1
    a = a0
    b = b0
    l = (b - a) / getFibValue(num)
    while num > 0:
        it_number += 1
        x1 = a + l * getFibValue(num - 2)
        x2 = b - l * getFibValue(num - 2)

        if function.evalf().subs({alpha: x1}) < function.evalf().subs({alpha: x2}):
            b = x2
        else:
            a = x1
        num -= 1
    return a + (abs(b - a) / 2)


def gradient_descent(function, start_x, start_y, learn_rate, n_steps=500, tolerance=0.0001):
    new_x = start_x
    new_y = start_y
    x_derivative = sp.diff(function, x)
    y_derivative = sp.diff(function, y)
    for _ in range(n_steps):
        arr_x.append(new_x)
        arr_y.append(new_y)
        diff_x = -learn_rate * x_derivative.evalf().subs({x: new_x, y: new_y})
        diff_y = -learn_rate * y_derivative.evalf().subs({x: new_x, y: new_y})
        if abs(diff_x) < tolerance and abs(diff_y) < tolerance:
            break
        new_x += diff_x
        new_y += diff_y
    return [new_x, new_y]


def gradient_descent_with_step_friction(function, start_x, start_y, alpha_x, alpha_y, e, lamd, n_steps=500,
                                        tolerance=0.0001):
    new_x = start_x
    new_y = start_y
    x_derivative = sp.diff(function, x)
    y_derivative = sp.diff(function, y)
    for _ in range(n_steps):
        arr_x.append(new_x)
        arr_y.append(new_y)
        old_x = new_x
        old_y = new_y
        new_x = new_x - alpha_x * x_derivative.evalf().subs({x: new_x, y: old_y})
        new_y = new_y - alpha_y * y_derivative.subs({x: new_x, y: old_y})
        if function.evalf().subs({x: new_x, y: new_y}) - function.evalf().subs(
                {x: old_x, y: old_y}) > -alpha_x * e * abs(x_derivative.evalf().subs({x: old_x, y: old_y}) ** 2):
            alpha_x *= lamd
        if function.evalf().subs({x: new_x, y: new_y}) - function.evalf().subs(
                {x: old_x, y: old_y}) > -alpha_y * e * abs(x_derivative.evalf().subs({x: old_x, y: old_y}) ** 2):
            alpha_y *= lamd
        if abs(new_x - old_x) < tolerance and abs(new_y - old_y) < tolerance:
            break
    return [new_x, new_y]


def gradient_descent_with_golden_section_or_fibo(function, start_x, start_y, method_value, n_steps=50,
                                                 tolerance=0.0001):
    new_x = start_x
    new_y = start_y
    x_derivative = sp.diff(function, x)
    y_derivative = sp.diff(function, y)
    for _ in range(n_steps):
        old_x = new_x
        old_y = new_y

        arr_x.append(new_x)
        arr_y.append(new_y)

        func_new_x = new_x - alpha * x_derivative.evalf().subs({x: new_x, y: old_y})
        func_new_y = new_y - alpha * y_derivative.evalf().subs({x: new_x, y: old_y})

        # print(function.evalf().subs({x: func_new_x, y: func_new_y}))
        if method_value == 1:
            new_alpha = goldenSectionFirstStep(function.evalf().subs({x: func_new_x, y: func_new_y}), 0, 1, tolerance)
        elif method_value == 2:
            new_alpha = fibonacci(function.evalf().subs({x: func_new_x, y: func_new_y}), 0, 1, tolerance)
        print(new_alpha)
        # function for golden section
        new_x = old_x - new_alpha * x_derivative.evalf().subs({x: old_x, y: old_y})
        new_y = old_y - new_alpha * y_derivative.evalf().subs({x: old_x, y: old_y})
        print(new_x, new_y)

        if abs(new_x - old_x) < tolerance and abs(new_y - old_y) < tolerance:
            break
    return [new_x, new_y]


def get_gradient_norm(x_derivative, y_derivative, new_x, new_y):
    return math.sqrt(pow(x_derivative.evalf().subs({x: new_x, y: new_y}), 2) +
                     pow(y_derivative.evalf().subs({x: new_x, y: new_y}), 2))


def MEGA_gradient_descent(function, start_x, start_y, n, tolerance=0.001):
    new_x = start_x
    new_y = start_y

    x_derivative = sp.diff(function, x)
    y_derivative = sp.diff(function, y)

    betta = 0
    p_x = -x_derivative.evalf().subs({x: new_x, y: new_y})
    p_y = -y_derivative.evalf().subs({x: new_x, y: new_y})
    n_count = 0

    while get_gradient_norm(x_derivative, y_derivative, new_x, new_y) > tolerance:
        prev_x = new_x
        prev_y = new_y

        arr_x.append(new_x)
        arr_y.append(new_y)

        func_new_x = new_x - alpha * x_derivative.evalf().subs({x: new_x, y: prev_y})
        func_new_y = new_y - alpha * y_derivative.evalf().subs({x: prev_x, y: new_y})

        learn_rate = goldenSectionFirstStep(function.evalf().subs({x: func_new_x, y: func_new_y}), 0, 1, tolerance)

        new_x += (learn_rate * p_x)
        new_y += (learn_rate * p_y)
        p_x = -x_derivative.evalf().subs({x: new_x, y: new_y}) + betta * p_x
        p_y = -y_derivative.evalf().subs({x: new_x, y: new_y}) + betta * p_y
        n_count += 1
        if n_count == n:
            n_count = 0
            betta = 0
        else:
            betta = get_gradient_norm(x_derivative, y_derivative, new_x, new_y) ** 2 / get_gradient_norm(x_derivative,
                                                                                                         y_derivative,
                                                                                                         prev_x,
                                                                                                         prev_y) ** 2
        print(new_x, new_y, betta)

    return [new_x, new_y]


func = 9 * x ** 2 + y ** 2
# func = (x + 10) ** 2 + (y - 3) ** 2
# print(gradient_descent(func, 10, 10, 0.01))
# print(gradient_descent_with_step_friction(func, 10, 10, 0.003, 0.003, 0.006, 0.5))
# print(gradient_descent_with_golden_section_or_fibo(func, 10, 10, 1))  # last value: 1 for golden section, 2 for fibo
# print(gradient_descent_with_golden_section_or_fibo(func, 10, 10, 1))
print(MEGA_gradient_descent(func, 70, 70, 2))

fig, ax = plt.subplots(1, 2)

a = np.arange(-100, 100, 0.2)
b = np.arange(-100, 100, 0.2)
agrid, bgrid = np.meshgrid(a, b)

zgrid = 9 * agrid ** 2 + bgrid ** 2

# print(arr_x, arr_y)

ax[0].contour(agrid, bgrid, zgrid)
ax[0].plot(arr_x, arr_y)

plt.show()

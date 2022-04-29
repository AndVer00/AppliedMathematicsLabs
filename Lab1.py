import math


def dichotomy(a, b, delta, e, n):
    function_counter = n * 2
    n = n + 1
    if 2 * delta > e or a > b:
        print("Error")
        return
    print([a, b])
    x1 = (a + b) / 2 - delta
    x2 = (a + b) / 2 + delta
    if function(x1) < function(x2):
        b = x2
    else:
        a = x1
    if abs(a - b) > e:
        return dichotomy(a, b, delta, e, n)
    else:
        print("Dichotomy, n", n)
        print("Dichotomy, functions", function_counter)
        return (a + b) / 2


def goldenSectionFirstStep(a, b, e):
    t = 1.618034
    length = abs(a - b)
    x11 = a + (1 - 1 / t) * length
    x12 = a + length / t
    x11value = function(x11)
    x12value = function(x12)
    if x11value < x12value:
        b = x12
        nextPoint = x11
        nextValue = x11value
    else:
        a = x11
        nextPoint = x12
        nextValue = x12value
    if abs(a - b) > e:
        return goldenSectionNextSteps(nextPoint, nextValue, a, b, e, 1)
    else:
        return (a + b) / 2


def goldenSectionNextSteps(x1, x1value, a, b, e, n):
    n += 1
    x2 = a + b - x1
    if x2 < x1:
        temp = x2
        x2 = x1
        x1 = temp
        x2value = x1value
        x1value = function(x1)
    else:
        x2value = function(x2)
    if x1value < x2value:
        b = x2
        nextPoint = x1
        nextValue = x1value
    else:
        a = x1
        nextPoint = x2
        nextValue = x2value
    if abs(a - b) > e:
        return goldenSectionNextSteps(nextPoint, nextValue, a, b, e, n)
    else:
        print("Golden section", n)
        return (a + b) / 2


def getFibValue(n):
    return int(math.pow((1 + math.sqrt(5)) / 2, n) - math.pow((1 - math.sqrt(5)) / 2, n)) / math.sqrt(5)


def fibonacci(a0, b0, e):
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
        print("[", a, ",", b, "]")
        x1 = a + l * getFibValue(num - 2)
        x2 = b - l * getFibValue(num - 2)

        if function(x1) < function(x2):
            b = x2
        else:
            a = x1
        num -= 1
    print("Fibonacci", it_number)
    return a + (abs(b - a) / 2)


def parabola(x1, x3, d, e):
    k = 0
    x2 = x1
    f1 = function(x1)
    f3 = function(x3)
    while (function(x2) >= f1 or function(x2) >= f3) and (x2 <= x3 - d):
        x2 += d

    if x2 >= x3 - d:
        if function(x1) < function(x3):
            return x1
        else:
            return x3

    x2_old = x3

    print(x2)
    while abs(x2_old - x2) >= e:
        print("[", x1, ",", x3, "]")
        k += 1
        f1 = function(x1)
        f2 = function(x2)
        f3 = function(x3)

        a1 = (f2 - f1) / (x2 - x1)
        a2 = (1 / (x3 - x2)) * (((f3 - f1) / (x3 - x1)) - ((f2 - f1) / (x2 - x1)))

        x = (1 / 2) * (x1 + x2 - (a1 / a2))

        x2_old = x2

        if x <= x2:
            x3 = x2
            x2 = x
        else:
            x1 = x2
            x2 = x

    print("Parabola:", k)
    return x


def goldenSectionForBrentMethod(a, b, x):
    if x > (a + b) / 2:
        u = x * (math.sqrt(5) - 1) / 2 + a * ((3 - math.sqrt(5)) / 2)
    else:
        u = x * (math.sqrt(5) - 1) / 2 + b * ((3 - math.sqrt(5)) / 2)
    return u


def brentMethod(a, b, eps):
    n = 0
    v = w = x = a + ((3 - math.sqrt(5)) / 2) * (b - a)
    e = 0
    fx = function(x)
    fw = function(w)
    fv = function(v)
    while True:
        n += 1
        if b - a < eps:
            print("Brent", n)
            return (a + b) / 2
        else:
            p = (math.pow(v - x, 2) * (fx - fw) + math.pow(w - x, 2) * (fv - fx))
            q = (v - x) * (fx - fw) + (w - x) * (fv - fx)
            if q == 0:
                u = goldenSectionForBrentMethod(a, b, x)
                fu = function(u)
            else:
                u = x + 0.5 * (p / q)
                fu = function(u)
                e = p / q
                if u < a or u > b or abs(p / q) > 0.5 * abs(e):
                    u = goldenSectionForBrentMethod(a, b, x)
            if function(u) <= function(x):
                if u >= x:
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu
            else:
                if u >= x:
                    b = u
                else:
                    a = u
                if fu <= fw or w == x:
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif fu < fv or v == x or v == w:
                    v = u
                    fv = fu


def function(value):
    return math.log(math.pow(value, 2)) + 1 - math.sin(value)


print(dichotomy(-2, 2, 0.00001, 0.0005, 0))
print(goldenSectionFirstStep(-2, 2, 0.00001))
print(fibonacci(-2, 2, 0.00001))
print(parabola(-2, 2, 0.00001, 0.00001))
print(brentMethod(-2, 2, 0.00001))

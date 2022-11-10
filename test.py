from micrograd.engine import Value


def add_test():
    a, b = Value(1), Value(1)

    print(f'result: {(a + b).data}')
    print(f'result: {(a + 3).data}')


def mul_test():
    a, b = Value(2), Value(2)
    print(f'result of mul = {(a * b).data}')
    print(f'result of mul = {(a * 4).data}')


def pow_test():
    a = Value(2)
    print(f'result of power op : {(a**10).data}')


def neg_test():
    a = Value(333)
    print(f'negative result = {-a.data}')


def radd_test():
    a, b = Value(44), Value(33)
    print(f'reverse addition result = {(a+b).data}')
    print(f'reverse addition result = {(b+a).data}')


def truediv_test():
    a, b = Value(10), Value(5)
    print(f'truediv result = {(a/b).data}')


if __name__ == '__main__':
    add_test()
    mul_test()
    pow_test()
    neg_test()
    radd_test()
    truediv_test()

from micrograd.engine import Value

if __name__ == '__main__':
    a = Value(1)
    b = 3
    result = Value(0)
    result = a + b
    print(result.__repr__())

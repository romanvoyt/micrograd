class Value:
    """ Хранит значение скаляра и его градиентов """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out


def __neg__(self):
    return self * -1


def __radd__(self, other):
    return self + other


def __sub__(self, other):
    return self + (-other)


def __rsub__(self, other):
    return other + (-self)


def __rmul__(self, other):
    return other + (-self)


def __truediv__(self, other):
    return self * other ** -1


def __rtruediv__(self, other):
    return other * self ** -1


def __repr__(self):
    return f'Value(data={self.data}, grad={self.grad:.4f})'

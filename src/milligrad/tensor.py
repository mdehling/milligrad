import numpy as np


# FIXME: This is a bit of a hack to implement 'reverse broadcasting' and I
# should really think about a cleaner way.

def np_broadcast_axes(x, y):
    assert x.ndim == y.ndim, "dimension mismatch"
    return tuple(i for i, (r,s) in enumerate(zip(x.shape,y.shape)) if r != s)

def np_reduce_like(x, y):
    return x.sum(axis=np_broadcast_axes(x, y), keepdims=True)


class Tensor:

    def __init__(self, value, /, requires_grad=None, *,
            _label='', _children=()):

        self._dtype = np.float32

        if isinstance(value, np.ndarray):
            self.value = value
        else:
            self.value = np.array(value, dtype=self._dtype)

        if requires_grad is None:
            requires_grad = any(child.requires_grad for child in _children)
        self.requires_grad = requires_grad

        if requires_grad:
            self.grad = np.zeros_like(self.value)

        self._label = _label
        self._children = set(_children)
        self._backward = lambda: None

    def _zero_grad(self):
        assert self.requires_grad
        self.grad.fill(0)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, _label='const')

        value = np.add(self.value, other.value)
        result = Tensor(value, _label='+', _children=(self, other))

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += np_reduce_like(result.grad, self.grad)
                if other.requires_grad:
                    other.grad += np_reduce_like(result.grad, other.grad)
            result._backward = _backward

        return result

    def __radd__(self, other):
        other = Tensor(other, requires_grad=False, _label='const')
        return other + self

    def __neg__(self):
        value = np.negative(self.value)
        result = Tensor(value, _label='-', _children=(self,))

        if result.requires_grad:
            def _backward():
                self.grad -= result.grad
            result._backward = _backward

        return result

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, _label='const')

        value = np.subtract(self.value, other.value)
        result = Tensor(value, _label='-', _children=(self, other))

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += np_reduce_like(result.grad, self.grad)
                if other.requires_grad:
                    other.grad -= np_reduce_like(result.grad, other.grad)
            result._backward = _backward

        return result

    def __rsub__(self, other):
        other = Tensor(other, requires_grad=False, _label='const')
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, _label='const')

        value = np.multiply(self.value, other.value)
        result = Tensor(value, _label='*', _children=(self, other))

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += np_reduce_like(
                        result.grad * other.value, self.grad
                    )
                if other.requires_grad:
                    other.grad += np_reduce_like(
                        self.value * result.grad, other.grad
                    )
            result._backward = _backward

        return result

    def __rmul__(self, other):
        other = Tensor(other, requires_grad=False, _label='const')
        return other * self

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, _label='const')

        value = np.divide(self.value, other.value)
        result = Tensor(value, _label='/', _children=(self, other))

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += np_reduce_like(
                        result.grad / other.value, self.grad
                    )
                if other.requires_grad:
                    other.grad += np_reduce_like(
                        - self.value / pow(result.grad, 2), other.grad
                    )
            result._backward = _backward

        return result

    def __rtruediv__(self, other):
        other = Tensor(other, requires_grad=False, _label='const')
        return other / self

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False, _label='const')

        # for ndim < 2 numpy matmul adds dimensions which I don't feel like
        # supporting in _backward() below.
        assert self.value.ndim >=2 and self.value.ndim >= 2, \
                'matmul only supports ndim >= 2'

        value = np.matmul(self.value, other.value)
        result = Tensor(value, _label='@', _children=(self, other))

        if result.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += result.grad @ other.value.swapaxes(-2,-1)
                if other.requires_grad:
                    other.grad += self.value.swapaxes(-2,-1) @ result.grad
            result._backward = _backward

        return result

    def __rmatmul__(self, other):
        other = Tensor(other, requires_grad=False, _label='const')
        return other @ self

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("__pow__ only defined for int or float exponents")

        value = np.power(self.value, other)
        result = Tensor(value, _label=f'**{other}', _children=(self, ))

        if result.requires_grad:
            def _backward():
                self.grad += (other * self.value ** (other-1)) * result.grad
            result._backward = _backward

        return result

    def exp(self):
        value = np.exp(self.value)
        result = Tensor(value, _label='exp', _children=(self,))

        if result.requires_grad:
            def _backward():
                self.grad += result.value * result.grad
            result._backward = _backward

        return result

    def log(self):
        value = np.log(self.value)
        result = Tensor(value, _label='log', _children=(self,))

        if result.requires_grad:
            def _backward():
                self.grad += result.grad / self.value
            result._backward = _backward

        return result

    def relu(self):
        value = np.maximum(self.value, 0.0)
        result = Tensor(value, _label='relu', _children=(self,))

        if result.requires_grad:
            def _backward():
                self.grad += (self.value >= 0.0) * result.grad
            result._backward = _backward

        return result

    def sigmoid(self):
        value = 1 / (1+np.exp(-self.value))
        result = Tensor(value, _label='sigmoid', _children=(self,))

        if result.requires_grad:
            def _backward():
                self.grad += result.value * (1 - result.value) * result.grad
            result._backward = _backward

        return result

    def mean(self):
        value = np.mean(self.value)
        result = Tensor(value, _label='mean', _children=(self,))

        if result.requires_grad:
            def _backward():
                self.grad += np.mean(result.grad)
            result._backward = _backward

        return result

    def backward(self):
        nodes = list()
        visited = set()

        def topo_sort(node):
            if not node.requires_grad or node in visited:
                return
            visited.add(node)
            for child in node._children:
                topo_sort(child)
            nodes.append(node)
        topo_sort(self)

        self.grad = np.ones_like(self.value, dtype=self._dtype)
        for node in reversed(nodes):
            node._backward()

    def __repr__(self):
        repr = f"Tensor(value={self.value}, requires_grad={self.requires_grad}"
        if self.requires_grad:
            repr += f", grad={self.grad}"
        repr += f", _label='{self._label}'"
        return repr + ")"


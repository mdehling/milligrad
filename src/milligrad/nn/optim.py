__all__ = ['Optimizer', 'SGD']


class Optimizer:

    def __init__(self, model):
        self.model = model

    def zero_grad(self):
        for p in self.model.parameters():
            p._zero_grad()


class SGD(Optimizer):

    def __init__(self, model, lr=1e-3):
        super().__init__(model)
        self.lr = lr

    def step(self):
        for p in self.model.parameters():
            p.value -= self.lr * p.grad

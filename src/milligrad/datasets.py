__all__ = ['make_linear', 'make_moons']

import numpy as np


def make_linear(p0, p1, /, n=100, *, noise=None):
    if not isinstance(p0, np.ndarray): p0 = np.array(p0)
    if not isinstance(p1, np.ndarray): p1 = np.array(p1)
    t = np.random.rand(n,1)
    qs = p0[np.newaxis,:] + (p1-p0)[np.newaxis,:]*t

    if noise is not None:
        qs += noise * np.random.randn(*qs.shape)

    return qs


def make_moons(n=100, *, noise=None):
    n0, n1 = n//2, n-n//2
    t0, t1 = np.pi*np.random.rand(n0,1), np.pi*np.random.rand(n1,1)
    x0, y0 = np.hstack([-0.5+np.cos(t0), -0.25+np.sin(t0)]), np.zeros((n0,1))
    x1, y1 = np.hstack([+0.5+np.cos(t1), +0.25-np.sin(t1)]), np.ones((n1,1))
    x, y = np.vstack([x0,x1]), np.vstack([y0,y1])

    if noise is not None:
        x += noise * np.random.randn(n,2)

    return x, y

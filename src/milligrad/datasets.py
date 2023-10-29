__all__ = [
    'make_linear',
    'make_moons',
    'make_blob_circle',
]

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


def make_blob_circle(k=5, n=250, std=0.2):
    ni = n//k; n0 = n - (k-1)*ni
    x = std * np.random.randn(n0,2)
    y = np.zeros((n0,k)); y[:,0] = 1
    xs, ys = [x], [y]
    for i in range(1,k):
        rad = 2*np.pi*i/(k-1)
        x = std * np.random.randn(ni,2) + [[np.cos(rad),np.sin(rad)]]
        y = np.zeros((ni,k)); y[:,i] = 1
        xs.append(x)
        ys.append(y)
    x, y = np.vstack(xs), np.vstack(ys)
    return x, y

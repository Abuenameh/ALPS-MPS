import numpy as np

def mathematica(x):
    if x == 'True' or x == 'False':
        return x
    try:
        return '{' + ','.join([mathematica(xi) for xi in iter(x)]) + '}'
    except:
        try:
            return '{:.16}'.format(x).replace('j', 'I').replace('e', '*^')
        except:
            return str(x)

speckles = {}

L = 50
seed = 1
delta = 3

def speckle(W):
    if speckles.has_key(W):
        return speckles[W]

    if delta == 0:
        speckleW = np.zeros(L)
        speckleW.fill(W)
        speckles[W] = speckleW
        return speckleW

    np.random.seed(seed)

    FFTD = 200
    FFTL = int(delta * FFTD)

    A = (4 / np.pi) * (W / FFTD)
    a = [[A * np.exp(2 * np.pi * np.random.random() * 1j) if (i * i + j * j < 0.25 * FFTD * FFTD) else 0 for i in
          range(-FFTL // 2, FFTL // 2, 1)] for j in range(-FFTL // 2, FFTL // 2, 1)]

    b = np.fft.fft2(a)
    s = np.real(b * np.conj(b))
    s2 = np.sqrt(s)
    speckleW = s2.flatten()[0:L]
    speckles[W] = speckleW
    return speckleW

res = speckle(1e12)
print mathematica(res)

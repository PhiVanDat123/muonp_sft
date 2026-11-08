"""Spectral helpers ported from muon-p/utils.py."""
import torch


def compile_if_cuda(fn):
    """Use upstream compilation on CUDA; keep CPU checks usable in eager mode."""
    if not torch.cuda.is_available():
        return fn
    return torch.compile(fn)

def _parse_exponent(exponent) -> float:
    """Parse an exponent that may be a number, a decimal string, or a rational
    string like '1/3', '3/5', '1/15', '13/15', '1/7', etc.
    """
    if isinstance(exponent, (int, float)):
        return float(exponent)
    s = str(exponent).strip()
    if "/" in s:
        num, denom = s.split("/", 1)
        return float(num) / float(denom)
    return float(s)


def zeropower_via_svd(G, steps=None, exponent="0", halfpower_c=None):
    orig_dtype = G.dtype
    U, S, V = G.float().svd()
    return (U @ V.T).to(orig_dtype)


def halfpower_via_svd(G, exponent="1/3", steps=None, halfpower_c=None):
    orig_dtype = G.dtype
    U, S, V = G.float().svd()
    p = _parse_exponent(exponent)
    return ((U * S.pow(p)) @ V.T).to(orig_dtype)


@compile_if_cuda
def scalar_zero(x):
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Scalar version of the quintic Newton-Schulz iteration for zero power.
    # Matrix form: X = a*X + b*(X@X.T)@X + c*(X@X.T)@(X@X.T)@X
    # On singular values: x_new = a*x + b*x^3 + c*x^5
    return a * x + b * x**3 + c * x**5

@compile_if_cuda
def scalar_half(x, y):
    c, d = (-3.000000,  -0.795918)  # empirical coefficients for half power
    # x, y are matrices, c, d are scalars.
    f = y - (y**4 - x**2) * (c*x+d*y)
    return f

@compile_if_cuda
def scalar_third(x, y, c=None):
    if c is None:
        c = 1.090452 # empirical coefficient for cube root
    # x, y are matrices, c, d are scalars.
    f = y - c * (y**3 - x)
    return f


@compile_if_cuda
def scalar_fifth(x, y, c=None):
    if c is None:
        c = 1.090452 # empirical coefficient for fifth root
    # x, y are matrices, c, d are scalars.
    f = y - c * (y**5 - x)
    return f


@compile_if_cuda
def scalar_seventh(x, y, c=None):
    """Scalar Newton-Schulz step for 1/7 power: f = y - c * (y^7 - x).

    Fixed point y* satisfies y*^7 = x. Theoretical Newton step at y* ~ 1
    gives c = 1/7 ≈ 0.143; convergence range is c ∈ (0, 2/7 ≈ 0.286).
    """
    if c is None:
        c = 1.0 / 7
    f = y - c * (y**7 - x)
    return f


@compile_if_cuda
def scalar_fifteenth(x, y, c=None):
    """Scalar Newton-Schulz step for 1/15 power: f = y - c * (y^15 - x).

    Fixed point y* satisfies y*^15 = x. Theoretical Newton step at y* ~ 1
    gives c = 1/15; tune empirically via grid search.
    """
    if c is None:
        c = 1.0 / 15
    f = y - c * (y**15 - x)
    return f


@compile_if_cuda
def scalar_thirteen_fifteenth(x, y, c=None):
    """Scalar Newton-Schulz step for 13/15 power: f = y - c * (y^15 - x^13).

    Fixed point y* satisfies y*^15 = x^13, i.e. y* = x^{13/15}. Theoretical
    Newton step at y* ~ 1 gives c = 1/15.
    """
    if c is None:
        c = 1.0 / 15
    f = y - c * (y**15 - x**13)
    return f


@compile_if_cuda
def poly_half(x, y, c=None, d=None):
    #c, d = (-0.2330, 0.8015)  # empirical coefficients for half power
    if c is None or d is None:
        c, d = (3.000000, -0.795918)  # empirical coefficients for half power
    # x, y are matrices, c, d are scalars.
    Y = y@y.mT
    f = y - (Y@Y - x@x.mT) @ (c*x+d*y)

    return f

@compile_if_cuda
def poly_fifth(x, y, c=None):
    # x, y are matrices, c, d are scalars.
    if c is None:
        c = 0.2 
    Y = y@y.mT
    f = y - c * (Y@Y@y - x)
    return f


@compile_if_cuda
def poly_seventh(x, y, c=None):
    """Newton-Schulz step for the 1/7 power: f = y - c * (y^7 - x).

    In matrix form, y^7 is realized as (y @ y.mT)^3 @ y, which equals
    U S^7 V^T for the SVD y = U S V^T. The fixed point therefore satisfies
    y* = x^{1/7} (in the singular-value sense).

    Default c = 1/7 (theoretical Newton step at unit singular values);
    safe range c ∈ (0, 2/7 ≈ 0.286). Tune empirically with optimize_power.py.
    """
    if c is None:
        c = 1.0 / 7
    A = y @ y.mT       # (y@y.mT)^1, acts as y^2
    A2 = A @ A         # (y@y.mT)^2, acts as y^4
    y7 = A2 @ A @ y    # (y@y.mT)^3 @ y = y^7
    f = y - c * (y7 - x)
    return f

@compile_if_cuda
def poly_three_fifth(x, y, c=None):
    # x, y are matrices, c are scalars.
    if c is None:
        c = 0.2 
    f = y - c * (y@y.mT@y@y.mT@y - x@x.mT@x)
    return f

@compile_if_cuda
def poly_third(x, y, c=None):
    # c = 0.040830 # empirical coefficient for cube root
    # x, y are matrices, c, d are scalars.
    if c is None:
        c = 0.66 
    f = y - c * (y@y.mT@y - x)
    return f


@compile_if_cuda
def poly_fifteenth(x, y, c=None):
    """Newton-Schulz step for the 1/15 power: f = y - c * (y^15 - x).

    In matrix form, y^15 is realized as (y @ y.mT)^7 @ y, which equals
    U S^15 V^T for the SVD y = U S V^T. The fixed point therefore satisfies
    y* = x^{1/15} (in the singular-value sense).

    Default c = 1/15 (theoretical Newton step at unit singular values);
    tune empirically with optimize_power.py for best convergence.
    """
    if c is None:
        c = 1.0 / 15
    A = y @ y.mT          # (y@y.mT)^1, acts as y^2
    A2 = A @ A            # (y@y.mT)^2, acts as y^4
    A4 = A2 @ A2          # (y@y.mT)^4, acts as y^8
    y15 = A4 @ A2 @ A @ y # (y@y.mT)^7 @ y = y^15
    f = y - c * (y15 - x)
    return f


@compile_if_cuda
def poly_thirteen_fifteenth(x, y, c=None):
    """Newton-Schulz step for the 13/15 power: f = y - c * (y^15 - x^13).

    Matrix forms (acting on singular values via U S^k V^T):
        y^15 = (y @ y.mT)^7 @ y
        x^13 = (x @ x.mT)^6 @ x

    Fixed point y* satisfies y*^15 = x^13, i.e. y* = x^{13/15}.
    Default c = 1/15; tune empirically with optimize_power.py.
    """
    if c is None:
        c = 1.0 / 15
    # y^15 = (y@y.mT)^7 @ y   (6 matmuls)
    A = y @ y.mT
    A2 = A @ A
    A4 = A2 @ A2
    y15 = A4 @ A2 @ A @ y
    # x^13 = (x@x.mT)^6 @ x   (5 matmuls)
    B = x @ x.mT
    B2 = B @ B
    B4 = B2 @ B2
    x13 = B4 @ B2 @ x
    f = y - c * (y15 - x13)
    return f



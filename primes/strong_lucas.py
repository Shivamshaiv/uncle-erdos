import sys
sys.path.insert(0, "../")

from afunctions.gcd_func import computeGCD
from afunctions.symbols import jacobi



def selfridge(n):
    d, s = 5, 1
    while True:
        ds = d * s
        if computeGCD(ds, n) > 1:
            return ds, 0, 0
        if jacobi(ds, n) == -1:
            return ds, 1, (1 - ds) / 4
        d, s = d + 2, s * -1

def lucasPQ(p, q, m, n):
    # nth element of lucas sequence with
    # parameters p and q (mod m); ignore
    # modulus operation when m is zero
    def mod(x):
        if m == 0: return x
        return x % m
    def half(x):
        if x % 2 == 1: x = x + m
        return mod(x / 2)
    un, vn, qn = 1, p, q
    u = 0 if n % 2 == 0 else 1
    v = 2 if n % 2 == 0 else p
    k = 1 if n % 2 == 0 else q
    n, d = n // 2, p * p - 4 * q
    while n > 0:
        u2 = mod(un * vn)
        v2 = mod(vn * vn - 2 * qn)
        q2 = mod(qn * qn)
        n2 = n // 2
        if n % 2 == 1:
            uu = half(u * v2 + u2 * v)
            vv = half(v * v2 + d * u * u2)
            u, v, k = uu, vv, k * q2
        un, vn, qn, n = u2, v2, q2, n2
    return u, v, k

def is_prime(n):
    '''An implementation of the Strong Lucas Primality Testing'''
    d, p, q = selfridge(n)
    if p == 0: return n == d
    u, v, k = lucasPQ(p, q, n, n+1)
    return u == 0

#print(is_prime(2305843009213693951))
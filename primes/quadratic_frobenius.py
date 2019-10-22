import sys
from random import SystemRandom
random = SystemRandom().randrange

sys.path.insert(0, "../")

from afunctions.gcd_func import computeGCD
from afunctions.gcd_func import extended_gcd
from afunctions.symbols import jacobi



def is_prime(num):
    '''Checks a primality of numbers with the Quadratic Frobenius test, the implementation
    can be proved to false detection of compostite number with 77/7710 = 0.00998 probablity.'''
    for _ in range(77):
        if frobenius_pseudoprime(num):
            return True
    return False



def frobenius_pseudoprime(integer):
    assert integer & 1 and integer >= 3
    a, b, d = choose_ab(integer)
    w1 = (a ** 2 * extended_gcd(b, integer)[0] - 2) % integer
    m = (integer - jacobi_symbol(d, integer)) >> 1
    wm, wm1 = compute_wm_wm1(w1, m, integer)
    if w1 * wm != 2 * wm1 % integer:
        return False
    b = pow(b, (integer - 1) >> 1, integer)
    return b * wm % integer == 2

def choose_ab(integer):
    random = SystemRandom().randrange
    a, b = random(1, integer), random(1, integer)
    d = a ** 2 - 4 * b
    while is_square(d) or computeGCD(2 * d * a * b, integer) != 1:
        a, b = random(1, integer), random(1, integer)
        d = a ** 2 - 4 * b
    return a, b, d

def is_square(integer):
    if integer < 0:
        return False
    if integer < 2:
        return True
    x = integer >> 1
    seen = set([x])
    while x * x != integer:
        x = (x + integer // x) >> 1
        if x in seen:
            return False
        seen.add(x)
    return True

def extended_gcd(n, d):
    x1, x2, y1, y2 = 0, 1, 1, 0
    while d:
        n, (q, d) = d, divmod(n, d)
        x1, x2, y1, y2 = x2 - q * x1, x1, y2 - q * y1, y1
    return x2, y2

def jacobi_symbol(n, d):
    return jacobi(n,d)

def compute_wm_wm1(w1, m, n):
    a, b = 2, w1
    for shift in range(m.bit_length() - 1, -1, -1):
        if m >> shift & 1:
            a, b = (a * b - w1) % n, (b * b - 2) % n
        else:
            a, b = (a * a - 2) % n, (a * b - w1) % n
    return a, b


print(is_prime(2305843009213693951))
print(help(is_prime))
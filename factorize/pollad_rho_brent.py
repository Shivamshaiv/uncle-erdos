from random import randrange
import sys
sys.path.insert(0, "../")
from afunctions.gcd_func import gcd
from primes.miller_rabin import is_prime as isprime


def pollardRho_brent(n):
    
    if isprime(n):
        return n
    g = n
    while g == n:
        y, c, m = randrange(1, n), randrange(1, n), randrange(1, n)
        g, r, q = 1, 1, 1
        while g == 1:
            x, k = y, 0
            for _ in range(r):
                y = (y**2 + c) % n
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r-k)):
                    y = (y**2 + c) % n
                    q = q * abs(x-y) % n
                g, k = gcd(q, n), k+m
            r *= 2
        if g == n:
            while True:
                ys = (ys**2+c) % n
                g = gcd(abs(x-ys), n)
                if g > 1:
                    break
    return g


print(pollardRho_brent(459245788321))
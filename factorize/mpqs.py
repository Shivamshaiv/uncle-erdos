from functools import reduce

try:
    import gmpy2 as gmpy
    gmpy_version = 2
    mpz = gmpy.mpz
except ImportError:
    try:
        import gmpy
        gmpy_version = 1
        mpz = gmpy.mpz
    except ImportError:
        gmpy_version = 0
        mpz = int
        gmpy = None


def listprod(a):
    return reduce(lambda x, y: x * y, a, 1)

__all__ = [listprod, gmpy, gmpy_version, mpz]

import six
import itertools
# Recursive sieve of Eratosthenes
def primegen():
    yield 2
    yield 3
    yield 5
    yield 7
    yield 11
    yield 13
    ps = primegen()  # yay recursion
    p = six.next(ps) and six.next(ps)
    q, sieve, n = p**2, {}, 13
    while True:
        if n not in sieve:
            if n < q:
                yield n
            else:
                next_, step = q + 2*p, 2*p
                while next_ in sieve:
                    next_ += step
                sieve[next_] = step
                p = six.next(ps)
                q = p**2
        else:
            step = sieve.pop(n)
            next_ = n + step
            while next_ in sieve:
                next_ += step
            sieve[next_] = step
        n += 2


def primes(n):
    # The primes STRICTLY LESS than n
    return list(itertools.takewhile(lambda p: p < n, primegen()))


def nextprime(n):
    if n < 2:
        return 2
    if n == 2:
        return 3
    n = (n + 1) | 1    # first odd larger than n
    m = n % 6
    if m == 3:
        if isprime(n+2):
            return n+2
        n += 4
    elif m == 5:
        if isprime(n):
            return n
        n += 2
    for m in itertools.count(n, 6):
        if isprime(m):
            return m
        if isprime(m+4):
            return m+4


def pfactor(n):
    s, d, q = 0, n-1, 2
    while not d & q - 1:
        s, q = s+1, q*2
    return s, d // (q // 2)


def _sprp(n, a, s=None, d=None):
    if n % 2 == 0:
        return False
    if (s is None) or (d is None):
        s, d = pfactor(n)
    x = pow(a, d, n)
    if x == 1:
        return True
    for _ in range(s):
        if x == n - 1:
            return True
        x = pow(x, 2, n)
    return False


def _sprp_gmpy2(n, a, s=None, d=None):
    return _util.gmpy.is_strong_prp(n, a)


# Used in SLPRP.  TODO: figure out what this does.
def chain(n, u1, v1, u2, v2, d, q, m):
    k = q
    while m > 0:
        u2, v2, q = (u2*v2) % n, (v2*v2 - 2*q) % n, (q*q) % n
        if m % 2 == 1:
            u1, v1 = u2*v1+u1*v2, v2*v1+u2*u1*d
            if u1 % 2 == 1:
                u1 = u1 + n
            if v1 % 2 == 1:
                v1 = v1 + n
            u1, v1, k = (u1//2) % n, (v1//2) % n, (q*k) % n
        m //= 2
    return u1, v1, k


def _isprime(n, tb=(3, 5, 7, 11), eb=(2,), mrb=()):  # TODO: more streamlining
    # tb: trial division basis
    # eb: Euler's test basis
    # mrb: Miller-Rabin basis

    # This test suite's first false positve is unknown but has been shown to
    # be greater than 2**64.
    # Infinitely many are thought to exist.

    if n % 2 == 0 or n < 13 or n == isqrt(n)**2:
        # Remove evens, squares, and numbers less than 13
        return n in (2, 3, 5, 7, 11)
    if any(n % p == 0 for p in tb):
        return n in tb  # Trial division

    for b in eb:  # Euler's test
        if b >= n:
            continue
        if not pow(b, n-1, n) == 1:
            return False
        r = n - 1
        while r % 2 == 0:
            r //= 2
        c = pow(b, r, n)
        if c == 1:
            continue
        while c != 1 and c != n-1:
            c = pow(c, 2, n)
        if c == 1:
            return False

    s, d = pfactor(n)
    if not sprp(n, 2, s, d):
        return False
    if n < 2047:
        return True
    # BPSW has two phases: SPRP with base 2 and SLPRP.
    # We just did the SPRP; now we do the SLPRP:
    if n >= 3825123056546413051:
        d = 5
        while True:
            if gcd(d, n) > 1:
                p, q = 0, 0
                break
            if jacobi(d, n) == -1:
                p, q = 1, (1 - d) // 4
                break
            d = -d - 2*d//abs(d)
        if p == 0:
            return n == d
        s, t = pfactor(n + 2)
        u, v, u2, v2, m = 1, p, 1, p, t//2
        k = q
        while m > 0:
            u2, v2, q = (u2*v2) % n, (v2*v2-2*q) % n, (q*q) % n
            if m % 2 == 1:
                u, v = u2*v+u*v2, v2*v+u2*u*d
                if u % 2 == 1:
                    u += n
                if v % 2 == 1:
                    v += n
                u, v, k = (u//2) % n, (v//2) % n, (q*k) % n
            m //= 2
        if (u == 0) or (v == 0):
            return True
        for _ in range(1, s):
            v, k = (v*v-2*k) % n, (k*k) % n
            if v == 0:
                return True
        return False

    if not mrb:
        if n < 1373653:
            mrb = [3]
        elif n < 25326001:
            mrb = [3, 5]
        elif n < 3215031751:
            mrb = [3, 5, 7]
        elif n < 2152302898747:
            mrb = [3, 5, 7, 11]
        elif n < 3474749660383:
            mrb = [3, 5, 6, 11, 13]
        elif n < 341550071728321:
            # This number is also a false positive for primes(19+1).
            mrb = [3, 5, 7, 11, 13, 17]
        elif n < 3825123056546413051:
            # Also a false positive for primes(31+1).
            mrb = [3, 5, 7, 11, 13, 17, 19, 23]
    # Miller-Rabin
    return all(sprp(n, b, s, d) for b in mrb)

if gmpy_version == 2:
    sprp = _sprp_gmpy2
    isprime = _util.gmpy.is_bpsw_prp
else:
    sprp = _sprp
    isprime = _isprime


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _isqrt(n):
    if n == 0:
        return 0
    x, y = n, (n + 1) // 2
    while y < x:
        x, y = y, (y + n//y) // 2
    return x


def _introot(n, r=2):
    if n < 0:
        return None if r % 2 == 0 else -introot(-n, r)
    if n < 2:
        return n
    if r == 2:
        return isqrt(n)
    lower, upper = 0, n
    while lower != upper - 1:
        mid = (lower + upper) // 2
        m = mid**r
        if m == n:
            return mid
        elif m < n:
            lower = mid
        elif m > n:
            upper = mid
    return lower


def _introot_gmpy(n, r=2):
    if n < 0:
        return None if r % 2 == 0 else -introot(-n, r)
    return gmpy.root(n, r)[0]


def _introot_gmpy2(n, r=2):
    if n < 0:
        return None if r % 2 == 0 else -introot(-n, r)
    return gmpy.iroot(n, r)[0]


def _jacobi(a, p):
    if (p % 2 == 0) or (p < 0):
        return None  # p must be a positive odd number
    if (a == 0) or (a == 1):
        return a
    a, t = a % p, 1
    while a != 0:
        while not a & 1:
            a //= 2
            if p & 7 in (3, 5):
                t *= -1
        a, p = p, a
        if (a & 3 == 3) and (p & 3) == 3:
            t *= -1
        a %= p
    return t if p == 1 else 0


# greatest integer l such that b**l <= x.
def ilog(x, b):
    l = 0
    while x >= b:
        x //= b
        l += 1
    return l


# Returns the largest integer that, when squared/cubed/etc, yields n, or 0
# if no such integer exists.
# Note that the power to which this number is raised will be prime.
def ispower(n):
    for p in primegen():
        r = introot(n, p)
        if r is None:
            continue
        if r ** p == n:
            return r
        if r == 1:
            return 0


# legendre symbol (a|m)
# TODO: which is faster?
def _legendre1(a, p):
    return ((pow(a, (p-1) >> 1, p) + 1) % p) - 1


# TODO: pretty sure this computes the Jacobi symbol
def _legendre2(a, p):
    if a == 0:
        return 0
    x, y, L = a, p, 1
    while 1:
        if x > (y >> 1):
            x = y - x
            if y & 3 == 3:
                L = -L
        while x & 3 == 0:
            x >>= 2
        if x & 1 == 0:
            x >>= 1
            if y & 7 == 3 or y & 7 == 5:
                L = -L
        if x == 1:
            return ((L+1) % p) - 1
        if x & 3 == 3 and y & 3 == 3:
            L = -L
        x, y = y % x, x


def _legendre_gmpy(n, p):
    if (n > 0) and (p % 2 == 1):
        return _util.gmpy.legendre(n, p)
    else:
        return _legendre1(n, p)


# modular sqrt(n) mod p
# p must be prime
def mod_sqrt(n, p):
    a = n % p
    if p % 4 == 3:
        return pow(a, (p+1) >> 2, p)
    elif p % 8 == 5:
        v = pow(a << 1, (p-5) >> 3, p)
        i = ((a*v*v << 1) % p) - 1
        return (a*v*i) % p
    elif p % 8 == 1:  # Shank's method
        q, e = p-1, 0
        while q & 1 == 0:
            e += 1
            q >>= 1
        n = 2
        while legendre(n, p) != -1:
            n += 1
        w, x, y, r = pow(a, q, p), pow(a, (q+1) >> 1, p), pow(n, q, p), e
        while True:
            if w == 1:
                return x
            v, k = w, 0
            while v != 1 and k+1 < r:
                v = (v*v) % p
                k += 1
            if k == 0:
                return x
            d = pow(y, 1 << (r-k-1), p)
            x, y = (x*d) % p, (d*d) % p
            w, r = (w*y) % p, k
    else:
        return a  # p == 2


# modular inverse of a mod m
def _modinv(a, m):
    a, x, u = a % m, 0, 1
    while a:
        x, u, m, a = u, x - (m//a)*u, a, m % a
    return x


def _modinv_gmpy(a, m):
    return int(_util.gmpy.invert(a, m))


if gmpy_version > 0:
    gcd = gmpy.gcd
    jacobi = _util.gmpy.jacobi
    legendre = _legendre_gmpy
    modinv = _modinv_gmpy
    if gmpy_version == 2:
        isqrt = gmpy.isqrt
        introot = _introot_gmpy2
    else:
        isqrt = gmpy.sqrt
        introot = _introot_gmpy
else:
    gcd = _gcd
    isqrt = _isqrt
    introot = _introot
    jacobi = _jacobi
    legendre = _legendre1
    modinv = _modinv



def factor(n):
    """
    Prime factorisation using the Multiple Polynomial Quadratic Sieve.
    There is a lot to optimize and improve this implementation but it factorisation is extremly fast
    The running time complexity is L[1/2,1] in L notation which translates to O(e^((1+o(1))*sqrt(logn)*sqrt(log log n)))
    """

    from six.moves import xrange
    from math import log

    # Special cases: this function poorly handles primes and perfect powers:
    m = ispower(n)
    if m:
        return m
    if isprime(n):
        return n

    root_2n = isqrt(2*n)
    bound = ilog(n**6, 10)**2  # formula chosen by experiment

    while True:
        try:
            prime, mod_root, log_p, num_prime = [], [], [], 0

            # find a number of small primes for which n is a quadratic residue
            p = 2
            while p < bound or num_prime < 3:
                leg = legendre(n % p, p)
                if leg == 1:
                    prime += [p]
                    # the rhs was [int(mod_sqrt(n, p))].
                    # If we get errors, put it back.
                    mod_root += [mod_sqrt(n, p)]
                    log_p += [log(p, 10)]
                    num_prime += 1
                elif leg == 0:
                    return p
                p = nextprime(p)

            x_max = len(prime)*60  # size of the sieve

            # maximum value on the sieved range
            m_val = (x_max * root_2n) >> 1

            """
            fudging the threshold down a bit makes it easier to find powers of
            primes as factors as well as partial-partial relationships, but it
            also makes the smoothness check slower. there's a happy medium
            somewhere, depending on how efficient the smoothness check is
            """
            thresh = log(m_val, 10) * 0.735

            # skip small primes. they contribute very little to the log sum
            # and add a lot of unnecessary entries to the table instead, fudge
            # the threshold down a bit, assuming ~1/4 of them pass
            min_prime = mpz(thresh * 3)
            fudge = sum(log_p[i] for i, p in enumerate(prime) if p < min_prime)
            fudge = fudge // 4
            thresh -= fudge

            smooth, used_prime, partial = [], set(), {}
            num_smooth, num_used_prime, num_partial = 0, 0, 0
            num_poly, root_A = 0, isqrt(root_2n // x_max)

            while num_smooth <= num_used_prime:
                # find an integer value A such that:
                # A is =~ sqrt(2*n) // x_max
                # A is a perfect square
                # sqrt(A) is prime, and n is a quadratic residue mod sqrt(A)
                while True:
                    root_A = nextprime(root_A)
                    leg = legendre(n, root_A)
                    if leg == 1:
                        break
                    elif leg == 0:
                        return root_A
                A = root_A**2
                # solve for an adequate B. B*B is a quadratic residue mod n,
                # such that B*B-A*C = n. this is unsolvable if n is not a
                # quadratic residue mod sqrt(A)
                b = mod_sqrt(n, root_A)
                B = (b + (n - b*b) * modinv(b + b, root_A)) % A
                C = (B*B - n) // A        # B*B-A*C = n <=> C = (B*B-n)//A
                num_poly += 1
                # sieve for prime factors
                sums, i = [0.0]*(2*x_max), 0
                for p in prime:
                    if p < min_prime:
                        i += 1
                        continue
                    logp = log_p[i]
                    g = gcd(A, p)
                    if g == p:
                        continue
                    inv_A = modinv(A // g, p // g) * g
                    # modular root of the quadratic
                    a, b, k = (mpz(((mod_root[i] - B) * inv_A) % p),
                               mpz(((p - mod_root[i] - B) * inv_A) % p),
                               0)
                    while k < x_max:
                        if k+a < x_max:
                            sums[k+a] += logp
                        if k+b < x_max:
                            sums[k+b] += logp
                        if k:
                            sums[k-a+x_max] += logp
                            sums[k-b+x_max] += logp
                        k += p
                    i += 1
                # check for smooths
                i = 0
                for v in sums:
                    if v > thresh:
                        x, vec, sqr = x_max-i if i > x_max else i, set(), []
                        # because B*B-n = A*C
                        # (A*x+B)^2 - n = A*A*x*x+2*A*B*x + B*B - n
                        #               = A*(A*x*x+2*B*x+C)
                        # gives the congruency
                        # (A*x+B)^2 = A*(A*x*x+2*B*x+C) (mod n)
                        # because A is chosen to be square, it doesn't
                        # need to be sieved
                        sieve_val = (A*x + 2*B)*x + C
                        if sieve_val < 0:
                            vec, sieve_val = {-1}, -sieve_val
                        for p in prime:
                            while sieve_val % p == 0:
                                if p in vec:
                                    """
                                    track perfect sqr facs to avoid sqrting
                                    something huge at the end
                                    """
                                    sqr += [p]
                                vec ^= {p}
                                sieve_val = mpz(sieve_val // p)
                        if sieve_val == 1:  # smooth
                            smooth += [(vec, (sqr, (A*x+B), root_A))]
                            used_prime |= vec
                        elif sieve_val in partial:
                            """
                            combine two partials to make a (xor) smooth that
                            is, every prime factor with an odd power is in our
                            factor base
                            """
                            pair_vec, pair_vals = partial[sieve_val]
                            sqr += list(vec & pair_vec) + [sieve_val]
                            vec ^= pair_vec
                            smooth += [(vec, (sqr + pair_vals[0],
                                        (A*x+B)*pair_vals[1],
                                         root_A*pair_vals[2]))]
                            used_prime |= vec
                            num_partial += 1
                        else:
                            # save partial for later pairing
                            partial[sieve_val] = (vec, (sqr, A*x+B, root_A))
                    i += 1
                num_smooth, num_used_prime = len(smooth), len(used_prime)
            used_prime = sorted(list(used_prime))
            # set up bit fields for gaussian elimination
            masks, mask, bitfields = [], 1, [0]*num_used_prime
            for vec, _ in smooth:
                masks += [mask]
                i = 0
                for p in used_prime:
                    if p in vec:
                        bitfields[i] |= mask
                    i += 1
                mask <<= 1
            # row echelon form
            offset = 0
            null_cols = []
            for col in xrange(num_smooth):
                # This occasionally throws IndexErrors.
                pivot = bitfields[col-offset] & masks[col] == 0
                # TODO: figure out why it throws errors and fix it.
                for row in xrange(col+1-offset, num_used_prime):
                    if bitfields[row] & masks[col]:
                        if pivot:
                            bitfields[col-offset], bitfields[row] = \
                              bitfields[row], bitfields[col-offset]
                            pivot = False
                        else:
                            bitfields[row] ^= bitfields[col-offset]
                if pivot:
                    null_cols += [col]
                    offset += 1
            # reduced row echelon form
            for row in xrange(num_used_prime):
                mask = bitfields[row] & -bitfields[row]        # lowest set bit
                for up_row in xrange(row):
                    if bitfields[up_row] & mask:
                        bitfields[up_row] ^= bitfields[row]
            # check for non-trivial congruencies
            # TODO: if none exist, check combinations of null space columns...
            # if _still_ none exist, sieve more values
            for col in null_cols:
                all_vec, (lh, rh, rA) = smooth[col]
                lhs = lh   # sieved values (left hand side)
                rhs = [rh]  # sieved values - n (right hand side)
                rAs = [rA]  # root_As (cofactor of lhs)
                i = 0
                for field in bitfields:
                    if field & masks[col]:
                        vec, (lh, rh, rA) = smooth[i]
                        lhs += list(all_vec & vec) + lh
                        all_vec ^= vec
                        rhs += [rh]
                        rAs += [rA]
                    i += 1
                factor = gcd(listprod(rAs)*listprod(lhs) - listprod(rhs), n)
                if 1 < factor < n:
                    return factor
        except IndexError:
            pass
        bound *= 1.2




def _find_prime_near(order):
  num1 = (1<<order )+ 1
  while not _isprime(num1):
    num1 = num1 + 2
  return num1

def find_semiprimes_of_order(order):
  num1 = _find_prime_near(order)
  num2 = _find_prime_near(order+1)
  return num1*num2

p = find_semiprimes_of_order(70)
print(p)
print(factor(p))
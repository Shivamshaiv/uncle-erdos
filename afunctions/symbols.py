def jacobi(n,k):
  '''Computes the Jacobi Symbol (n,k),
  The complexity is  O( (log n)(log k) )'''
  assert(k>0 and k%2 == 1)
  t = 1
  while n != 0:
    while n%2==0:
      n = n/2
      r = k % 8
      if r==3 or r==5:
        t = -t
    n,k=k,n
    if n%4==k%4==3:
      t = -t
    n = n%k
  if k == 1:
    return t
  else:
    return 0

def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
    Euler's criterion. p is a prime, a is
    relatively prime to p (if p divides
    a, then a|p = 0)
    Returns 1 if a has a square root modulo
    p, -1 otherwise.
    """
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls
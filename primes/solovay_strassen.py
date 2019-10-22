import sys
import random
sys.path.insert(0, "../")

from afunctions.symbols import jacobi

def is_prime(n,trials=3):
  '''Checks wether a number is prime by Solovay–Strassen primality test. The trial argument is 
     used to define the number iterations used to determine primality and
     Solovay–Strassen primality test declares n probably prime with a probability at most 1/2^(trails)'''
  for k in range(trials):
    a = random.randint(2,n-1)
    x = jacobi(a,n)
    md = pow(a,(n-1)//2,n)
    if md == n-1:
      md = -1
    if md==0 or md != x:
      return False
  return True

print(is_prime(34534537139))
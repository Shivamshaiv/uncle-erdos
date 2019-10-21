def s_gcd(a, b) : 
  '''Stein's algorithm for binay GCD'''
  if (a == 0) : 
      return b 

  if (b == 0) : 
      return a 

  k = 0

  while(((a | b) & 1) == 0) : 
      a = a >> 1
      b = b >> 1
      k = k + 1

  # Dividing a by 2 until a becomes odd  
  while ((a & 1) == 0) : 
      a = a >> 1

  # From here on, 'a' is always odd.  
  while(b != 0) : 

      # If b is even, remove all  
      # factor of 2 in b  
      while ((b & 1) == 0) : 
          b = b >> 1

      # Now a and b are both odd. Swap if 
      # necessary so a <= b, then set  
      # b = b - a (which is even). 
      if (a > b) : 

          # Swap u and v. 
          temp = a 
          a = b 
          b = temp 

      b = (b - a) 

  # restore common factors of 2  
  return (a << k) 


def computeGCD(x, y): 
  '''Euclidian Algorithm for computing GCD'''
  while(y): 
    x, y = y, x % y 

  return x 


def gcd(x,y):
  '''Euclidian Algorithm for computing GCD'''
  return computeGCD(x, y)

def extended_gcd(n, d):
  '''Extended Euclidian Algorithm for computing (a,b) such that for given n,d : an+bd = gcd(n,d)'''
  x1, x2, y1, y2 = 0, 1, 1, 0
  while d:
      n, (q, d) = d, divmod(n, d)
      x1, x2, y1, y2 = x2 - q * x1, x1, y2 - q * y1, y1
  return x2, y2
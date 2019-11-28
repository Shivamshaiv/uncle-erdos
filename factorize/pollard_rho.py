import sys
sys.path.insert(0, "../")

from afunctions.gcd_func import computeGCD


def euler_phi(n):   # O(sqrt(n)) most efficent known
  result = n
  i = 2
  while(i*i<=n):
    if n%i == 0:
      while(n%i==0):
        n/=i
      result -= result/i
    i = i + 1
  if n > 1:
    result -= result/n
  return int(result)




def f(x):
	""" function for pollard's rho """
	return x**2 + 1

def factor(n,b):
	""" Factor using Pollard's p-1 method """

	a = 2;
	for j in range(2,b):
		a = a**j % n
	
	d = computeGCD(a-1,n);
	print ("d:",d,"a-1:",a-1)
	if 1 < d < n: return d;
	else: return -1;

def factorRho(n,x_1):
	""" Factor using pollard's rho method """
	
	x = x_1;
	xp = f(x) % n
	p = computeGCD(x - xp,n)

	#print ("x_i's: {")
	while p == 1:
		#print (x),
		# in the ith iteration x = x_i and x' = x_2i
		x = f(x) % n
		xp = f(xp) % n
		xp = f(xp) % n
		p = computeGCD(x-xp,n)

	#print ("}")

	if p == n: return -1
	else: return p

def testFactor():
	
	print("Pollard's p-1 factoring")
	
	n = 13493
	s = 2
	d = -1

	print ("n=%i, initial bound=%i" % (n,s))

	while s < n and d == -1:
		s +=1
		d = factor(n,s)
		print("Round %i = %i" % (s,d))

	if d == -1: print ("No Factor could be found ...")
	else: print ("%i has a factor of %i, with b=%i" % (n,d,s))

def factor(num,x_1 = 2):
	'''Returns a factor of a natural number with the Poland Rho Factoring Method
	An optional argument x_1 is for the starting value of algorithmic iteration'''
	p = factorRho(num,x_1)
	return p

def testFactorRho():

	print ("Pollard's Rho factoring" )
	n = 2305843009213693957   
	x_1 = 2

	#print ("n= %i, x_1= %i" % (n,x_1))
	
	p = factorRho(n,x_1)
	print ("p=",p)

print(factor(2305843009213693957))
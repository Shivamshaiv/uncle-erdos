from hashutil import encode
from certificate import Compute_Certificate
from firebase import firebase
firebase = firebase.FirebaseApplication('https://uncle-erdos.firebaseio.com/', None)
import time

def lucas_lehmer_test(p,start,end,sid = 'solver1'):
	'''
	Outputs if 2**p - 1 is prime?
	'''
	start = 0
	end = p-2 

	startt = time.time()
	algoid = 'llt1'
	startvar = start
	endvar = end
	pid = 'prob1'
	cachearr1 = []
	cachearr2 = []

	proof_c = ''
	s = 4
	M = 2**p - 1


	for _ in range(start,end):
		s = pow(s*s - 2,1,M)
		proof_c += str(s)

	proof_cert = encode(proof_c)

	c = Compute_Certificate(algoid,
		startvar,
		endvar,
		pid,
		sid,
		proof_cert,
		cachearr1,cachearr2,
		startt)

	print(vars(c))
	solve_cert = vars(c)
	firebase.post('/problemchain/prob1', solve_cert)

	if  s == 0:
		return True
	else:
		return False





for num in range(5,99,2):
	print(num,lucas_lehmer_test(num,0,0))


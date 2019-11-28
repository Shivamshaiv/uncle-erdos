import time 
from hashutil import encode

class Compute_Certificate:
	def __init__(self,
		algoid,
		startvar,
		endvar,
		pid,
		sid,
		proof_cert,
		cachearr1,cachearr2,
		startt):

		self.endt = time.time()

		self.algoid = algoid
		self.startvar = startvar
		self.endvar = endvar
		self.pid = pid
		self.sid = sid
		self.proof_cert = proof_cert
		self.cachearr1 = cachearr1
		self.cachearr2 = cachearr2
		self.startt = startt

		temp = ''
		temp += str(self.endt)
		temp += str(self.algoid) 
		temp += str(self.startvar)
		temp += str(self.endvar)
		temp += str(self.pid)
		temp += str(self.sid)
		temp += str(self.proof_cert) 
		temp += str(self.cachearr1)
		temp += str(self.cachearr2)
		temp += str(self.startt)

		self.shash = encode(temp)



import hashlib
from itertools import cycle
import base64


def xor_crypt_string(data, key = '1234', encode = False, decode = False):

   
   if decode:
      data = base64.decodestring(data)
   xored = ''.join(chr(ord(x) ^ ord(y)) for (x,y) in zip(data, cycle(key)))
   
   if encode:
      return base64.encodestring(xored).strip()
   return xored

def encode(s):
	key = 'zeu1okOaBzxlwjoDHo28l-Sk4JUnkWFmN6wjzizD_sM='
	#f = Fernet(key)
	#s = bytes(s,encoding = 'utf8')
	token = xor_crypt_string(s)
	token = bytes(token,encoding = 'utf8')
	return hashlib.sha256(token).hexdigest()
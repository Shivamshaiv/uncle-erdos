import datetime
from firebase import firebase
firebase = firebase.FirebaseApplication('https://uncle-erdos.firebaseio.com/', None)
import time


#data = {'name': 'Ozgur Vatansever', 'age': 26,'created_at': datetime.datetime.now()}
#data2 = {'name': 'Shivam Rohan','proof':'fdgfdbfdbdfbdf','created_at': datetime.datetime.now()}
#snapshot = firebase.post('/prob1', data2)
#snapshot2 = firebase.post('/users', data)

while True:
    res = firebase.get('/prob1', None,params={'print': 'pretty'},headers={'X_FANCY_HEADER': 'very fancy'})
    print(res)
    time.sleep(2)

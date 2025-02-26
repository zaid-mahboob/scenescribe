import firebase_admin
from firebase_admin import db

cred_obj = firebase_admin.credentials.Certificate('/home/scenescribe/Desktop/scenescribe/credentials.json')
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL':'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
    })

while(1):
    ref = db.reference("/intValue").get()
    print(ref)
# ref.get()
# while(1):
#     print(f"Value of button is: {db.reference('/intValue').get()}")

# ref = db.reference("/")
# print(ref.get())

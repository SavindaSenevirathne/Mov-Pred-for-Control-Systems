
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

class firebase:
    "Firebase sending and retrieval happens here"
    def __init__(self):
        super().__init__()
        # initialize the app only once
        if not firebase_admin._apps:
            cred = credentials.Certificate("./firebase/fyp-movement-prediction-firebase-adminsdk-cgzow-495f7076cd.json")
            # firebase_admin.initialize_app(cred)

            # Initialize the app with a service account, granting admin privileges
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://fyp-movement-prediction.firebaseio.com/'
            })

    def send(self, child, data):
        ref = db.reference()
        log_ref = ref.child(child)
        log_ref.push(data)
    
    def getData(self):
        ref = db.reference('dataLogs')
        print(ref.get())
    
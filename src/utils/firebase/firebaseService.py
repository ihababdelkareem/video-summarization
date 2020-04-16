import firebase_admin
from firebase_admin import credentials, firestore
import os
import sys
sys.path.append(os.path.dirname(__file__).replace('video-summarization/src','',1))
from src.utils.inputVideo import InputVideo
import uuid

class FirebaseService:
    def __init__(self):
        self.cred = credentials.Certificate("src/utils/firebase/key.json")
        firebase_admin.initialize_app(self.cred)
        self.db = firestore.client()

    def storeVideo(self,video):
        keyframes = video.getKeyframeList()
        name = video.getVideoName()
        length_in_seconds = video.getLengthInSeconds()
        video_id = str(uuid.uuid4())
        doc_ref = self.db.collection('videos').document(video_id)
        keyframes_to_store = [kf.getStorageObject() for kf in keyframes]
        doc_ref.set({
        'name':name,
        'length':length_in_seconds,
        'keyframes': keyframes_to_store
        })

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2 as cv
from scipy import spatial
import os
import random
class KerasModel:
    def __init__(self):
     print("Loading Keras Model")
     self.model = InceptionV3(weights='imagenet', include_top=False)
     print("Model Loaded")

    def getFeatureVector(self,img1,log=True):
        cv.imwrite('a.jpg',img1)
        a = image.load_img('a.jpg', target_size=(224, 224))
        a_data = image.img_to_array(a)
        a_data = np.expand_dims(a_data,axis=0)
        a_data = preprocess_input(a_data)
        a_feature = np.array(self.model.predict(a_data)).flatten() # fc layer vector: 2048
        if(log):
            print('.')
        
        os.remove('a.jpg')
        return a_feature

    def getFeatureVectorList(self,img_list):
     res = []
     for i in img_list:
        feat = self.getFeatureVector(i,log=False)
        res.append(feat)
     return res



    def getPairDifference(self,vec1,vec2):
        return 0.5*(2 - spatial.distance.cosine(vec1,vec2)) # cosine sim between a and b

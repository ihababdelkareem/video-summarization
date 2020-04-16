
from keras import backend as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
import scipy.spatial as spatial
import cv2 as cv
import random
import os
class DeepRankingModel:
    def __init__(self):
        print("XXXX")
        self.model = self.deep_rank_model()
        print("yyy")
        self.model.load_weights('src/utils/deep_ranking/DR.h5')
        print("zzz")

    def convnet_model_(self):
        vgg_model = VGG16(weights=None, include_top=False)
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
        convnet_model = Model(inputs=vgg_model.input, outputs=x)
        return convnet_model

    def convnet_model(self):
    	vgg_model = VGG16(weights=None, include_top=False)
    	x = vgg_model.output
    	x - GlobalAveragePooling2D()(x)
    	x = Dense(4096, activation='relu')(x)
    	x = Dropout(0.6)(x)
    	x = Dense(4096, activation='relu')(x)
    	x = Dropout(0.6)(x)

    def deep_rank_model(self):

        convnet_model = self.convnet_model_()
        first_input = Input(shape=(224,224,3))
        first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
        first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
        first_max = Flatten()(first_max)
        first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

        second_input = Input(shape=(224,224,3))
        second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
        second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
        second_max = Flatten()(second_max)
        second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

        merge_one = concatenate([first_max, second_max])

        merge_two = concatenate([merge_one, convnet_model.output])
        emb = Dense(4096)(merge_two)
        l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

        final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

        return final_model

    def getFeatureVectorList(self,img_list):
     res = []
     for i in img_list:
      ri = random.randint(1,10000000)
      cv.imwrite('{}.jpg'.format(ri),i)
      image1 = load_img('{}.jpg'.format(ri))
      image1 = img_to_array(image1).astype("float64")
      image1 = transform.resize(image1, (224, 224))
      image1 *= 1. / 255
      image1 = np.expand_dims(image1, axis = 0)
      embedding1 = self.model.predict([image1, image1, image1])[0]
      res.append(embedding1)
      os.remove('{}.jpg'.format(ri))
     return res

    def getPairDifference(self,vec1,vec2):
        return 0.5*(2 - spatial.distance.cosine(vec1,vec2)) # cosine sim between a and b

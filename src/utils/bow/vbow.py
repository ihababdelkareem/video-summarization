import cv2
import numpy as np
from glob import glob
import argparse
from src.utils.bow.helpers import *
from matplotlib import pyplot as plt
import scipy

class BOV:
    def __init__(self, no_clusters,images):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = images
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel(self):
        """
        This method contains the entire module
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """
        self.trainImageCount = len(self.images)
        for im in self.images:
            kp, des = self.im_helper.features(im)
            self.descriptor_list.append(des)
        #empty_des = sum(x is None for x in self.descriptor_list)
        #self.descriptor_list.remove(None)
        bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(n_images = self.trainImageCount, descriptor_list=self.descriptor_list)
        self.bov_helper.standardize()

    def getImageVocab(self,image):
        kp, des = self.im_helper.features(image)
        # print kp

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(self.no_clusters)]])
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)
        # print test_ret

        # print vocab
        for each in test_ret:
            vocab[0][each] += 1
        vocab = self.bov_helper.scale.transform(vocab)
        return vocab[0]

    def getPairDifferenceFromVocabsOfFrames(self,frame_a,frame_b,metric):
        vocab_a,vocab_b = self.getImageVocab(frame_a) , self.getImageVocab(frame_b)
        def getMappedCosinesim(a,b):
            return 0.5*(2 - (scipy.spatial.distance.cosine(a,b))) # maps (-1 to 1) -> (0 to 1)
        difference_metric_mapping = {
        'cosine': getMappedCosinesim # more to be added for pdf diffs
        }
        return difference_metric_mapping[metric](vocab_a,vocab_b)

# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import glob

import numpy as np
import time
from datetime import datetime

from scipy.spatial import distance
from math import*
from cv2 import cv2
from scipy import spatial

from src.utils.preTrained.feature_extractor.feature_extractor import FeatureExtractor
import src.utils.preTrained.feature_extractor.utils as utils




def feature_extraction_queue(feature_extractor, image_path, layer_names,
                             batch_size, num_classes, num_images=100000):
    '''
    Given a directory containing images, this function extracts features
    for all images. The layers to extract features from are specified
    as a list of strings. First, we seek for all images in the directory,
    sort the list and feed them to the filename queue. Then, batches are
    processed and features are stored in a large object `features`.

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param layer_names: list of str, list of layer names
    :param batch_size: int, batch size
    :param num_classes: int, number of classes for ImageNet (1000 or 1001)
    :param num_images: int, number of images to process (default=100000)
    :return:
    '''

    # Add a list of images to process, note that the list is ordered.
    image_files = utils.find_files(image_path, ("jpg", "png"))
    num_images = min(len(image_files), num_images)
    image_files = image_files[0:num_images]

    num_examples = len(image_files)
    num_batches = int(np.ceil(num_examples/batch_size))

    # Fill-up last batch so it is full (otherwise queue hangs)
    utils.fill_last_batch(image_files, batch_size)

    print("#"*80)
    print("Batch Size: {}".format(batch_size))
    print("Number of Examples: {}".format(num_examples))
    print("Number of Batches: {}".format(num_batches))

    # Add all the images to the filename queue
    feature_extractor.enqueue_image_files(image_files)

    # Initialize containers for storing processed filenames and features
    feature_dataset = {'filenames': []}
    for i, layer_name in enumerate(layer_names):
        layer_shape = feature_extractor.layer_size(layer_name)
        layer_shape[0] = len(image_files)  # replace ? by number of examples
        feature_dataset[layer_name] = np.zeros(layer_shape, np.float32)
        print("Extracting features for layer '{}' with shape {}".format(layer_name, layer_shape))

    print("#"*80)

    # Perform feed-forward through the batches
    for batch_index in range(num_batches):

        t1 = time.time()

        # Feed-forward one batch through the network
        outputs = feature_extractor.feed_forward_batch(layer_names)

        for layer_name in layer_names:
            start = batch_index*batch_size
            end   = start+batch_size
            feature_dataset[layer_name][start:end] = outputs[layer_name]

        # Save the filenames of the images in the batch
        feature_dataset['filenames'].extend(outputs['filenames'])

        t2 = time.time()
        examples_in_queue = outputs['examples_in_queue']
        examples_per_second = batch_size/float(t2-t1)

        print("[{}] Batch {:04d}/{:04d}, Batch Size = {}, Examples in Queue = {}, Examples/Sec = {:.2f}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), batch_index+1,
            num_batches, batch_size, examples_in_queue, examples_per_second
        ))

    # If the number of pre-processing threads >1 then the output order is
    # non-deterministic. Therefore, we order the outputs again by filenames so
    # the images and corresponding features are sorted in alphabetical order.
    if feature_extractor.num_preproc_threads > 1:
        utils.sort_feature_dataset(feature_dataset)

    # We cut-off the last part of the final batch since this was filled-up
    feature_dataset['filenames'] = feature_dataset['filenames'][0:num_examples]
    for layer_name in layer_names:
        feature_dataset[layer_name] = feature_dataset[layer_name][0:num_examples]

    return feature_dataset


################################################################################
################################################################################
def getAdjacentMatch(frame_list):

    i = 1
    for f in frame_list:
        cv2.imwrite('./utils/preTrained/images_dir/' +str(i)+ '.jpg', f)
        i = i + 1


    network_name = 'inception_v4'
    checkpoint = './utils/preTrained/checkpoints/inception_v4.ckpt'
    image_path = './utils/preTrained/images_dir/'
    out_file = './features.h5'
    layer_names = 'Logits'
    preproc_func = None
    num_preproc_threads = 2
    batch_size = 64
    num_classes = 1001

    # resnet_v2_101/logits,resnet_v2_101/pool4 => to list of layer names
    layer_names = layer_names.split(",")

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(
        network_name=network_name,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        num_classes=num_classes,
        preproc_func_name=preproc_func,
        preproc_threads=num_preproc_threads
    )

    # Print the network summary, use these layer names for feature extraction
    # feature_extractor.print_network_summary()

    # Feature extraction example using a filename queue to feed images
    feature_dataset = feature_extraction_queue(
        feature_extractor, image_path, layer_names,
        batch_size, num_classes)

    # Write features to disk as HDF5 file
    utils.write_hdf5(out_file, layer_names, feature_dataset)
    print("Successfully written features to: {}".format(out_file))

    # Close the threads and close session.
    feature_extractor.close()
    print("Finished.")

    featuresList = feature_dataset['Logits']

    diffList = []
    for i in range(len(featuresList)-1):
        diffList.append(cosine_similarity(featuresList[i], featuresList[i+1]))
        print(featuresList[i])
        print(featuresList[i+1])
    rmFiles
    return diffList

################################################################################

def rmFiles():
    files = glob.glob('./utils/preTrained/images_dir/*')
    for f in files:
        os.remove(f)

def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def euclidian_distance(x, y):
    return distance.euclidean(x, y)



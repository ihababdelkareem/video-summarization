import cv2
import numpy as np
import math
def returnFeatureDetectionAlgorithm(algoName):
    if(algoName == 'SIFT'):
        return cv2.xfeatures2d.SIFT_create()
    if(algoName == 'SURF'):
        return  cv2.xfeatures2d.SURF_create()
    if(algoName == 'ORB'):
        return cv2.ORB_create(nfeatures=1500)
    return None

def getKeyPoints_Descriptors(algo,img):
    return algo.detectAndCompute(img, None)

def showAndWait(img):
    x=1
    cv2.imshow('im', img)
    cv2.waitKey(0)

def returnMatchingImage(img1, kp1, img2, kp2, matches, k):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:k], None)

def returnSortedMatchesBetweenTwoImages(algoName,des1, des2, knn = True):
    if(algoName == 'SIFT' or algoName == 'SURF'):
        if(knn == True):
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                    good.append(m[0])

            return sorted(good, key=lambda x: x.distance)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            return sorted(matches, key=lambda x: x.distance)

    if(algoName == 'ORB'):
        if(knn == True):
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                    good.append(m[0])
            return sorted(good, key=lambda x: x.distance)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
            matches = bf.match(des1, des2)
            return sorted(matches, key = lambda x:x.distance)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
    reprojThresh,matches):

    newMatches = []

    for m in matches:
      newMatches.append((m.trainIdx,m.queryIdx))

    ptsA = np.float32([kpsA[i] for (_, i) in newMatches])
    ptsB = np.float32([kpsB[i] for (i, _) in newMatches])

    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
    return (newMatches, H, status)

def getFeatureVectorList(imgList,algoName='SIFT'):
    algo = returnFeatureDetectionAlgorithm(algoName)
    res = []
    for img in imgList:
      kps, des = getKeyPoints_Descriptors(algo, img)
      res.append((kps,des))
    return res

def getAdjacentMatch(kps1,des1,kps2,des2,algoName='SURF',knn=False):
    algo = returnFeatureDetectionAlgorithm(algoName)
    matches = returnSortedMatchesBetweenTwoImages(algoName, des2, des1, knn)
    print('KPS1: {} , KPS2: {}, matches: {}'.format(len(kps1),len(kps2),len(matches)))
    return len(kps2)/len(kps1)

def getMatchingImage(img1,img2,algoName='SIFT',knn=False):
    algo = returnFeatureDetectionAlgorithm(algoName)
    kps1,des1 = getKeyPoints_Descriptors(algo, img1)
    kps2,des2 = getKeyPoints_Descriptors(algo, img2)
    matches = returnSortedMatchesBetweenTwoImages(algoName, des2, des1, knn)
    return returnMatchingImage(img2, kps2, img1, kps1, matches, 100)

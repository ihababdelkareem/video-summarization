import cv2 as cv

def hsvColorHist(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

def hsvColorHistEvaluation(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return cv.calcHist([hsv], [0, 1,2], None, [16,16,16], [0, 180, 0, 256, 0, 256]).flatten()

def hsvHistsForSeriesOfImages(images):
    hsv_images = []
    for image in images:
        hsv_images.append(hsvColorHistEvaluation(image))
    return hsv_images

def compareHists(hist1, hist2, method = 'correlation'):
    if(method == 'correlation'):
        method = cv.HISTCMP_CORREL
    elif(method == 'chi_square'):
        method = cv.HISTCMP_CHISQR
    elif(method == 'intersect'):
        method = cv.HISTCMP_INTERSECT
    else:
        method = cv.HISTCMP_BHATTACHARYYA
    return cv.compareHist(hist1, hist2, method)

def adjacentHistComparison(hist_array, method):
    hist_differences = []
    for i in range(len(hist_array) - 1):
        hist_differences.append(compareHists(hist_array[i], hist_array[i +1], method))
    return hist_differences

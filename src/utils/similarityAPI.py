
import requests
import cv2 as cv
def getSimilarity(img1,img2):
    cv.imwrite('1.jpg',img1)
    cv.imwrite('2.jpg',img2)
    r = requests.post(
        "https://api.deepai.org/api/image-similarity",
        files={
            'image1': open('1.jpg', 'rb'),
            'image2': open('2.jpg','rb'),
        },
        headers={'api-key': 'c8542f42-16e3-47ae-9a96-3bbabd1d5b5b'}
    )
    print(r.json())

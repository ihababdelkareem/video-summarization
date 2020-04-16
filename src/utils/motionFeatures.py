import cv2
import numpy as np

def showOF(video):
#cap = cv2.VideoCapture("C://Users//bader y. anini//Documents//GitHub//video-summarization//database//database//v22.mpg")

    frame1 = video.getNthFrame(0)
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    i = 1
    prvsRGB = frame1
    while(1):
        frame2 = video.getNthFrame(i)
        i = i+1
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame1', frame2)
        cv2.imshow('frame3', prvsRGB)
        k = cv2.waitKey(0) & 0xff
        # if k == 32:
        #     cv2.imshow('frame2', rgb)
        # k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
        prvsRGB = frame2

    cv2.destroyAllWindows()